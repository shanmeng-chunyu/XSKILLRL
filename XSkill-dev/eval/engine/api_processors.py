"""
Processor functions for API-based single sample inference.
"""

import os
import json
import re
from qwen_vl_utils import fetch_image # type: ignore
from utils.function_call_parser import parse_function_call_response
from utils.context_utils import process_image
from utils.context_utils import pil_to_base64_data_uri
from utils.context_utils import estimate_tokens
from utils.llm_judger import compute_score
from utils.result_utils import save_trajectory
from search.tree import SearchNode
from engine.api_model_caller import create_model_caller

SYSTEM_PROMPT = ""


def _resolve_image_source(image_folder, image_ref):
    image_ref = str(image_ref)
    if image_ref.startswith(("http://", "https://", "data:image/")):
        return image_ref
    if os.path.isabs(image_ref):
        return image_ref
    return os.path.join(image_folder, image_ref)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _source_urls(sample):
    urls = []
    for value in _as_list(sample.get("source")):
        text = str(value or "").strip()
        if text and text not in urls:
            urls.append(text)
    return urls


def _append_source_urls(question, sample):
    urls = _source_urls(sample)
    if not urls:
        return question
    question_text = str(question or "").strip()
    existing = question_text.lower()
    missing_urls = [url for url in urls if url.lower() not in existing]
    if not missing_urls:
        return question_text
    source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(missing_urls, start=1))
    source_section = (
        "Source URLs provided by the benchmark. Use the visit tool on these URLs "
        "when they are relevant:\n"
        f"{source_lines}"
    )
    if question_text:
        return f"{question_text}\n\n{source_section}"
    return source_section


def set_global_prompts(system_prompt):
    """
    Set global prompt variable for API processors.
    
    Args:
        system_prompt: System prompt string
    """
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = system_prompt


# ============================================================================
# Helper Functions
# ============================================================================

def parse_and_load_multiple_images(sample, args, save_dir):
    """
    Parse <image> placeholders from question text and load all images.
    Supports both single and multiple images with backward compatibility.
    
    Args:
        sample: Data sample with 'problem'/'question' and 'images' fields
        args: Command line arguments with image_folder
        save_dir: Directory to save images
        
    Returns:
        tuple: (image_map, image_list, question_text, user_content_list)
            - image_map: dict mapping image names to PIL.Image objects
            - image_list: list of (image_name, PIL.Image) tuples in order
            - question_text: question text with <image> placeholders removed
            - user_content_list: list of content items for API message (images + text)
        Returns (None, None, None, None) if image loading fails
    """
    question = _append_source_urls(sample.get('problem', sample.get('question', '')), sample)
    image_paths = sample.get('images', [])
    
    # Count <image> placeholders in question
    image_placeholder_pattern = r'<image>'
    placeholder_count = len(re.findall(image_placeholder_pattern, question, re.IGNORECASE))
    
    # Determine how many images we need
    # Use placeholder count if available, otherwise use number of images in sample
    # If neither exists, default to 1 (backward compatibility)
    if placeholder_count > 0:
        num_images_needed = placeholder_count
    elif len(image_paths) > 0:
        num_images_needed = len(image_paths)
    else:
        num_images_needed = 0  # Text-only sample (no placeholders, no images)

    num_images_to_load = min(num_images_needed, len(image_paths)) if len(image_paths) > 0 else 0
    if num_images_to_load == 0 and num_images_needed > 0:
        print(f"Warning: Need {num_images_needed} image(s) but none provided in sample")
        return None, None, None, None
    
    # Load images
    image_map = {}
    image_list = []
    user_content_list = []
    
    for i in range(num_images_to_load):
        image_path = _resolve_image_source(args.image_folder, image_paths[i])
        # Naming convention: first image is 'original_image', others are 'original_image_1', 'original_image_2', etc.
        image_name = f'original_image_{i}' if i > 0 else 'original_image'
        
        try:
            original_image = fetch_image({'image': image_path, 'max_pixels': args.max_pixels})
            processed_image = process_image(original_image, args.max_pixels, args.min_pixels)
            # Store original PIL object for potential cropping
            image_map[image_name] = original_image
            image_list.append((image_name, original_image))
            
            # Add to user content for API message
            user_content_list.append({
                "type": "image_url",
                "image_url": {"url": pil_to_base64_data_uri(original_image)}
            })
            
            os.makedirs(save_dir, exist_ok=True)
            image_file_path = os.path.join(save_dir, "original_image.jpg") if i == 0 else os.path.join(save_dir, f"original_image_{i}.jpg")
            processed_image.save(image_file_path)
            
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None, None, None
    
    # Remove <image> placeholders from question text
    question_text = re.sub(image_placeholder_pattern, '', question, flags=re.IGNORECASE).strip()
    # Clean up any extra whitespace/newlines
    question_text = re.sub(r'\n+', '\n', question_text).strip()
    
    # Add text content at the end
    if question_text:
        user_content_list.append({"type": "text", "text": question_text})
    
    return image_map, image_list, question_text, user_content_list


def _initialize_sample_and_image(sample, args, save_dir):
    """
    Initialize sample data and load/process images (supports multiple images).
    
    Returns:
        tuple: (question_id, question, ground_truth, image_map, image_list, question_text, user_content_list)
        Returns None if image loading fails
    """
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    question_id = str(question_id if question_id is not None else "N/A")
    question = sample.get('problem', sample.get('question', ''))
    ground_truth = (
        sample.get("solution")
        or sample.get("answer")
        or sample.get("ground_truth")
        or sample.get("label")
    )
    
    # Save initial trajectory info
    initial_traj_info = {
        "doc_id": question_id,
        "initial_prompt": question,
        "ground_truth": ground_truth
    }
    save_trajectory(save_dir,initial_traj_info)
    
    # Parse and load multiple images
    image_map, image_list, question_text, user_content_list = parse_and_load_multiple_images(sample, args, save_dir)
    if image_map is None:
        return None
    
    # Save injected system prompt for observability
    try:
        with open(os.path.join(save_dir, 'injected_system_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(SYSTEM_PROMPT)
    except Exception:
        pass
    
    return question_id, question, ground_truth, image_map, image_list, question_text, user_content_list


def _create_initial_search_node(question_text, image_map, image_list, user_content_list, system_prompt):
    """
    Create initial SearchNode with conversation history and image map.
    Supports both single and multiple images.
    
    Args:
        question_text: Question text (with <image> placeholders removed)
        image_map: dict mapping image names to PIL.Image objects
        image_list: list of (image_name, PIL.Image) tuples in order
        user_content_list: list of content items for API message (images + text)
        system_prompt: System prompt string
        
    Returns:
        SearchNode: Initialized search node
    """
    # Build text conversation history with image references
    # Format: "[Image: name1]\n[Image: name2]\n{question_text}"
    image_refs = [f"[Image: {name}]" for name, _ in image_list]
    user_content_text = "\n".join(image_refs) + (f"\n{question_text}" if question_text else "")
    
    # Build API conversation history
    # Use user_content_list if provided (contains images + text), otherwise fallback to text only
    if user_content_list:
        api_user_content = user_content_list
    else:
        api_user_content = [{"type": "text", "text": question_text}] if question_text else []
    
    return SearchNode(
        conversation_history=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_text}
        ],
        api_conversation_history=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": api_user_content
            }
        ],
        image_map=image_map.copy(),
        current_turn=0,
        current_token_count=estimate_tokens(system_prompt + question_text)
    )


def _extract_responses_from_node(node):
    """
    Extract conversation history and assistant responses from node.
    
    Returns:
        tuple: (conversation_history, assistant_responses)
    """
    conversation_history = node.conversation_history
    assistant_responses = [
        msg['content'] for msg in conversation_history
        if msg['role'] == 'assistant'
    ]
    return conversation_history, assistant_responses


def _looks_like_non_answer(text):
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True, "empty final answer"
    if lowered.startswith("error:"):
        return True, "final answer is an error"
    non_answer_markers = [
        "<tool_call",
        "</tool_call>",
        '"tool_name"',
        '"name"',
        "code_interpreter",
        "web_search",
        "image_search",
        "visit",
        "zoom",
        "reached max turns",
        "without a definitive answer",
        "could not parse model response",
        "api attempts failed",
    ]
    if any(marker in lowered for marker in non_answer_markers):
        return True, "final answer is a tool call, parser failure, or unfinished rollout"
    return False, ""


def _extract_final_answer_for_scoring(final_answer):
    """Return a concrete answer string for scoring, or (None, reason)."""
    text = str(final_answer or "").strip()
    if not text:
        return None, "empty final answer"
    if text.lower().startswith("error:"):
        return None, "final answer is an error"

    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_matches:
        candidate = answer_matches[-1].strip()
        is_non_answer, reason = _looks_like_non_answer(candidate)
        return (None, reason) if is_non_answer else (candidate, "")

    final_match = re.search(r"(?:final\s+answer|answer)\s*(?::|\uff1a)\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        candidate = final_match.group(1).strip()
        is_non_answer, reason = _looks_like_non_answer(candidate)
        return (None, reason) if is_non_answer else (candidate, "")

    is_non_answer, reason = _looks_like_non_answer(text)
    if is_non_answer:
        return None, reason

    return text, ""


def _evaluate_trajectory(question, ground_truth, conversation_history, assistant_responses, question_id, final_answer=None):
    """
    Evaluate trajectory and compute scores.
    
    Returns:
        tuple: (accuracy_score, trajectory_score, trajectory_analysis, trajectory_text)
    """
    accuracy_score = 0.0
    trajectory_score = 0.0
    trajectory_analysis = ""
    trajectory_text = ""
    
    # Generate trajectory text
    for message in conversation_history:
        trajectory_text += f"**{message['role']}**: {message['content']}\n\n"
    
    has_verifier_endpoint = bool(
        os.environ.get("VERIFIER_END_POINTS")
        or os.environ.get("VERIFIER_END_POINT")
        or os.environ.get("REASONING_END_POINTS")
        or os.environ.get("REASONING_END_POINT")
    )
    if has_verifier_endpoint:
        if ground_truth:
            prediction_for_scoring, blocked_reason = _extract_final_answer_for_scoring(final_answer)
            if not prediction_for_scoring:
                trajectory_analysis = (
                    "Evaluation forced to 0.0 because the rollout did not produce a concrete final answer: "
                    f"{blocked_reason}."
                )
                return accuracy_score, trajectory_score, trajectory_analysis, trajectory_text

            extra_info = {
                "acc_reward_weight": 1.0,
                "gpt_extract_answer": True,
                "extract_answer_tags": "strict",
            }
            try:
                accuracy_score, analysis = compute_score(
                    prompt=question,
                    predict_str_list=[prediction_for_scoring],
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )
                trajectory_score = accuracy_score
                trajectory_analysis = analysis if analysis else "Evaluation completed."
            except Exception as e:
                print(f"Error during evaluation for {question_id}: {e}")
                accuracy_score = 0.0
                trajectory_score = 0.0
                trajectory_analysis = f"Evaluation failed: {str(e)}"
        else:
            trajectory_analysis = "Evaluation skipped: missing ground truth."
    else:
        print("VERIFIER endpoint not found in environment variables. Skipping evaluation.")
        trajectory_analysis = "Evaluation skipped: verifier endpoint not configured."
    
    return accuracy_score, trajectory_score, trajectory_analysis, trajectory_text


def _build_result_dict(question_id, question, final_answer, ground_truth, conversation_history,
                       accuracy_score, trajectory_text, trajectory_score, trajectory_analysis):
    """
    Build result dictionary with consistent format.
    """
    return {
        "question_id": question_id,
        "prompt": question,
        "final_answer": final_answer,
        "ground_truth": ground_truth,
        "conversation_history": conversation_history,
        "accuracy_score": accuracy_score,
        "trajectory_text": trajectory_text,
        "trajectory_score": trajectory_score,
        "trajectory_analysis": trajectory_analysis,
    }


def _ensure_recoverable_trajectory(save_dir, result):
    """Create a minimal traj.jsonl when a rollout returned but trajectory logging is missing."""
    traj_path = os.path.join(save_dir, "traj.jsonl")
    if os.path.exists(traj_path):
        return

    try:
        save_trajectory(save_dir, {
            "doc_id": result.get("question_id"),
            "initial_prompt": result.get("prompt", ""),
            "ground_truth": result.get("ground_truth"),
            "recovered": True,
        })
        turn_idx = 0
        for message in result.get("conversation_history", []):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            save_trajectory(save_dir, {
                "turn_idx": turn_idx,
                "text_output": message.get("content", ""),
                "recovered": True,
            })
            turn_idx += 1
        if turn_idx == 0:
            save_trajectory(save_dir, {
                "turn_idx": 0,
                "text_output": result.get("final_answer", ""),
                "recovered": True,
            })
        print(f"[Trajectory Recovery] Recreated missing traj.jsonl at {traj_path}")
    except Exception as exc:
        print(f"[Trajectory Recovery] Failed to recreate traj.jsonl at {traj_path}: {exc}")


# ============================================================================
# Core Processing Functions
# ============================================================================

def _run_greedy_loop(current_node, model_caller, args, sampling_params, question_text, save_dir):
    """
    Run greedy inference loop with support for reflection.
    
    Returns:
        SearchNode: Final node after greedy inference
    """
    final_answer = None
    
    for turn in range(args.max_turns):
        # Check termination conditions
        if len(current_node.image_map) >= args.max_images:
            current_node.mark_final("Error: Reached max image limit.")
            break
        
        if current_node.current_token_count >= args.max_total_tokens - sampling_params['max_tokens']:
            current_node.mark_final("Error: Reached max token limit.")
            break
        
        # Call model
        try:
            response_text = model_caller(current_node)
        except Exception as e:
            print(f"Error calling model at turn {turn}: {e}")
            import traceback
            traceback.print_exc()
            current_node.mark_final(f"Error: Exception in model call: {str(e)}")
            # Save error to trajectory
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": f"Error: Exception in model call: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            break
        
        # Check if response is an error string from API
        if isinstance(response_text, str) and response_text.startswith("Error:"):
            error_msg = response_text
            print(f"Warning: {error_msg} at turn {turn}")
            current_node.mark_final(error_msg)
            # Save error to trajectory with detailed error info
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": error_msg,
                "error": error_msg,
                "error_type": "api_call_failed"
            })
            break
        
        if response_text is None:
            error_msg = "Error: Empty response from model."
            print(f"Warning: {error_msg} at turn {turn}")
            current_node.mark_final(error_msg)
            # Save error to trajectory
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": error_msg,
                "error": "Empty response from model"
            })
            break
        
        # Check if model_caller executed a tool call
        tool_call_executed = False
        if current_node.api_conversation_history:
            # Find the last assistant message
            for msg in reversed(current_node.api_conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    # If this assistant message has tool_calls, tool call was executed
                    if "tool_calls" in msg and msg["tool_calls"]:
                        tool_call_executed = True
                    break  # Only check the last assistant message
        
        # If a tool call was executed, continue loop to get model's text response based on tool results
        if tool_call_executed:
            # Tool call was executed by model_caller, continue to get model's text response
            continue
        
        # Parse response (only for non-tool-call responses)
        action, data = parse_function_call_response(response_text)
        
        # Handle answer action
        if action == "answer":
            final_answer = current_node.final_answer or data
            current_node.mark_final(final_answer)
            break
        
        if action == "text" and isinstance(data, str) and data.strip() and turn < args.max_turns - 1:
            correction = (
                "Your previous message described a plan but did not execute it. "
                "Continue now by either calling exactly one available tool through the tool-calling interface, "
                "or, if you already know the answer, provide the final answer inside complete <answer>...</answer> tags. "
                "Do not only describe the next action."
            )
            current_node.api_conversation_history.append({"role": "user", "content": correction})
            current_node.conversation_history.append({"role": "user", "content": correction})
            save_trajectory(save_dir, {
                "turn_idx": turn,
                "text_output": "[Parser correction] Model returned plain text without a tool call or <answer>; requesting a concrete tool call or final answer.",
                "node_id": getattr(current_node, 'node_id', ''),
            })
            continue

        current_node.mark_final("Error: Could not parse model response.")
        break
    
    # Extract final answer from node
    if current_node.is_final:
        final_answer = current_node.final_answer
    else:
        final_answer = "Error: Reached max turns without a definitive answer."
        current_node.mark_final(final_answer)
    
    return current_node


def _process_single_sample_unified(sample, args, sampling_params):
    """
    Process a single sample using greedy inference.
    
    Args:
        sample: Data sample to process
        args: Command line arguments (args.output_dir used as save_dir)
        sampling_params: Sampling parameters dictionary
        
    Returns:
        Result dictionary with trajectory information, or None if init fails
    """
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize sample and images (supports multiple images)
    init_result = _initialize_sample_and_image(sample, args, save_dir)
    if init_result is None:
        return None
    question_id, question, ground_truth, image_map, image_list, question_text, user_content_list = init_result
    
    # Create initial SearchNode with multiple images support
    current_system_prompt = getattr(args, '_sample_system_prompt', SYSTEM_PROMPT)
    initial_node = _create_initial_search_node(question_text, image_map, image_list, user_content_list, current_system_prompt)

    # Create model caller
    model_caller = create_model_caller(args, sampling_params, save_dir)
    
    try:
        # Run greedy inference loop
        final_node = _run_greedy_loop(
            initial_node, model_caller, args, sampling_params, question_text, save_dir
        )
        final_answer = final_node.final_answer
        
        # Extract responses and evaluate
        conversation_history, assistant_responses = _extract_responses_from_node(final_node)
        accuracy_score, trajectory_score, trajectory_analysis, trajectory_text = _evaluate_trajectory(
            question_text, ground_truth, conversation_history, assistant_responses, question_id, final_answer=final_answer
        )
        
        # Build result dict
        result = _build_result_dict(
            question_id, question_text, final_answer, ground_truth, conversation_history,
            accuracy_score, trajectory_text, trajectory_score, trajectory_analysis
        )
        
        return result
    finally:
        close_fn = getattr(model_caller, 'close', None)
        if callable(close_fn):
            close_fn()


# ============================================================================
# Public API - Backward compatible aliases
# ============================================================================


def process_single_sample(sample, args, sampling_params, rollout_idx=None):
    """
    Process a single sample using greedy inference.
    
    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        rollout_idx: Optional rollout index for multi-rollout mode (0-based)
        
    Returns:
        Result dictionary with trajectory information
    """
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    question_id = str(question_id if question_id is not None else "N/A")
    
    # Determine save directory based on rollout mode
    rollouts_per_sample = getattr(args, 'rollouts_per_sample', 1)
    if rollout_idx is not None and rollouts_per_sample > 1:
        # Multi-rollout mode: save to rollout_N subdirectory
        sample_dir = os.path.join(args.output_dir, question_id)
        save_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
    else:
        # Single rollout mode: save directly to question_id directory
        save_dir = os.path.join(args.output_dir, question_id)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Temporarily override output_dir for processors
    original_output_dir = args.output_dir
    args.output_dir = save_dir
    
    try:
        # Run greedy search
        final_result = _process_single_sample_unified(sample, args, sampling_params)
        
        if final_result:
            _ensure_recoverable_trajectory(save_dir, final_result)
            # Save metrics to the rollout directory
            metrics = {
                "accuracy_score": final_result["accuracy_score"],
                "trajectory_score": final_result.get("trajectory_score"),
                "trajectory_analysis": final_result.get("trajectory_analysis"),
                "final_answer": final_result.get("final_answer"),
                "ground_truth": final_result.get("ground_truth"),
            }
            with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
        
        return final_result
    
    finally:
        # Restore original output_dir
        args.output_dir = original_output_dir
