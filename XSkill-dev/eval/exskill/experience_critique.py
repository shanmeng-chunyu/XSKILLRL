import os
import json
import time
import re
import ast
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image

from prompts.experience_prompts import (
    INTRA_SAMPLE_CRITIQUE,
)

from .llm_client import ExperienceLLM


# --------- Constants ---------

# Retry configuration
MAX_RETRIES = 3

# Token configuration
MAX_TOKENS = 12288


def _normalize_experience_op(item) -> Optional[dict]:
    if isinstance(item, str):
        text = item.strip().strip("-").strip()
        if not text:
            return None
        return {"option": "add", "experience": text}

    if not isinstance(item, dict):
        return None

    exp_txt = (
        item.get("experience")
        or item.get("exp")
        or item.get("text")
        or item.get("lesson")
        or item.get("tip")
        or item.get("rule")
        or item.get("content")
        or item.get("new_experience")
        or item.get("modified_experience")
    )
    if isinstance(exp_txt, list):
        exp_txt = " ".join(str(part).strip() for part in exp_txt if str(part).strip())
    if not isinstance(exp_txt, str) or not exp_txt.strip():
        return None

    option = str(item.get("option") or item.get("action") or item.get("type") or "add").strip().lower()
    if option in ("new", "create", "insert", "addition", "execution_tip", "decision_rule", "tip", "rule"):
        option = "add"
    if option not in ("add", "modify"):
        option = "add"

    op = {"option": option, "experience": exp_txt.strip()}
    modified_from = item.get("modified_from") or item.get("id") or item.get("experience_id")
    if option == "modify" and modified_from:
        op["modified_from"] = str(modified_from).strip()
    return op


def _collect_ops_from_parsed(parsed) -> List[dict]:
    if isinstance(parsed, list):
        ops = []
        for item in parsed:
            op = _normalize_experience_op(item)
            if op:
                ops.append(op)
        return ops

    if not isinstance(parsed, dict):
        op = _normalize_experience_op(parsed)
        return [op] if op else []

    direct = _normalize_experience_op(parsed)
    if direct:
        return [direct]

    ops = []
    for key in ("experiences", "experience", "operations", "updates", "add", "execution_tips", "decision_rules", "tips", "rules"):
        value = parsed.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, list):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            op = _normalize_experience_op(item)
            if op:
                ops.append(op)
    return ops


def _fallback_ops_from_text(resp: str, max_ops: int) -> List[dict]:
    lines = []
    for raw_line in resp.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if len(line) < 20 or len(line) > 260:
            continue
        lowered = line.lower()
        if any(skip in lowered for skip in ("```", "<question>", "</", "score:", "explanation:", "analysis framework", "trajectory review")):
            continue
        if any(trigger in lowered for trigger in ("when ", "if ", "for ", "use ", "check ", "avoid ", "verify ", "search ", "compare ", "inspect ")):
            lines.append(line)
        if len(lines) >= max_ops:
            break
    return [{"option": "add", "experience": line} for line in lines[:max_ops]]


def _parse_experience_ops_response(resp: str, max_ops: int = 2) -> List[dict]:
    if not isinstance(resp, str) or not resp.strip():
        return []

    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", resp, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced if block.strip())

    stripped = resp.strip()
    candidates.append(stripped)

    list_match = re.search(r"\[\s*\{.*?\}\s*\]", resp, flags=re.DOTALL)
    if list_match:
        candidates.append(list_match.group(0))

    object_match = re.search(r"\{\s*\"(?:experiences|operations|updates)\"\s*:\s*.*\}", resp, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    seen = set()
    for payload in candidates:
        if not payload or payload in seen:
            continue
        seen.add(payload)
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(payload)
            except Exception:
                continue
            ops = _collect_ops_from_parsed(parsed)
            if ops:
                return ops[:max_ops]

    return _fallback_ops_from_text(resp, max_ops=max_ops)


# --------- Intra-sample critique ---------

def intra_sample_experiences(question: str, groundtruth: str, summaries: Dict[str, str], llm: ExperienceLLM, max_ops: int = 2, debug_dir: Optional[str] = None, system_prompt: Optional[str] = None, used_experiences: Optional[Dict[str, str]] = None) -> List[dict]:
    """Generate experience critiques from trajectory summaries.
    
    Args:
        question: The question/task being solved
        groundtruth: Ground truth answer (if available)
        summaries: Dictionary mapping node_id to summary text
        llm: ExperienceLLM instance for generating critiques
        max_ops: Maximum number of operations to generate
        debug_dir: Optional directory to save debug information
        system_prompt: Optional system prompt used in the task
        used_experiences: Optional dict mapping experience ID to original content 
                         (experiences that were retrieved and used for this sample)
    
    Returns:
        List of experience operation dictionaries
    """
    if not summaries:
        return []
    formatted_summaries = []
    for i, (nid, summ) in enumerate(summaries.items()):
        formatted_summaries.append(f"Trajectory {i+1} (node {nid}):\n{summ}")
    
    # Format question section with system prompt (matching summary format)
    question_section_parts = []
    if isinstance(question, str) and question.strip():
        question_section_parts.append(question.strip())
    if isinstance(system_prompt, str) and system_prompt.strip():
        question_section_parts.append(f"\n==== System prompt:\n{system_prompt.strip()}")
    
    formatted_question = "\n".join(question_section_parts) if question_section_parts else ""
    
    # Format used experiences (original content of experiences that were used for this sample)
    formatted_experiences = ""
    if used_experiences:
        exp_lines = []
        for exp_id, exp_content in used_experiences.items():
            exp_lines.append(f"[{exp_id}] {exp_content}")
        formatted_experiences = "\n\n".join(exp_lines)
    
    # Selection logic for prompt template (always use ground truth if available)
    prompt = INTRA_SAMPLE_CRITIQUE.format(
        max_ops=max_ops,
        question=formatted_question,
        summaries="\n\n".join(formatted_summaries),
        experiences=formatted_experiences or "(No experiences were retrieved for this sample)",
        groundtruth=groundtruth or "[REDACTED]",
    )
    
    # Find question images (top-level original_image files in debug_dir)
    question_images = []
    if debug_dir:
        sample_path = Path(debug_dir)
        if sample_path.exists() and sample_path.is_dir():
            for img_file in sample_path.glob("original_image*"):
                if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    question_images.append(img_file)
    
    # Add retry mechanism for API calls
    resp = None
    llm_start = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            # Use chat_with_image if question images are available
            if question_images:
                pil_images = [Image.open(img_path).convert('RGB') for img_path in question_images]
                resp = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images if len(pil_images) > 1 else pil_images[0],
                    max_tokens=MAX_TOKENS,
                    return_placeholder_on_error=False
                )
            else:
                # Fallback to text-only
                resp = llm.chat(prompt, max_tokens=MAX_TOKENS)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Warning: Experience critique failed (attempt {attempt+1}/{MAX_RETRIES}): {e}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    llm_time = time.time() - llm_start
    print(f"  [Timing] cross rollout critique generation: {llm_time:.1f}s")
    if resp is None:
        raise RuntimeError("Failed to generate experiences after all retries")
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "exp_intra_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(os.path.join(debug_dir, "exp_intra_resp.txt"), "w", encoding="utf-8") as f:
                f.write(resp)
        except Exception:
            pass
    ops = _parse_experience_ops_response(resp, max_ops=max_ops)
    if not ops:
        debug_path = os.path.join(debug_dir, "exp_intra_resp.txt") if debug_dir else "<debug_dir not set>"
        print(f"  Warning: Failed to parse experience operations; raw response saved at {debug_path}")
    return ops
