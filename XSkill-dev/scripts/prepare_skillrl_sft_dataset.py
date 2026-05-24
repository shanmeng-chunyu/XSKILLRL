"""Prepare multimodal SFT data for the SkillRL/verl SFT trainer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.dataset import load_records_from_spec
from xskill_rl.skillrl.skill_bank import SkillsOnlyMemory
from xskill_rl.skillrl.verl_export import DEFAULT_SYSTEM_PROMPT, write_parquet

IMAGE_ONLY_PROMPT = (
    "Please answer the question shown in the image. Provide the final answer "
    "using the required answer format."
)
TOOL_PROTOCOL = """
You may use tools when external information, webpage reading, OCR, image inspection, or calculation is needed.
Call exactly one tool at a time using this format:
<tool_call>{"name":"web_search","arguments":{"query":"search terms","max_results":5}}</tool_call>

Available tools:
- web_search: {"query": "...", "max_results": 5}
- visit: {"url": "https://...", "goal": "what to find"}
- image_search: {"search_type": "text|reverse", "query": "...", "image_url": "original_image or tool_image_N", "max_results": 5}
- code_interpreter: {"code": "python code"}
- zoom: {"code": "python code to crop or inspect images"}

After each tool observation, continue reasoning or call another tool. When ready, give the final answer as <answer>...</answer>.
""".strip()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _iter_input_file(path_like: str) -> Iterable[Dict[str, Any]]:
    path = Path(path_like).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        yield from payload if isinstance(payload, list) else [payload]
    else:
        raise ValueError(f"Unsupported input file type: {path}")


def _extract_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    prompt_candidates: List[Any] = []

    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        sample = extra_info.get("sample")
        if isinstance(sample, dict):
            payload.update(sample)
            if "prompt" in sample:
                prompt_candidates.append(sample["prompt"])

    env_kwargs = record.get("env_kwargs")
    if isinstance(env_kwargs, dict):
        payload.update(env_kwargs)
        if "prompt" in env_kwargs:
            prompt_candidates.append(env_kwargs["prompt"])

    for key in ("prompt", "images", "data_source"):
        if key not in payload and key in record:
            payload[key] = record[key]
    if "prompt" in record:
        prompt_candidates.append(record["prompt"])

    reward_model = record.get("reward_model")
    if isinstance(reward_model, dict) and not payload.get("solution"):
        payload["solution"] = reward_model.get("ground_truth", "")

    if not payload:
        payload.update(record)
    if prompt_candidates:
        payload["_prompt_candidates"] = prompt_candidates
    return payload


def _load_samples(input_specs: Sequence[str], input_files: Sequence[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if input_specs:
        rl_records, _ = load_records_from_spec(input_specs, mixing_strategy="concat", seed=42)
        samples.extend(record["extra_info"]["sample"] for record in rl_records)
    for input_file in input_files:
        samples.extend(_extract_sample(record) for record in _iter_input_file(input_file))
    return samples


def _message_content_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(_as_text(text))
            elif item is not None:
                parts.append(_as_text(item))
        return "\n".join(part for part in parts if part)
    return _as_text(content)


def _problem_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        user_messages = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "user":
                content = _message_content_text(message.get("content", ""))
                if content:
                    user_messages.append(content)
        if user_messages:
            return user_messages[-1]
        return ""

    text = _as_text(prompt).strip()
    if not text:
        return ""
    matches = list(re.finditer(r"(?im)^\s*USER\s*:\s*", text))
    if matches:
        return text[matches[-1].end() :].strip()
    return text


def _extract_problem(sample: Dict[str, Any]) -> str:
    for key in ("problem", "question", "query", "instruction", "task"):
        value = _as_text(sample.get(key)).strip()
        if value:
            return value
    prompt_candidates = sample.get("_prompt_candidates") or [sample.get("prompt")]
    for prompt in prompt_candidates:
        value = _problem_from_prompt(prompt).strip()
        if value:
            return value
    return ""


def _extract_solution(sample: Dict[str, Any]) -> str:
    for key in ("solution", "answer", "ground_truth", "label", "target"):
        value = _as_text(sample.get(key)).strip()
        if value:
            return value
    return ""


def _as_image_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _as_image_list(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        if "image" in value:
            return _as_image_list(value.get("image"))
        for key in ("path", "url"):
            if value.get(key):
                return _as_image_list(value.get(key))
        images: List[Any] = []
        for item in value.values():
            images.extend(_as_image_list(item))
        return images
    if isinstance(value, (list, tuple)):
        images = []
        for item in value:
            images.extend(_as_image_list(item))
        return images
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_images(sample: Dict[str, Any]) -> List[Any]:
    for key in ("images", "image", "image_path", "image_paths"):
        images = _as_image_list(sample.get(key))
        if images:
            return images
    return []


def _build_prompt(
    sample: Dict[str, Any],
    memory: SkillsOnlyMemory | None,
    top_k: int,
    *,
    enable_tools: bool,
) -> str:
    problem = _extract_problem(sample) or IMAGE_ONLY_PROMPT
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if memory is not None:
        retrieval = memory.retrieve(task_description=problem, top_k=top_k, metadata=sample)
        skill_text = memory.format_for_prompt(retrieval)
        if skill_text and skill_text != "No relevant skills found for this task.":
            system_prompt = f"{system_prompt}\n\n## Retrieved Relevant Experience\n{skill_text}"
    if enable_tools:
        system_prompt = f"{system_prompt}\n\n## Tool Protocol\n{TOOL_PROTOCOL}"
    return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{problem}"


def _build_response(sample: Dict[str, Any], response_format: str) -> str:
    solution = _extract_solution(sample)
    if response_format == "answer_tag":
        return f"<answer>{solution}</answer>"
    if response_format == "final_answer":
        return f"Final answer: {solution}"
    return solution


def samples_to_sft_records(
    samples: Sequence[Dict[str, Any]],
    *,
    memory: SkillsOnlyMemory | None,
    top_k: int,
    response_format: str,
    max_records: int | None,
    include_images: bool,
    enable_tools: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if max_records is not None and len(records) >= max_records:
            break
        problem = _extract_problem(sample)
        solution = _extract_solution(sample)
        images = _extract_images(sample)
        if not solution:
            continue
        if not problem and not images:
            continue
        record = {
            "prompt": _build_prompt(sample, memory, top_k, enable_tools=enable_tools),
            "response": _build_response(sample, response_format),
            "extra_info": {
                "index": index,
                "doc_id": _as_text(sample.get("doc_id") or sample.get("question_id") or index),
                "benchmark_name": _as_text(sample.get("benchmark_name") or sample.get("data_source") or "xskill"),
                "has_skill_prompt": memory is not None,
                "has_images": bool(images),
                "num_images": len(images),
                "image_only_prompt": not bool(problem) and bool(images),
                "tool_protocol": bool(enable_tools),
            },
        }
        if include_images:
            record["images"] = images
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SkillRL/verl multimodal SFT parquet data from XSkill samples.")
    parser.add_argument("--input-spec", action="append", default=[])
    parser.add_argument("--input-file", action="append", default=[])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--skill-bank-json", default=None)
    parser.add_argument("--skill-retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--response-format", choices=["answer_tag", "final_answer", "plain"], default="answer_tag")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the XSkill tool-call protocol in SFT prompts.",
    )
    args = parser.parse_args()

    if not args.input_spec and not args.input_file:
        raise SystemExit("Provide at least one --input-spec or --input-file")

    memory = None
    if args.skill_bank_json:
        memory = SkillsOnlyMemory(
            skills_json_path=args.skill_bank_json,
            retrieval_mode=args.skill_retrieval_mode,
            embedding_model_path=args.embedding_model_path,
            task_specific_top_k=args.task_specific_top_k,
        )

    samples = _load_samples(args.input_spec, args.input_file)
    records = samples_to_sft_records(
        samples,
        memory=memory,
        top_k=args.top_k,
        response_format=args.response_format,
        max_records=args.max_records,
        include_images=args.include_images,
        enable_tools=args.enable_tools,
    )
    if not records:
        raise SystemExit("No SFT records were generated")

    write_parquet(args.output_path, records)
    manifest = {
        "format": "skillrl_sft_multimodal" if args.include_images else "skillrl_sft_text",
        "num_records": len(records),
        "include_images": args.include_images,
        "tool_protocol": args.enable_tools,
        "num_records_with_images": sum(1 for record in records if record.get("images")),
        "skill_bank": {
            "enabled": memory is not None,
            "skills_json_path": args.skill_bank_json,
            "retrieval_mode": args.skill_retrieval_mode if memory is not None else None,
            "top_k": args.top_k if memory is not None else None,
        },
        "response_format": args.response_format,
    }
    Path(args.output_path).with_suffix(Path(args.output_path).suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} SFT records to {args.output_path}")


if __name__ == "__main__":
    main()
