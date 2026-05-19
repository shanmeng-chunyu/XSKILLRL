"""Export XSkill samples into SkillRL/verl-compatible RL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from xskill_rl.dataset import load_records_from_spec

from .skill_bank import SkillsOnlyMemory


DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal reasoning agent. Use the provided skills when they "
    "are relevant, reason carefully over the question and images, and produce "
    "the final answer."
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _as_verl_image_items(value: Any) -> List[Dict[str, str]]:
    return [{"image": image} for image in _as_text_list(value)]


def make_skill_augmented_messages(
    problem: str,
    *,
    skill_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    system_content = system_prompt
    if skill_text:
        system_content = (
            f"{system_content}\n\n"
            "## Retrieved Relevant Experience\n"
            f"{skill_text}"
        )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": problem},
    ]


def sample_to_verl_record(
    sample: Dict[str, Any],
    *,
    index: int = 0,
    memory: Optional[SkillsOnlyMemory] = None,
    top_k: int = 6,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    problem = _as_text(sample.get("problem") or sample.get("question") or "")
    skill_text = ""
    retrieval_payload: Dict[str, Any] = {}
    if memory is not None:
        retrieval_payload = memory.retrieve(
            task_description=problem,
            top_k=top_k,
            metadata=sample,
        )
        skill_text = memory.format_for_prompt(retrieval_payload)
    doc_id = _as_text(sample.get("doc_id") or sample.get("question_id") or index)
    solution = _as_text(sample.get("solution", ""))
    images = _as_text_list(sample.get("images", []))
    verl_images = _as_verl_image_items(images)
    sample_json = json.dumps(sample, ensure_ascii=False)

    benchmark_name = _as_text(sample.get("benchmark_name") or "xskill")
    extra_info = {
        "index": index,
        "doc_id": doc_id,
        "benchmark_name": benchmark_name,
        "solution": solution,
        "answer": solution,
        "sample_json": sample_json,
        "skill_retrieval": {
            "enabled": memory is not None,
            "task_type": _as_text(retrieval_payload.get("task_type")),
            "retrieval_mode": _as_text(retrieval_payload.get("retrieval_mode")),
            "top_k": int(top_k),
        },
    }

    env_kwargs = {
        "doc_id": doc_id,
        "benchmark_name": benchmark_name,
        "problem": problem,
        "images": images,
        "solution": solution,
        "prompt": make_skill_augmented_messages(
            problem,
            skill_text=skill_text,
            system_prompt=system_prompt,
        ),
    }

    return {
        "data_source": benchmark_name,
        "prompt": env_kwargs["prompt"],
        "images": verl_images,
        "env_kwargs": env_kwargs,
        "reward_model": {
            "style": "xskill_rule_or_judge",
            "ground_truth": solution,
        },
        "extra_info": extra_info,
    }


def samples_to_verl_records(
    samples: Sequence[Dict[str, Any]],
    *,
    memory: Optional[SkillsOnlyMemory] = None,
    top_k: int = 6,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    return [
        sample_to_verl_record(
            sample,
            index=index,
            memory=memory,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        for index, sample in enumerate(samples)
    ]


def load_verl_records_from_spec(
    specs: Sequence[str],
    *,
    mixing_strategy: str = "concat",
    seed: int = 42,
    skills_json_path: Optional[str] = None,
    retrieval_mode: str = "template",
    embedding_model_path: Optional[str] = None,
    top_k: int = 6,
    task_specific_top_k: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rl_records, base_manifest = load_records_from_spec(
        specs,
        mixing_strategy=mixing_strategy,
        seed=seed,
    )
    samples = [record["extra_info"]["sample"] for record in rl_records]

    memory = None
    if skills_json_path:
        memory = SkillsOnlyMemory(
            skills_json_path=skills_json_path,
            retrieval_mode=retrieval_mode,
            embedding_model_path=embedding_model_path,
            task_specific_top_k=task_specific_top_k,
        )

    records = samples_to_verl_records(
        samples,
        memory=memory,
        top_k=top_k,
        system_prompt=system_prompt,
    )
    manifest = {
        **base_manifest,
        "format": "skillrl_verl",
        "skill_bank": {
            "enabled": skills_json_path is not None,
            "skills_json_path": skills_json_path,
            "retrieval_mode": retrieval_mode if skills_json_path else None,
            "top_k": top_k if skills_json_path else None,
            "task_specific_top_k": task_specific_top_k,
        },
    }
    return records, manifest


def write_jsonl(path: Path | str, records: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(path: Path | str, records: Sequence[Dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for parquet export") from exc

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet export") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(target, index=False)
