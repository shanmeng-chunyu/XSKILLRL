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
    problem = sample.get("problem") or sample.get("question") or ""
    skill_text = ""
    retrieval_payload: Dict[str, Any] = {}
    if memory is not None:
        retrieval_payload = memory.retrieve(
            task_description=problem,
            top_k=top_k,
            metadata=sample,
        )
        skill_text = memory.format_for_prompt(retrieval_payload)

    benchmark_name = sample.get("benchmark_name") or "xskill"
    extra_info = {
        "index": index,
        "doc_id": sample.get("doc_id") or sample.get("question_id") or str(index),
        "benchmark_name": benchmark_name,
        "solution": sample.get("solution", ""),
        "answer": sample.get("solution", ""),
        "sample": sample,
        "skill_retrieval": {
            "enabled": memory is not None,
            "task_type": retrieval_payload.get("task_type"),
            "retrieval_mode": retrieval_payload.get("retrieval_mode"),
            "top_k": top_k,
        },
    }

    env_kwargs = {
        "doc_id": extra_info["doc_id"],
        "benchmark_name": benchmark_name,
        "problem": problem,
        "images": list(sample.get("images", [])),
        "solution": sample.get("solution", ""),
        "prompt": make_skill_augmented_messages(
            problem,
            skill_text=skill_text,
            system_prompt=system_prompt,
        ),
        "sample": sample,
    }

    return {
        "data_source": benchmark_name,
        "prompt": env_kwargs["prompt"],
        "images": list(sample.get("images", [])),
        "env_kwargs": env_kwargs,
        "reward_model": {
            "style": "xskill_rule_or_judge",
            "ground_truth": sample.get("solution", ""),
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
