"""Dataset helpers for RL-facing exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .benchmark_protocol import expand_records_sqrt_size, read_json


def sample_to_record(sample: Dict) -> Dict:
    return {
        "prompt": sample.get("problem", ""),
        "answer": sample.get("solution", ""),
        "images": list(sample.get("images", [])),
        "extra_info": {
            "doc_id": sample.get("doc_id"),
            "benchmark_name": sample.get("benchmark_name"),
            "solution": sample.get("solution", ""),
            "sample": sample,
        },
    }


def samples_to_grouped_records(samples: Sequence[Dict]) -> List[Dict]:
    return [sample_to_record(sample) for sample in samples]


def _parse_spec(spec: str) -> Tuple[str | None, Path]:
    if "=" in spec:
        benchmark_name, path = spec.split("=", 1)
        return benchmark_name.strip(), Path(path.strip())
    return None, Path(spec)


def load_records_grouped_by_benchmark(specs: Sequence[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for spec in specs:
        benchmark_name, path = _parse_spec(spec)
        samples = read_json(path)
        if not isinstance(samples, list):
            raise ValueError(f"Expected a list of samples in {path}")
        if benchmark_name is None:
            benchmark_name = path.parent.name.lower().replace("-", "_")
        normalized = []
        for sample in samples:
            payload = dict(sample)
            payload.setdefault("benchmark_name", benchmark_name)
            normalized.append(payload)
        grouped[benchmark_name] = normalized
    return grouped


def load_records_from_spec(
    specs: Sequence[str],
    *,
    mixing_strategy: str = "concat",
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    grouped = load_records_grouped_by_benchmark(specs)
    merged_samples: List[Dict] = []
    per_benchmark = {}
    for benchmark_name in sorted(grouped):
        samples = grouped[benchmark_name]
        merged_samples.extend(samples)
        per_benchmark[benchmark_name] = {"original_count": len(samples)}

    if mixing_strategy == "concat":
        manifest = {
            "mixing_strategy": mixing_strategy,
            "benchmark_counts": per_benchmark,
            "total_records": len(merged_samples),
        }
        return samples_to_grouped_records(merged_samples), manifest

    if mixing_strategy == "sqrt_size":
        expanded_samples, expand_manifest = expand_records_sqrt_size(
            merged_samples,
            benchmark_field="benchmark_name",
            seed=seed,
        )
        manifest = {
            "mixing_strategy": mixing_strategy,
            **expand_manifest,
        }
        return samples_to_grouped_records(expanded_samples), manifest

    raise ValueError(f"Unsupported mixing strategy: {mixing_strategy}")
