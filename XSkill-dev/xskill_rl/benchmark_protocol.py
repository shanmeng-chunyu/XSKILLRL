"""Shared benchmark split helpers for the local mixed-training protocol."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path | str):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path | str, payload) -> None:
    target = Path(path)
    ensure_parent(target)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_images(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def normalize_benchmark_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    aliases = {
        "mm_browsecomp": "mmbrowsecomp",
        "mm_browse_comp": "mmbrowsecomp",
        "mmbrowse_comp": "mmbrowsecomp",
        "tir_bench": "tirbench",
        "visual_tool_bench": "visualtoolbench",
        "agent_vista": "agentvista",
    }
    return aliases.get(normalized, normalized)


def ensure_benchmark_prefixed_id(doc_id, benchmark_name: str) -> str:
    benchmark = normalize_benchmark_name(benchmark_name)
    doc_id_text = str(doc_id).strip()
    if not doc_id_text:
        return benchmark
    prefix = f"{benchmark}_"
    if doc_id_text == benchmark or doc_id_text.startswith(prefix):
        return doc_id_text
    return f"{prefix}{doc_id_text}"


def normalize_record(
    record: Dict,
    *,
    benchmark_name: str,
    doc_id: str,
    problem: str,
    solution: str,
    images: Sequence[str],
    extra_fields: Optional[Dict] = None,
) -> Dict:
    payload = dict(extra_fields or {})
    payload.update(
        {
            "doc_id": ensure_benchmark_prefixed_id(doc_id, benchmark_name),
            "problem": problem,
            "solution": solution,
            "images": list(images),
            "benchmark_name": normalize_benchmark_name(benchmark_name),
        }
    )
    return payload


def _make_label(record: Dict, keys: Sequence[str]) -> str:
    values = []
    for key in keys:
        raw = record.get(key, "")
        if raw is None:
            raw = ""
        values.append(str(raw).strip().lower() or "unknown")
    return "||".join(values)


def choose_stratify_keys(
    records: Sequence[Dict],
    candidate_key_groups: Sequence[Sequence[str]],
    *,
    min_count: int = 5,
) -> Optional[List[str]]:
    for keys in candidate_key_groups:
        labels = [_make_label(record, keys) for record in records]
        counts = Counter(labels)
        if len(counts) < 2:
            continue
        if min(counts.values()) >= min_count:
            return list(keys)
    return None


def _stable_shuffle(records: Sequence[Dict], seed: int, salt: str) -> List[Dict]:
    salted = []
    for index, record in enumerate(records):
        digest = hashlib.md5(
            f"{seed}:{salt}:{record.get('doc_id', index)}".encode("utf-8")
        ).hexdigest()
        salted.append((digest, record))
    salted.sort(key=lambda item: item[0])
    return [record for _, record in salted]


def stratified_train_test_split(
    records: Sequence[Dict],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_key_groups: Optional[Sequence[Sequence[str]]] = None,
    min_count: int = 5,
) -> Tuple[List[Dict], List[Dict], Dict]:
    items = [dict(record) for record in records]
    if not items:
        return [], [], {
            "seed": seed,
            "test_ratio": test_ratio,
            "stratify_keys": [],
            "counts": {"all": 0, "train": 0, "test": 0},
        }

    candidate_key_groups = candidate_key_groups or []
    stratify_keys = choose_stratify_keys(
        items,
        candidate_key_groups,
        min_count=min_count,
    )

    if not stratify_keys:
        shuffled = _stable_shuffle(items, seed, "all")
        test_size = int(round(len(shuffled) * test_ratio))
        if len(shuffled) > 1:
            test_size = max(1, min(test_size, len(shuffled) - 1))
        else:
            test_size = 0
        test_records = shuffled[:test_size]
        train_records = shuffled[test_size:]
    else:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for record in items:
            grouped[_make_label(record, stratify_keys)].append(record)

        train_records = []
        test_records = []
        for label in sorted(grouped):
            group = _stable_shuffle(grouped[label], seed, label)
            test_size = int(round(len(group) * test_ratio))
            test_size = max(1, min(test_size, len(group) - 1))
            test_records.extend(group[:test_size])
            train_records.extend(group[test_size:])

    train_records = sorted(train_records, key=lambda item: item["doc_id"])
    test_records = sorted(test_records, key=lambda item: item["doc_id"])
    manifest = {
        "seed": seed,
        "test_ratio": test_ratio,
        "stratify_keys": stratify_keys or [],
        "counts": {
            "all": len(items),
            "train": len(train_records),
            "test": len(test_records),
        },
    }
    if stratify_keys:
        manifest["stratify_distribution"] = dict(
            Counter(_make_label(record, stratify_keys) for record in items)
        )
    return train_records, test_records, manifest


def write_split_bundle(
    output_dir: Path | str,
    *,
    benchmark_name: str,
    all_records: Sequence[Dict],
    train_records: Sequence[Dict],
    test_records: Sequence[Dict],
    manifest: Dict,
) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "all.json", list(all_records))
    write_json(target_dir / "train.json", list(train_records))
    write_json(target_dir / "test.json", list(test_records))
    payload = {
        "benchmark_name": benchmark_name,
        **manifest,
    }
    write_json(target_dir / f"split_manifest_seed{manifest['seed']}.json", payload)


def allocate_global_val_split(
    records: Sequence[Dict],
    *,
    val_ratio: float = 0.05,
    seed: int = 42,
    benchmark_field: str = "benchmark_name",
) -> Tuple[List[Dict], List[Dict], Dict]:
    items = [dict(record) for record in records]
    if not items:
        return [], [], {
            "seed": seed,
            "val_ratio": val_ratio,
            "counts": {"merged_train": 0, "train_core": 0, "global_val": 0},
        }

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in items:
        grouped[str(record.get(benchmark_field, "unknown"))].append(record)

    total_target = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    desired = {
        name: len(group) * val_ratio
        for name, group in grouped.items()
    }
    base = {name: int(math.floor(value)) for name, value in desired.items()}
    current = sum(base.values())
    remainders = sorted(
        ((desired[name] - base[name], name) for name in grouped),
        reverse=True,
    )
    index = 0
    while current < total_target and index < len(remainders):
        _, name = remainders[index]
        if base[name] < len(grouped[name]):
            base[name] += 1
            current += 1
        index += 1

    val_records: List[Dict] = []
    train_records: List[Dict] = []
    per_benchmark_counts = {}
    for name in sorted(grouped):
        group = _stable_shuffle(grouped[name], seed, f"global-val:{name}")
        take = min(base[name], len(group))
        val_records.extend(group[:take])
        train_records.extend(group[take:])
        per_benchmark_counts[name] = {
            "merged_train": len(group),
            "train_core": len(group[take:]),
            "global_val": len(group[:take]),
        }

    train_records = sorted(train_records, key=lambda item: (item.get(benchmark_field, ""), item["doc_id"]))
    val_records = sorted(val_records, key=lambda item: (item.get(benchmark_field, ""), item["doc_id"]))
    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {
            "merged_train": len(items),
            "train_core": len(train_records),
            "global_val": len(val_records),
        },
        "per_benchmark": per_benchmark_counts,
    }
    return train_records, val_records, manifest


def expand_records_sqrt_size(
    records: Sequence[Dict],
    *,
    benchmark_field: str = "benchmark_name",
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    items = [dict(record) for record in records]
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in items:
        grouped[str(record.get(benchmark_field, "unknown"))].append(record)

    if not grouped:
        return [], {"benchmark_counts": {}}

    max_count = max(len(group) for group in grouped.values())
    expanded: List[Dict] = []
    benchmark_counts = {}
    for name in sorted(grouped):
        group = _stable_shuffle(grouped[name], seed, f"sqrt:{name}")
        target = int(round(math.sqrt(len(group) * max_count)))
        target = max(len(group), target)
        generated: List[Dict] = []
        cursor = 0
        while len(generated) < target:
            generated.append(dict(group[cursor % len(group)]))
            cursor += 1
        expanded.extend(generated)
        benchmark_counts[name] = {
            "original_count": len(group),
            "expanded_count": len(generated),
        }

    total_expanded = len(expanded)
    for name, stats in benchmark_counts.items():
        stats["sampling_probability"] = (
            stats["expanded_count"] / total_expanded if total_expanded else 0.0
        )

    expanded = sorted(
        expanded,
        key=lambda item: (item.get(benchmark_field, ""), item["doc_id"]),
    )
    return expanded, {"benchmark_counts": benchmark_counts, "total_expanded": total_expanded}
