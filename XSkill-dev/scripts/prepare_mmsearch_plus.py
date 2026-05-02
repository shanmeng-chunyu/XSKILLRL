"""Prepare MMSearch-Plus into the local mixed-training protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import (
    normalize_images,
    normalize_record,
    read_json,
    stratified_train_test_split,
    write_split_bundle,
)


def _collect_images(item: Dict) -> List[str]:
    if item.get("images"):
        return normalize_images(item["images"])
    ordered = []
    for key in sorted(item):
        if key.startswith("img_") and item[key]:
            ordered.append(item[key])
    return normalize_images(ordered)


def normalize_mmsearch_plus(records: List[Dict]) -> List[Dict]:
    normalized = []
    for index, item in enumerate(records):
        doc_id = item.get("doc_id") or item.get("id") or f"mmsearch_plus_{index:04d}"
        problem = item.get("problem") or item.get("question") or ""
        solution = item.get("solution") or item.get("answer") or ""
        images = _collect_images(item)
        category = item.get("category") or "unknown"
        difficulty = item.get("difficulty") or "unknown"
        extra = dict(item)
        extra["category"] = category
        extra["difficulty"] = difficulty
        normalized.append(
            normalize_record(
                extra,
                benchmark_name="mmsearch_plus",
                doc_id=doc_id,
                problem=problem,
                solution=solution,
                images=images,
                extra_fields=extra,
            )
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    records = normalize_mmsearch_plus(read_json(args.input_json))
    train_records, test_records, manifest = stratified_train_test_split(
        records,
        test_ratio=args.test_ratio,
        seed=args.seed,
        candidate_key_groups=[["category", "difficulty"], ["category"]],
    )
    write_split_bundle(
        args.output_dir,
        benchmark_name="mmsearch_plus",
        all_records=records,
        train_records=train_records,
        test_records=test_records,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
