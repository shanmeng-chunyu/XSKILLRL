"""Prepare TIR-Bench into the local mixed-training protocol."""

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


ALLOWED_TASKS = {"refcoco", "maze", "instrument", "ocr", "contrast"}


def normalize_tirbench(records: List[Dict]) -> List[Dict]:
    normalized = []
    for index, item in enumerate(records):
        task = (
            item.get("_tirbench_task")
            or item.get("task")
            or item.get("category")
            or item.get("subset")
            or "unknown"
        )
        task = str(task).strip().lower()
        if task not in ALLOWED_TASKS:
            continue
        doc_id = item.get("doc_id") or item.get("id") or f"tirbench_{index:04d}"
        problem = item.get("problem") or item.get("question") or ""
        solution = item.get("solution") or item.get("answer") or ""
        images = normalize_images(item.get("images") or item.get("image") or item.get("img"))
        extra = dict(item)
        extra["_tirbench_task"] = task
        normalized.append(
            normalize_record(
                extra,
                benchmark_name="tirbench",
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

    records = normalize_tirbench(read_json(args.input_json))
    train_records, test_records, manifest = stratified_train_test_split(
        records,
        test_ratio=args.test_ratio,
        seed=args.seed,
        candidate_key_groups=[["_tirbench_task"]],
    )
    manifest["allowed_tasks"] = sorted(ALLOWED_TASKS)
    write_split_bundle(
        args.output_dir,
        benchmark_name="tirbench",
        all_records=records,
        train_records=train_records,
        test_records=test_records,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
