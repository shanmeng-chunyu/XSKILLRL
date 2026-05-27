"""Prepare VisualToolBench into the local mixed-training protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import (
    ensure_benchmark_prefixed_id,
    normalize_images,
    normalize_record,
    read_json,
    stratified_train_test_split,
    write_split_bundle,
)


def normalize_visualtoolbench(records: List[Dict]) -> List[Dict]:
    normalized = []
    for index, item in enumerate(records):
        turncase = str(item.get("turncase") or item.get("turn_case") or "").strip().lower()
        eval_focus = str(item.get("eval_focus") or "").strip().lower()
        if turncase != "single-turn" or eval_focus != "hybrid_tool_reasoning":
            continue

        prompts = item.get("turn_prompts") or item.get("prompts") or []
        answers = item.get("turn_golden_answers") or item.get("answers") or []
        images_by_turn = item.get("images_by_turn") or []
        problem = prompts[0] if prompts else item.get("problem") or item.get("question") or ""
        solution = answers[0] if answers else item.get("solution") or item.get("answer") or ""
        images = normalize_images(images_by_turn[0] if images_by_turn else item.get("images"))
        prompt_category = item.get("prompt_category") or "unknown"
        doc_id = item.get("doc_id") or item.get("id") or f"visualtoolbench_{index:04d}"
        doc_id = ensure_benchmark_prefixed_id(doc_id, "visualtoolbench")
        extra = dict(item)
        extra["prompt_category"] = prompt_category
        normalized.append(
            normalize_record(
                extra,
                benchmark_name="visualtoolbench",
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

    records = normalize_visualtoolbench(read_json(args.input_json))
    train_records, test_records, manifest = stratified_train_test_split(
        records,
        test_ratio=args.test_ratio,
        seed=args.seed,
        candidate_key_groups=[["prompt_category"]],
    )
    write_split_bundle(
        args.output_dir,
        benchmark_name="visualtoolbench",
        all_records=records,
        train_records=train_records,
        test_records=test_records,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
