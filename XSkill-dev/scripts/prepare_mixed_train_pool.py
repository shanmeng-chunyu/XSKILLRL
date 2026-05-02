"""Merge benchmark train splits and sample a small global validation set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import allocate_global_val_split, read_json, write_json


def load_train_records(paths: List[str]) -> List[Dict]:
    merged = []
    for path_str in paths:
        path = Path(path_str)
        records = read_json(path)
        benchmark_name = path.parent.name.lower().replace("-", "_")
        for record in records:
            payload = dict(record)
            payload.setdefault("benchmark_name", benchmark_name)
            merged.append(payload)
    return sorted(merged, key=lambda item: (item["benchmark_name"], item["doc_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_train = load_train_records(args.train_json)
    train_core, global_val, manifest = allocate_global_val_split(
        merged_train,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    write_json(output_dir / "merged_train.json", merged_train)
    write_json(output_dir / "train_core.json", train_core)
    write_json(output_dir / "global_val.json", global_val)
    write_json(output_dir / "mixing_manifest.json", manifest)


if __name__ == "__main__":
    main()
