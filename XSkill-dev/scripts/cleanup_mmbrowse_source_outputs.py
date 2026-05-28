#!/usr/bin/env python3
"""Remove MMBrowseComp accumulation outputs whose protocol record has source URLs.

Use this after changing source URL prompt injection.  The script reads the
mixed protocol JSON, finds ``mmbrowsecomp`` samples with a non-empty ``source``
field, then removes matching sample directories from an accumulation output
directory so they can be rerun.

Default mode is dry-run. Use ``--delete`` to remove matched directories, or
``--quarantine-dir`` to move them elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SKIP_DIR_NAMES = {"snapshots", "__pycache__", "logs", "memory_bank", "api_timings"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_benchmark(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def has_source(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(has_source(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_source(item) for item in value)
    return bool(value)


def source_doc_ids(records: list[dict[str, Any]], benchmark: str) -> set[str]:
    ids: set[str] = set()
    benchmark = normalize_benchmark(benchmark)
    for record in records:
        if normalize_benchmark(record.get("benchmark_name")) != benchmark:
            continue
        if not has_source(record.get("source")):
            continue
        doc_id = record.get("doc_id") or record.get("question_id") or record.get("id")
        if doc_id is not None:
            ids.add(str(doc_id))
    return ids


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "traj.jsonl").exists() or (path / "metrics.json").exists():
        return True
    if any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir()):
        return True
    return False


def ensure_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved:
        raise ValueError(f"Refusing to delete output root itself: {child}")
    if os.path.commonpath([str(parent_resolved), str(child_resolved)]) != str(parent_resolved):
        raise ValueError(f"Refusing path outside output root: {child}")


def apply_cleanup(output_dir: Path, matches: list[Path], *, delete: bool, quarantine_dir: Path | None) -> None:
    if not delete and quarantine_dir is None:
        return
    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    for sample_dir in matches:
        ensure_child(output_dir, sample_dir)
        if quarantine_dir is not None:
            target = quarantine_dir / sample_dir.name
            if target.exists():
                suffix = 1
                while (quarantine_dir / f"{sample_dir.name}.{suffix}").exists():
                    suffix += 1
                target = quarantine_dir / f"{sample_dir.name}.{suffix}"
            print(f"[MOVE] {sample_dir} -> {target}")
            shutil.move(str(sample_dir), str(target))
        else:
            print(f"[DELETE] {sample_dir}")
            shutil.rmtree(sample_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Accumulation output directory, e.g. output/xskill_accum/qwen3vl8b_mixed_train_core_seed42",
    )
    parser.add_argument(
        "--train-core",
        default="benchmark/_mixed_protocol/train_core.json",
        help="Mixed protocol JSON used for doc_id/source lookup.",
    )
    parser.add_argument("--benchmark", default="mmbrowsecomp")
    parser.add_argument("--delete", action="store_true", help="Actually delete matched output directories.")
    parser.add_argument("--quarantine-dir", help="Move matched directories here instead of deleting.")
    parser.add_argument("--report-json", help="Optional path to save a JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    train_core = Path(args.train_core).expanduser()

    if not output_dir.exists() or not output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {output_dir}")
    if not train_core.exists() or not train_core.is_file():
        raise SystemExit(f"train_core JSON not found: {train_core}")
    if args.delete and args.quarantine_dir:
        raise SystemExit("Use either --delete or --quarantine-dir, not both.")

    records = read_json(train_core)
    if not isinstance(records, list):
        raise SystemExit(f"Expected list records in {train_core}")

    target_ids = source_doc_ids(records, args.benchmark)
    matches = [
        sample_dir
        for sample_dir in sorted(output_dir.iterdir(), key=lambda path: path.name)
        if looks_like_sample_dir(sample_dir) and sample_dir.name in target_ids
    ]
    missing = sorted(target_ids - {path.name for path in matches})

    mode = "delete" if args.delete else ("quarantine" if args.quarantine_dir else "dry-run")
    print("=== mmbrowse source output cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"train_core: {train_core}")
    print(f"benchmark: {normalize_benchmark(args.benchmark)}")
    print(f"source_protocol_samples: {len(target_ids)}")
    print(f"matched_output_dirs: {len(matches)}")
    print(f"source_samples_without_output_dir: {len(missing)}")
    print(f"mode: {mode}")
    for sample_dir in matches:
        print(f"[MATCH] {sample_dir.name}")
    for sample_id in missing[:20]:
        print(f"[MISS] {sample_id}")
    if len(missing) > 20:
        print(f"[MISS] ... {len(missing) - 20} more")

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "output_dir": str(output_dir),
            "train_core": str(train_core),
            "benchmark": normalize_benchmark(args.benchmark),
            "matches": [str(path) for path in matches],
            "missing_output_dirs": missing,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_json: {report_path}")

    quarantine_dir = Path(args.quarantine_dir).expanduser() if args.quarantine_dir else None
    apply_cleanup(output_dir, matches, delete=args.delete, quarantine_dir=quarantine_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
