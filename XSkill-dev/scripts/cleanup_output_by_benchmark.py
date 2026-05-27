#!/usr/bin/env python3
"""Remove accumulation output sample directories by benchmark membership.

The script reads a mixed-protocol JSON file, maps sample directory names
(`doc_id`) to benchmark names, then scans an accumulation output directory and
selects sample folders whose doc_id belongs to any target benchmark.

Default mode is dry-run. Use --delete to remove matched sample directories, or
--quarantine-dir to move them elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TARGET_BENCHMARKS = ("tirbench", "mmsearch_plus", "mmbrowsecomp")
SKIP_DIR_NAMES = {"snapshots", "__pycache__"}


def normalize_benchmark(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace("-", "_")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_doc_id_index(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for record in records:
        doc_id = record.get("doc_id", record.get("question_id"))
        benchmark = normalize_benchmark(record.get("benchmark_name"))
        if doc_id is None or not benchmark:
            continue
        index[str(doc_id)].add(benchmark)
    return dict(index)


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "traj.jsonl").exists() or (path / "metrics.json").exists():
        return True
    if any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir()):
        return True
    return False


def find_matched_samples(
    output_dir: Path,
    doc_index: dict[str, set[str]],
    target_benchmarks: set[str],
    *,
    include_ambiguous_non_target: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sample_dir in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not looks_like_sample_dir(sample_dir):
            continue

        benchmarks = doc_index.get(sample_dir.name, set())
        if not benchmarks:
            skipped.append({
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "reason": "doc_id not found in train_core",
            })
            continue

        target_hits = sorted(benchmarks & target_benchmarks)
        non_target_hits = sorted(benchmarks - target_benchmarks)
        ambiguous = len(benchmarks) > 1

        if target_hits and (not non_target_hits or include_ambiguous_non_target):
            matches.append({
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "benchmarks": sorted(benchmarks),
                "target_hits": target_hits,
                "ambiguous": ambiguous,
            })
        elif target_hits and non_target_hits:
            skipped.append({
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "benchmarks": sorted(benchmarks),
                "reason": "ambiguous doc_id also belongs to non-target benchmark",
            })

    return matches, skipped


def ensure_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved:
        raise ValueError(f"Refusing to delete output root itself: {child}")
    if os.path.commonpath([str(parent_resolved), str(child_resolved)]) != str(parent_resolved):
        raise ValueError(f"Refusing path outside output root: {child}")


def apply_cleanup(
    output_dir: Path,
    matches: list[dict[str, Any]],
    *,
    delete: bool,
    quarantine_dir: Path | None,
) -> None:
    if not delete and quarantine_dir is None:
        return

    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for row in matches:
        sample_dir = Path(row["sample_dir"])
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
        help="Mixed protocol JSON used to map doc_id to benchmark_name.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Target benchmark to remove. Defaults: tirbench, mmsearch_plus, mmbrowsecomp. Can be repeated.",
    )
    parser.add_argument(
        "--include-ambiguous-non-target",
        action="store_true",
        help="Delete doc_id directories that map to both target and non-target benchmarks.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched sample directories. Without this, only prints a dry-run report.",
    )
    parser.add_argument(
        "--quarantine-dir",
        help="Move matched sample directories here instead of deleting them.",
    )
    parser.add_argument("--report-json", help="Optional path to save matched/skipped report JSON.")
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

    doc_index = build_doc_id_index(records)
    target_benchmarks = {
        normalize_benchmark(name)
        for name in (args.benchmark or DEFAULT_TARGET_BENCHMARKS)
    }
    matches, skipped = find_matched_samples(
        output_dir,
        doc_index,
        target_benchmarks,
        include_ambiguous_non_target=args.include_ambiguous_non_target,
    )

    print("=== benchmark output cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"train_core: {train_core}")
    print(f"target_benchmarks: {', '.join(sorted(target_benchmarks))}")
    print(f"matched_samples: {len(matches)}")
    print(f"skipped_samples: {len(skipped)}")
    print(f"mode: {'delete' if args.delete else ('quarantine' if args.quarantine_dir else 'dry-run')}")

    for row in matches:
        ambiguity = " ambiguous" if row["ambiguous"] else ""
        print(
            f"[MATCH]{ambiguity} {row['sample_id']} "
            f"benchmarks={','.join(row['benchmarks'])}"
        )
    for row in skipped[:20]:
        print(f"[SKIP] {row['sample_id']} reason={row['reason']}")
    if len(skipped) > 20:
        print(f"[SKIP] ... {len(skipped) - 20} more skipped samples")

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"matches": matches, "skipped": skipped}
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report_json: {report_path}")

    quarantine_dir = Path(args.quarantine_dir).expanduser() if args.quarantine_dir else None
    apply_cleanup(output_dir, matches, delete=args.delete, quarantine_dir=quarantine_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
