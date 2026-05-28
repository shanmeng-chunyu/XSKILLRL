#!/usr/bin/env python3
"""Remove accumulation samples whose error final answer was scored as correct.

The script scans sample directories under an XSkill accumulation output
directory. If any rollout ``metrics.json`` has ``final_answer`` starting with
``Error`` and ``accuracy_score`` equal to the configured score threshold
(``1.0`` by default), the whole sample directory is selected.

Default mode is dry-run. Use ``--delete`` to remove selected sample directories,
or ``--quarantine-dir`` to move them elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SKIP_DIR_NAMES = {"snapshots", "__pycache__"}


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_error_scored_correct(metrics: dict[str, Any], *, score_threshold: float) -> bool:
    final_answer = str(metrics.get("final_answer") or "").strip()
    score = as_float(metrics.get("accuracy_score"))
    return final_answer.lower().startswith("error") and score is not None and score >= score_threshold


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "metrics.json").exists() or (path / "traj.jsonl").exists():
        return True
    return any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())


def read_metrics(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] failed to read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"[WARN] expected object in {path}")
        return None
    return data


def find_bad_samples(output_dir: Path, *, score_threshold: float) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for sample_dir in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not looks_like_sample_dir(sample_dir):
            continue

        for metrics_path in sorted(sample_dir.rglob("metrics.json")):
            metrics = read_metrics(metrics_path)
            if metrics is None:
                continue
            if is_error_scored_correct(metrics, score_threshold=score_threshold):
                matches.append(
                    {
                        "sample_id": sample_dir.name,
                        "sample_dir": str(sample_dir),
                        "metrics_path": str(metrics_path),
                        "accuracy_score": metrics.get("accuracy_score"),
                        "trajectory_score": metrics.get("trajectory_score"),
                        "final_answer": metrics.get("final_answer"),
                    }
                )
                break
    return matches


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
        "--score-threshold",
        type=float,
        default=1.0,
        help="Treat error final answers with accuracy_score >= this value as bad. Default: 1.0",
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
    parser.add_argument("--report-json", help="Optional path to save matched sample report JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.exists() or not output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {output_dir}")
    if args.delete and args.quarantine_dir:
        raise SystemExit("Use either --delete or --quarantine-dir, not both.")

    matches = find_bad_samples(output_dir, score_threshold=args.score_threshold)

    print("=== error-scored sample cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"score_threshold: {args.score_threshold}")
    print(f"matched_samples: {len(matches)}")
    print(f"mode: {'delete' if args.delete else ('quarantine' if args.quarantine_dir else 'dry-run')}")
    for row in matches:
        print(
            f"[MATCH] {row['sample_id']} score={row['accuracy_score']} "
            f"metrics={row['metrics_path']} final_answer={row['final_answer']!r}"
        )

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report_json: {report_path}")

    quarantine_dir = Path(args.quarantine_dir).expanduser() if args.quarantine_dir else None
    apply_cleanup(output_dir, matches, delete=args.delete, quarantine_dir=quarantine_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
