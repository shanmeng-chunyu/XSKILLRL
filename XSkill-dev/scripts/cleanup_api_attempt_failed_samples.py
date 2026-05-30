#!/usr/bin/env python3
"""Find and remove samples whose rollouts hit "All API attempts failed".

The script scans sample directories under an XSkill accumulation output
directory. If any rollout ``metrics.json`` or trajectory file contains the
configured API failure marker, the whole sample directory is selected so it can
be rerun.

Default mode is dry-run. Use ``--delete`` to remove matched sample directories,
or ``--quarantine-dir`` to move them elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = (
    "Error: All API attempts failed",
    "All API attempts failed",
)
SKIP_DIR_NAMES = {"snapshots", "__pycache__", "logs", "memory_bank", "api_timings"}
SCAN_FILE_NAMES = {"metrics.json", "traj.jsonl", "traj.json"}


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if any((path / name).exists() for name in SCAN_FILE_NAMES):
        return True
    return any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError as exc:
        print(f"[WARN] failed to read {path}: {exc}")
        return ""


def json_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            payload: Any = json.load(f)
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return read_text(path)


def file_has_pattern(path: Path, patterns: tuple[str, ...]) -> bool:
    text = json_text(path) if path.suffix == ".json" else read_text(path)
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def find_bad_samples(output_dir: Path, *, patterns: tuple[str, ...]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for sample_dir in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not looks_like_sample_dir(sample_dir):
            continue
        scan_paths = sorted(
            path for path in sample_dir.rglob("*")
            if path.is_file() and path.name in SCAN_FILE_NAMES
        )
        for scan_path in scan_paths:
            if file_has_pattern(scan_path, patterns):
                matches.append(
                    {
                        "sample_id": sample_dir.name,
                        "sample_dir": str(sample_dir),
                        "matched_file": str(scan_path),
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
    matches: list[dict[str, str]],
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
        "--pattern",
        action="append",
        default=[],
        help="Additional failure marker. Can be passed multiple times.",
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

    patterns = DEFAULT_PATTERNS + tuple(args.pattern)
    matches = find_bad_samples(output_dir, patterns=patterns)

    print("=== API attempts failed cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"patterns: {patterns}")
    print(f"matched_samples: {len(matches)}")
    print(f"mode: {'delete' if args.delete else ('quarantine' if args.quarantine_dir else 'dry-run')}")
    for row in matches:
        print(f"[MATCH] {row['sample_id']} file={row['matched_file']}")

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
