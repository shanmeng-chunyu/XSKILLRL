#!/usr/bin/env python3
"""Remove samples whose trajectory files contain multiple doc_id entries.

Repeated accumulation runs can append a new trajectory to an existing
``traj.jsonl`` instead of starting from a clean file. In that case one rollout
trajectory may contain multiple initial records with ``doc_id``. This script
finds those contaminated sample directories and deletes or quarantines the
whole sample so it can be rerun cleanly.

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


SKIP_DIR_NAMES = {"snapshots", "__pycache__", "logs", "memory_bank", "api_timings"}
TRAJ_NAMES = {"traj.jsonl", "traj.json"}


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if any((path / name).exists() for name in TRAJ_NAMES):
        return True
    return any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())


def iter_json_objects(path: Path) -> list[dict[str, Any]]:
    """Read JSONL or JSON trajectory files and return object rows."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError as exc:
        print(f"[WARN] failed to read {path}: {exc}")
        return []

    stripped = text.strip()
    if not stripped:
        return []

    # Some users call it traj.json even though the writer appends JSONL rows.
    rows: list[dict[str, Any]] = []
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def doc_ids_in_traj(path: Path) -> list[str]:
    doc_ids: list[str] = []
    for item in iter_json_objects(path):
        if "doc_id" not in item:
            continue
        value = item.get("doc_id")
        if value is None:
            continue
        doc_ids.append(str(value))
    return doc_ids


def trajectory_is_contaminated(
    path: Path,
    *,
    require_distinct_doc_id: bool,
) -> tuple[bool, list[str]]:
    doc_ids = doc_ids_in_traj(path)
    if require_distinct_doc_id:
        return len(set(doc_ids)) > 1, doc_ids
    return len(doc_ids) > 1, doc_ids


def find_bad_samples(
    output_dir: Path,
    *,
    require_distinct_doc_id: bool,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for sample_dir in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not looks_like_sample_dir(sample_dir):
            continue
        traj_paths = sorted(
            path for path in sample_dir.rglob("*")
            if path.is_file() and path.name in TRAJ_NAMES
        )
        for traj_path in traj_paths:
            matched, doc_ids = trajectory_is_contaminated(
                traj_path,
                require_distinct_doc_id=require_distinct_doc_id,
            )
            if matched:
                matches.append(
                    {
                        "sample_id": sample_dir.name,
                        "sample_dir": str(sample_dir),
                        "traj_path": str(traj_path),
                        "doc_id_count": len(doc_ids),
                        "unique_doc_id_count": len(set(doc_ids)),
                        "doc_ids": doc_ids,
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
        "--require-distinct-doc-id",
        action="store_true",
        help="Only match trajectories containing two or more distinct doc_id values. Default matches any doc_id count > 1.",
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

    matches = find_bad_samples(
        output_dir,
        require_distinct_doc_id=args.require_distinct_doc_id,
    )

    print("=== multi-doc_id trajectory cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"require_distinct_doc_id: {args.require_distinct_doc_id}")
    print(f"matched_samples: {len(matches)}")
    print(f"mode: {'delete' if args.delete else ('quarantine' if args.quarantine_dir else 'dry-run')}")
    for row in matches:
        print(
            f"[MATCH] {row['sample_id']} doc_id_count={row['doc_id_count']} "
            f"unique={row['unique_doc_id_count']} traj={row['traj_path']} "
            f"doc_ids={row['doc_ids'][:5]}"
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
