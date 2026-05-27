"""Prefix benchmark JSON record IDs with their benchmark name.

This fixes collisions such as ``tirbench`` sample ``2`` and ``mmbrowsecomp``
sample ``2`` both writing to the same accumulation output directory.

By default the script is a dry run. Add ``--in-place`` to rewrite JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import (
    ensure_benchmark_prefixed_id,
    normalize_benchmark_name,
    write_json,
)


BENCHMARK_DIR_ALIASES = {
    "agentvista": "agentvista",
    "mm-browsecomp": "mmbrowsecomp",
    "mmbrowsecomp": "mmbrowsecomp",
    "mmsearch-plus": "mmsearch_plus",
    "mmsearch_plus": "mmsearch_plus",
    "tir-bench": "tirbench",
    "tirbench": "tirbench",
    "visualtoolbench": "visualtoolbench",
    "visual-tool-bench": "visualtoolbench",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _iter_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json":
            files.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _benchmark_from_path(path: Path) -> str | None:
    for parent in [path.parent, *path.parents]:
        name = parent.name.strip().lower()
        if name in BENCHMARK_DIR_ALIASES:
            return BENCHMARK_DIR_ALIASES[name]
    return None


def _record_benchmark(record: dict[str, Any], path: Path) -> str | None:
    raw = record.get("benchmark_name") or _benchmark_from_path(path)
    if raw is None:
        return None
    return normalize_benchmark_name(str(raw))


def _record_id(record: dict[str, Any]) -> Any:
    return record.get("doc_id") or record.get("question_id") or record.get("id")


def _fix_record(
    record: dict[str, Any],
    path: Path,
    *,
    update_raw_id: bool = False,
) -> tuple[dict[str, Any], bool]:
    benchmark = _record_benchmark(record, path)
    raw_id = _record_id(record)
    if not benchmark or raw_id is None:
        return record, False

    prefixed = ensure_benchmark_prefixed_id(raw_id, benchmark)
    updated = dict(record)
    changed = False

    if updated.get("benchmark_name") != benchmark:
        updated["benchmark_name"] = benchmark
        changed = True

    old_doc_id = updated.get("doc_id")
    if old_doc_id != prefixed:
        if old_doc_id is not None and "original_doc_id" not in updated:
            updated["original_doc_id"] = old_doc_id
        updated["doc_id"] = prefixed
        changed = True

    if "question_id" in updated and updated["question_id"] != prefixed:
        if "original_question_id" not in updated:
            updated["original_question_id"] = updated["question_id"]
        updated["question_id"] = prefixed
        changed = True

    if update_raw_id and "id" in updated and updated["id"] != prefixed:
        if "original_id" not in updated:
            updated["original_id"] = updated["id"]
        updated["id"] = prefixed
        changed = True

    return updated, changed


def _fix_payload(
    payload: Any,
    path: Path,
    *,
    update_raw_id: bool = False,
) -> tuple[Any, int]:
    if not isinstance(payload, list):
        return payload, 0

    changed_count = 0
    fixed_records = []
    for item in payload:
        if isinstance(item, dict):
            fixed, changed = _fix_record(item, path, update_raw_id=update_raw_id)
            fixed_records.append(fixed)
            changed_count += int(changed)
        else:
            fixed_records.append(item)
    return fixed_records, changed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair benchmark JSON doc_id/question_id fields so they are benchmark-prefixed."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(ROOT / "benchmark")],
        help="JSON files or directories to scan. Defaults to XSkill-dev/benchmark.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite files in place.")
    parser.add_argument(
        "--update-raw-id",
        action="store_true",
        help="Also rewrite a record's raw `id` field. By default only doc_id/question_id are changed.",
    )
    args = parser.parse_args()

    files = _iter_json_files([Path(path) for path in args.paths])
    total_changed = 0
    touched_files = 0
    for path in files:
        try:
            payload = _load_json(path)
        except Exception as exc:
            print(f"[skip] {path}: failed to read JSON: {exc}")
            continue

        fixed_payload, changed_count = _fix_payload(
            payload,
            path,
            update_raw_id=args.update_raw_id,
        )
        if changed_count == 0:
            continue

        touched_files += 1
        total_changed += changed_count
        action = "fixed" if args.in_place else "would fix"
        print(f"[{action}] {path}: {changed_count} record(s)")
        if args.in_place:
            write_json(path, fixed_payload)

    mode = "in-place" if args.in_place else "dry-run"
    print(f"mode={mode}; files_changed={touched_files}; records_changed={total_changed}")


if __name__ == "__main__":
    main()
