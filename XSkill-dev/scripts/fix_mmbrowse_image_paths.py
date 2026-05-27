"""Remove accidental benchmark prefixes from MMBrowseComp image paths.

MMBrowseComp images are usually remote URLs such as:

    https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/...

If a preprocessing step turns them into ``benchmark/https://...`` or
``MM-BrowseComp/https://...``, the image loader treats the value as a local
file path. This script repairs those records without changing their prefixed
sample IDs.
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

from xskill_rl.benchmark_protocol import normalize_benchmark_name, write_json


IMAGE_FIELDS = ("images", "image", "img")
KNOWN_PREFIXES = (
    "benchmark/",
    "xskill-dev/benchmark/",
    "xskill_dev/benchmark/",
    "mm-browsecomp/",
    "mmbrowsecomp/",
)


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


def _is_mmbrowse_record(record: dict[str, Any], path: Path) -> bool:
    benchmark = record.get("benchmark_name")
    if benchmark and normalize_benchmark_name(str(benchmark)) == "mmbrowsecomp":
        return True
    return any(parent.name.lower() in {"mm-browsecomp", "mmbrowsecomp"} for parent in path.parents)


def _strip_to_url(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    lower = normalized.lower()
    starts = [
        index
        for marker in ("https://", "http://", "data:image/")
        for index in [lower.find(marker)]
        if index >= 0
    ]
    if starts:
        return normalized[min(starts) :]
    return normalized


def _strip_known_prefixes(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    changed = True
    while changed:
        changed = False
        leading_len = len(normalized) - len(normalized.lstrip("./"))
        leading = normalized[:leading_len]
        stripped = normalized[leading_len:]
        lower = stripped.lower()
        for prefix in KNOWN_PREFIXES:
            if lower.startswith(prefix):
                normalized = leading + stripped[len(prefix) :]
                changed = True
                break
    return normalized


def _fix_image_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        fixed = _strip_to_url(value)
        if fixed == value:
            fixed = _strip_known_prefixes(value)
        return fixed, int(fixed != value)
    if isinstance(value, list):
        fixed_items = []
        changed_count = 0
        for item in value:
            fixed_item, changed = _fix_image_value(item)
            fixed_items.append(fixed_item)
            changed_count += changed
        return fixed_items, changed_count
    if isinstance(value, dict):
        fixed = dict(value)
        changed_count = 0
        for key in ("image", "url", "path"):
            if key in fixed:
                fixed[key], changed = _fix_image_value(fixed[key])
                changed_count += changed
        return fixed, changed_count
    return value, 0


def _fix_record(record: dict[str, Any], path: Path) -> tuple[dict[str, Any], int]:
    if not _is_mmbrowse_record(record, path):
        return record, 0
    fixed = dict(record)
    changed_count = 0
    for field in IMAGE_FIELDS:
        if field in fixed:
            fixed[field], changed = _fix_image_value(fixed[field])
            changed_count += changed
    return fixed, changed_count


def _fix_payload(payload: Any, path: Path) -> tuple[Any, int]:
    if not isinstance(payload, list):
        return payload, 0
    fixed_records = []
    changed_count = 0
    for item in payload:
        if isinstance(item, dict):
            fixed, changed = _fix_record(item, path)
            fixed_records.append(fixed)
            changed_count += changed
        else:
            fixed_records.append(item)
    return fixed_records, changed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair MMBrowseComp image paths accidentally prefixed with benchmark directories."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            str(ROOT / "benchmark" / "MM-BrowseComp"),
            str(ROOT / "benchmark" / "_mixed_protocol"),
        ],
        help="JSON files or directories to scan. Defaults to MM-BrowseComp and _mixed_protocol JSONs.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite files in place.")
    args = parser.parse_args()

    files = _iter_json_files([Path(path) for path in args.paths])
    touched_files = 0
    total_changed = 0
    for path in files:
        try:
            payload = _load_json(path)
        except Exception as exc:
            print(f"[skip] {path}: failed to read JSON: {exc}")
            continue
        fixed_payload, changed_count = _fix_payload(payload, path)
        if changed_count == 0:
            continue
        touched_files += 1
        total_changed += changed_count
        action = "fixed" if args.in_place else "would fix"
        print(f"[{action}] {path}: {changed_count} image value(s)")
        if args.in_place:
            write_json(path, fixed_payload)

    mode = "in-place" if args.in_place else "dry-run"
    print(f"mode={mode}; files_changed={touched_files}; image_values_changed={total_changed}")


if __name__ == "__main__":
    main()
