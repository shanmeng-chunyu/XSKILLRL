#!/usr/bin/env python3
"""Find and remove accumulation samples with failed tool 403 rollouts.

The script scans sample directories under an XSkill accumulation output
directory. If any rollout trajectory event for a sample contains a configured
tool call and a 403-style failure, the whole sample directory is selected.

Default mode is dry-run. Use --delete to remove selected sample directories, or
--quarantine-dir to move them elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ERROR_PATTERNS = (
    "api request failed with status 403",
    "status 403",
    "403 client error",
    "http 403",
    '"code":"403"',
    '"code": "403"',
    "'code': '403'",
)

DEFAULT_QUOTA_PATTERNS = (
    "you do not have enough money",
    "package quota",
    "not enough money",
    "insufficient quota",
    "insufficient balance",
    "quota",
    "billing",
    "\u4f59\u989d",
    "\u4f59\u989d\u4e0d\u8db3",
)

SKIP_DIR_NAMES = {
    "snapshots",
    "__pycache__",
}


def _json_text(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False).lower()
    except Exception:
        return str(obj).lower()


def _iter_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield value
            yield from _iter_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield value
            yield from _iter_values(value)


def _object_has_target_tool(obj: Any, tool_names: set[str]) -> bool:
    if not isinstance(obj, dict):
        return False

    for key in ("tool_name", "name"):
        value = obj.get(key)
        if isinstance(value, str) and value in tool_names:
            return True

    tool_call = obj.get("tool_call")
    if isinstance(tool_call, dict) and tool_call.get("tool_name") in tool_names:
        return True

    tool_error = obj.get("tool_error")
    if isinstance(tool_error, dict) and tool_error.get("tool_name") in tool_names:
        return True

    return any(isinstance(value, str) and value in tool_names for value in _iter_values(obj))


def _has_pattern(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _is_matching_403_result(text: Any, error_patterns: Iterable[str], quota_patterns: Iterable[str]) -> bool:
    """Strictly match the Bocha-style quota failure shown in trajectory tool results."""
    if text is None:
        return False
    lowered = str(text).lower()
    has_403 = _has_pattern(lowered, error_patterns)
    has_quota = _has_pattern(lowered, quota_patterns)
    return has_403 and has_quota


def _object_has_bad_target_tool_result(
    obj: Any,
    *,
    tool_names: set[str],
    error_patterns: Iterable[str],
    quota_patterns: Iterable[str],
) -> bool:
    if not isinstance(obj, dict):
        return False

    tool_call = obj.get("tool_call")
    if isinstance(tool_call, dict) and tool_call.get("tool_name") in tool_names:
        return _is_matching_403_result(
            tool_call.get("result"),
            error_patterns,
            quota_patterns,
        )

    tool_error = obj.get("tool_error")
    if isinstance(tool_error, dict) and tool_error.get("tool_name") in tool_names:
        return _is_matching_403_result(
            tool_error.get("error") or tool_error.get("result"),
            error_patterns,
            quota_patterns,
        )

    return False


def trajectory_has_bad_tool_call(
    traj_path: Path,
    *,
    tool_names: set[str],
    error_patterns: Iterable[str],
    quota_patterns: Iterable[str],
    loose: bool,
) -> tuple[bool, str]:
    """Return whether a trajectory contains target tool plus a 403-like error."""
    saw_target_tool = False
    saw_error = False
    first_tool_reason = ""
    first_error_reason = ""

    try:
        with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                line_lower = line.lower()
                line_has_target_tool = any(tool_name in line_lower for tool_name in tool_names)
                line_has_error = _is_matching_403_result(line_lower, error_patterns, quota_patterns)

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = None

                if obj is not None:
                    obj_has_target_tool = _object_has_target_tool(obj, tool_names)
                    obj_has_error = _object_has_bad_target_tool_result(
                        obj,
                        tool_names=tool_names,
                        error_patterns=error_patterns,
                        quota_patterns=quota_patterns,
                    )

                    if obj_has_target_tool:
                        saw_target_tool = True
                        first_tool_reason = first_tool_reason or f"{traj_path}:{line_no}"
                    if obj_has_error:
                        saw_error = True
                        first_error_reason = first_error_reason or f"{traj_path}:{line_no}"

                    if obj_has_target_tool and obj_has_error:
                        return True, f"{traj_path}:{line_no}"
                else:
                    if line_has_target_tool:
                        saw_target_tool = True
                        first_tool_reason = first_tool_reason or f"{traj_path}:{line_no}"
                    if line_has_error:
                        saw_error = True
                        first_error_reason = first_error_reason or f"{traj_path}:{line_no}"
                    if line_has_target_tool and line_has_error:
                        return True, f"{traj_path}:{line_no}"
    except OSError as exc:
        print(f"[WARN] failed to read {traj_path}: {exc}")
        return False, ""

    if loose and saw_target_tool and saw_error:
        return True, first_tool_reason or first_error_reason or str(traj_path)
    return False, ""


def looks_like_sample_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "traj.jsonl").exists():
        return True
    if any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir()):
        return True
    return False


def find_bad_samples(
    output_dir: Path,
    *,
    tool_names: set[str],
    error_patterns: Iterable[str],
    quota_patterns: Iterable[str],
    loose: bool,
) -> list[tuple[Path, str]]:
    bad_samples: list[tuple[Path, str]] = []
    for sample_dir in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not looks_like_sample_dir(sample_dir):
            continue
        for traj_path in sorted(sample_dir.rglob("traj.jsonl")):
            matched, reason = trajectory_has_bad_tool_call(
                traj_path,
                tool_names=tool_names,
                error_patterns=error_patterns,
                quota_patterns=quota_patterns,
                loose=loose,
            )
            if matched:
                bad_samples.append((sample_dir, reason))
                break
    return bad_samples


def ensure_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved:
        raise ValueError(f"Refusing to delete output root itself: {child}")
    if os.path.commonpath([str(parent_resolved), str(child_resolved)]) != str(parent_resolved):
        raise ValueError(f"Refusing path outside output root: {child}")


def delete_or_quarantine_samples(
    output_dir: Path,
    bad_samples: list[tuple[Path, str]],
    *,
    delete: bool,
    quarantine_dir: Path | None,
) -> None:
    if not delete and quarantine_dir is None:
        return

    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir, _ in bad_samples:
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
    parser = argparse.ArgumentParser(
        description="Delete or quarantine samples whose rollouts hit target tool 403 failures."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Accumulation output directory, e.g. output/xskill_accum/qwen3vl8b_mixed_train_core_seed42",
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
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional 403 error marker. Can be passed multiple times.",
    )
    parser.add_argument(
        "--quota-pattern",
        action="append",
        default=[],
        help="Additional quota/billing marker. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tool-name",
        action="append",
        default=[],
        help="Tool name to match. Default: web_search. Can be passed multiple times.",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Match when the same trajectory file has a target tool call and a 403 error on different lines.",
    )
    parser.add_argument(
        "--report-json",
        help="Optional path to save the matched sample report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.exists() or not output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {output_dir}")

    if args.delete and args.quarantine_dir:
        raise SystemExit("Use either --delete or --quarantine-dir, not both.")

    error_patterns = tuple(DEFAULT_ERROR_PATTERNS) + tuple(args.pattern)
    quota_patterns = tuple(DEFAULT_QUOTA_PATTERNS) + tuple(args.quota_pattern)
    tool_names = set(args.tool_name or ["web_search"])
    bad_samples = find_bad_samples(
        output_dir,
        tool_names=tool_names,
        error_patterns=error_patterns,
        quota_patterns=quota_patterns,
        loose=args.loose,
    )

    print("=== target tool 403 cleanup report ===")
    print(f"output_dir: {output_dir}")
    print(f"tool_names: {', '.join(sorted(tool_names))}")
    print(f"matched_samples: {len(bad_samples)}")
    print(f"mode: {'delete' if args.delete else ('quarantine' if args.quarantine_dir else 'dry-run')}")
    for sample_dir, reason in bad_samples:
        print(f"[MATCH] {sample_dir.name}  reason={reason}")

    if args.report_json:
        report_path = Path(args.report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [
            {"sample_dir": str(sample_dir), "sample_id": sample_dir.name, "reason": reason}
            for sample_dir, reason in bad_samples
        ]
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report_json: {report_path}")

    quarantine_dir = Path(args.quarantine_dir).expanduser() if args.quarantine_dir else None
    delete_or_quarantine_samples(
        output_dir,
        bad_samples,
        delete=args.delete,
        quarantine_dir=quarantine_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
