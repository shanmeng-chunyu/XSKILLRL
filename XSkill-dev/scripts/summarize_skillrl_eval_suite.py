#!/usr/bin/env python3
"""Summarize multiple SkillRL XSkill validation runs into one Markdown file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
DEFAULT_EVAL_ROOT = MONOREPO_ROOT / "SkillRL" / "checkpoints" / "xskill_eval"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def safe_run_name(path: Path) -> str:
    return path.name


def find_latest_record_dir(run_dir: Path) -> Path | None:
    record_root = run_dir / "experiment_records"
    if not record_root.exists():
        return None
    candidates = [path for path in record_root.iterdir() if path.is_dir() and (path / "validation.jsonl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_config(record_dir: Path | None) -> dict[str, Any]:
    if not record_dir:
        return {}
    path = record_dir / "resolved_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def select_path(payload: dict[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def latest_validation(record_dir: Path | None) -> dict[str, Any] | None:
    if not record_dir:
        return None
    rows = list(iter_jsonl(record_dir / "validation.jsonl"))
    return rows[-1] if rows else None


def infer_group(run_name: str, config: dict[str, Any]) -> tuple[str, str]:
    model_kind = "unknown"
    lowered = run_name.lower()
    if lowered.startswith("base_"):
        model_kind = "base"
    elif lowered.startswith("sft_"):
        model_kind = "sft"
    elif lowered.startswith("rl_"):
        model_kind = "rl"

    has_skill = bool(select_path(config, "env.use_skills_only_memory", False))
    if "with_skill" in lowered:
        has_skill = True
    elif "no_skill" in lowered:
        has_skill = False
    return model_kind, "with_skill" if has_skill else "no_skill"


def sample_count_from_validation_dir(run_dir: Path, benchmark: str) -> int | None:
    validation_dir = run_dir / "validation_dump"
    if not validation_dir.exists():
        return None
    count = 0
    for path in validation_dir.glob("*.jsonl"):
        for row in iter_jsonl(path):
            source = row.get("data_source") or row.get("benchmark") or row.get("benchmark_name")
            if source == benchmark:
                count += 1
    return count or None


def extract_benchmark_rows(run_dir: Path, record_dir: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_config(record_dir)
    val = latest_validation(record_dir)
    metrics = val.get("metrics", {}) if isinstance(val, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    run_name = safe_run_name(run_dir)
    model_kind, skill_mode = infer_group(run_name, config)
    run_summary = {
        "run": run_name,
        "model": model_kind,
        "skill": skill_mode,
        "stage": val.get("stage", "") if val else "",
        "step": val.get("step", "") if val else "",
        "model_path": select_path(config, "actor_rollout_ref.model.path", ""),
        "val_file": select_path(config, "data.val_files", ""),
        "record_dir": str(record_dir) if record_dir else "",
    }

    benchmarks = set()
    for key in metrics:
        match = re.match(r"val/([^/]+)/(.+)$", str(key))
        if match:
            benchmarks.add(match.group(1))

    rows = []
    for benchmark in sorted(benchmarks):
        test_score = as_float(metrics.get(f"val/{benchmark}/test_score"))
        success_rate = as_float(metrics.get(f"val/{benchmark}_success_rate"))
        tool_mean = as_float(metrics.get(f"val/{benchmark}/tool_call_count/mean"))
        sample_count = sample_count_from_validation_dir(run_dir, benchmark)
        rows.append(
            {
                "run": run_name,
                "model": model_kind,
                "skill": skill_mode,
                "benchmark": benchmark,
                "test_score": test_score,
                "success_rate": success_rate,
                "tool_call_mean": tool_mean,
                "sample_count": sample_count,
            }
        )
    return run_summary, rows


def discover_runs(eval_root: Path) -> list[Path]:
    if not eval_root.exists():
        return []
    return sorted([path for path in eval_root.iterdir() if path.is_dir()], key=lambda path: path.name)


def summarize_overall(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in benchmark_rows:
        grouped.setdefault(str(row["run"]), []).append(row)
    rows = []
    for run, group in sorted(grouped.items()):
        score_values = [value for value in (as_float(row.get("test_score")) for row in group) if value is not None]
        success_values = [value for value in (as_float(row.get("success_rate")) for row in group) if value is not None]
        tool_values = [value for value in (as_float(row.get("tool_call_mean")) for row in group) if value is not None]
        sample_values = [value for value in (as_float(row.get("sample_count")) for row in group) if value is not None]
        first = group[0]
        rows.append(
            {
                "run": run,
                "model": first.get("model"),
                "skill": first.get("skill"),
                "benchmarks": len(group),
                "macro_test_score": mean(score_values) if score_values else None,
                "macro_success_rate": mean(success_values) if success_values else None,
                "tool_call_mean": mean(tool_values) if tool_values else None,
                "sample_count": int(sum(sample_values)) if sample_values else "",
            }
        )
    return rows


def make_markdown(eval_root: Path, run_rows: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SkillRL Global Val Evaluation Summary",
        "",
        f"- eval_root: `{eval_root}`",
        f"- runs: `{len(run_rows)}`",
        "",
        "## Overall",
        "",
        markdown_table(
            summarize_overall(benchmark_rows),
            [
                "run",
                "model",
                "skill",
                "benchmarks",
                "macro_test_score",
                "macro_success_rate",
                "tool_call_mean",
                "sample_count",
            ],
        ),
        "",
        "## By Benchmark",
        "",
        markdown_table(
            benchmark_rows,
            [
                "run",
                "model",
                "skill",
                "benchmark",
                "test_score",
                "success_rate",
                "tool_call_mean",
                "sample_count",
            ],
        ),
        "",
        "## Runs",
        "",
        markdown_table(
            run_rows,
            ["run", "model", "skill", "stage", "step", "model_path", "val_file", "record_dir"],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser()
    run_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for run_dir in discover_runs(eval_root):
        record_dir = find_latest_record_dir(run_dir)
        run_summary, rows = extract_benchmark_rows(run_dir, record_dir)
        if not rows and not record_dir:
            continue
        run_rows.append(run_summary)
        benchmark_rows.extend(rows)

    markdown = make_markdown(eval_root, run_rows, benchmark_rows)
    print(markdown)
    if args.output_md:
        output_path = Path(args.output_md).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote Markdown summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
