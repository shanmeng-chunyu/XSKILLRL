#!/usr/bin/env python3
"""Summarize SkillRL validation/evaluation runs into one Markdown file.

The script is read-only except for the requested report output. It reads
SkillRL experiment_records/validation.jsonl files and, when available,
validation dump JSONL files for per-benchmark sample counts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
DEFAULT_EVAL_ROOT = MONOREPO_ROOT / "SkillRL" / "checkpoints" / "xskill_eval"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "output" / "eval_reports" / "global_val_eval_summary.md"


@dataclass
class BenchmarkMetrics:
    test_score: float | None = None
    success_rate: float | None = None
    tool_call_mean: float | None = None
    tool_call_rate: float | None = None
    tool_call_total: float | None = None
    sample_count: int | None = None
    dump_score_mean: float | None = None
    dump_tool_call_mean: float | None = None


@dataclass
class EvalRunSummary:
    run_name: str
    run_dir: Path
    record_dir: Path
    validation_file: Path
    validation_step: Any = ""
    validation_stage: str = ""
    model_path: str = ""
    skill_bank: str = "unknown"
    benchmarks: dict[str, BenchmarkMetrics] = field(default_factory=dict)
    overall_success_rate: float | None = None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _get_nested(data: Any, dotted_path: str, default: Any = "") -> Any:
    cur = data
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _latest_validation_file(record_root: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    if (record_root / "validation.jsonl").is_file():
        candidates.append(record_root / "validation.jsonl")
    if (record_root / "experiment_records").is_dir():
        candidates.extend(record_root.glob("experiment_records/*/validation.jsonl"))

    latest: tuple[Path, dict[str, Any], float] | None = None
    for path in candidates:
        rows = list(_iter_jsonl(path))
        if not rows:
            continue
        mtime = path.stat().st_mtime
        if latest is None or mtime > latest[2]:
            latest = (path, rows[-1], mtime)
    if latest is None:
        return None
    return latest[0], latest[1]


def _iter_run_dirs(eval_root: Path) -> Iterable[Path]:
    if not eval_root.exists():
        return []
    if (eval_root / "experiment_records").is_dir() or (eval_root / "validation.jsonl").is_file():
        return [eval_root]
    return [p for p in sorted(eval_root.iterdir()) if p.is_dir()]


def _infer_skill_bank(run_name: str, config: dict[str, Any]) -> str:
    lowered = run_name.lower()
    if "no_skill" in lowered or "noskill" in lowered:
        return "no"
    if "with_skill" in lowered or "skill_val" in lowered:
        return "yes"
    enabled = _get_nested(config, "env.use_skills_only_memory", "")
    if enabled is True or str(enabled).lower() == "true":
        return "yes"
    if enabled is False or str(enabled).lower() == "false":
        return "no"
    return "unknown"


def _record_dir_from_validation_file(validation_file: Path) -> Path:
    if validation_file.name == "validation.jsonl":
        return validation_file.parent
    return validation_file.parent


def _load_config(record_dir: Path) -> dict[str, Any]:
    path = record_dir / "resolved_config.json"
    if path.is_file():
        try:
            data = _read_json(path)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _extract_metric_benchmarks(metrics: dict[str, Any]) -> dict[str, BenchmarkMetrics]:
    benches: dict[str, BenchmarkMetrics] = defaultdict(BenchmarkMetrics)
    for key, value in metrics.items():
        score_match = re.fullmatch(r"val/(.+)/test_score", key)
        if score_match:
            benches[score_match.group(1)].test_score = _as_float(value)
            continue

        tool_match = re.fullmatch(r"val/(.+)/tool_call_count/mean", key)
        if tool_match:
            benches[tool_match.group(1)].tool_call_mean = _as_float(value)
            continue

        success_match = re.fullmatch(r"val/(.+)_success_rate", key)
        if success_match and success_match.group(1) != "success":
            benches[success_match.group(1)].success_rate = _as_float(value)
            continue
    return dict(benches)


def _candidate_dump_dirs(run_dir: Path, record_dir: Path, config: dict[str, Any]) -> list[Path]:
    dirs: list[Path] = []
    config_path = _get_nested(config, "trainer.validation_data_dir", "")
    if config_path:
        path = Path(str(config_path)).expanduser()
        if not path.is_absolute():
            path = run_dir / path
        dirs.append(path)
    dirs.extend(
        [
            run_dir / "validation_dump",
            run_dir / "eval_dumps",
            record_dir / "validation_dump",
        ]
    )
    seen = set()
    unique = []
    for path in dirs:
        resolved = str(path)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _dump_rows_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = _read_json(path)
    except Exception:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []

    array_keys = [k for k, v in data.items() if isinstance(v, list)]
    if not array_keys:
        return [data]
    n = max((len(data[k]) for k in array_keys), default=0)
    rows = []
    for i in range(n):
        row = {}
        for k, v in data.items():
            if isinstance(v, list):
                row[k] = v[i] if i < len(v) else None
            else:
                row[k] = v
        rows.append(row)
    return rows


def _load_dump_rows(run_dir: Path, record_dir: Path, config: dict[str, Any], step: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dump_dir in _candidate_dump_dirs(run_dir, record_dir, config):
        if not dump_dir.is_dir():
            continue
        files = list(dump_dir.rglob("*.jsonl")) + list(dump_dir.rglob("*.json"))
        if not files:
            continue
        step_files = []
        if step != "":
            step_str = str(step)
            step_files = [p for p in files if p.stem == step_str]
        selected = step_files or [max(files, key=lambda p: p.stat().st_mtime)]
        for path in selected:
            if path.suffix == ".jsonl":
                rows.extend(_iter_jsonl(path))
            elif path.suffix == ".json":
                rows.extend(_dump_rows_from_json(path))
        if rows:
            break
    return rows


def _apply_dump_counts(summary: EvalRunSummary, dump_rows: list[dict[str, Any]]) -> None:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_traj: set[tuple[str, str]] = set()

    for row in dump_rows:
        benchmark = row.get("data_source") or row.get("benchmark") or row.get("benchmark_name")
        if not benchmark:
            continue
        benchmark = str(benchmark)
        traj_uid = row.get("traj_uid") or row.get("id") or row.get("doc_id")
        if traj_uid:
            key = (benchmark, str(traj_uid))
            if key in seen_traj:
                continue
            seen_traj.add(key)
        by_benchmark[benchmark].append(row)

    for benchmark, rows in by_benchmark.items():
        metrics = summary.benchmarks.setdefault(benchmark, BenchmarkMetrics())
        metrics.sample_count = len(rows)
        scores = [_as_float(row.get("score")) for row in rows]
        scores = [v for v in scores if v is not None]
        if scores:
            metrics.dump_score_mean = mean(scores)
            if metrics.test_score is None:
                metrics.test_score = metrics.dump_score_mean

        tool_values = [
            _as_float(row.get("tool_calling", row.get("tool_call_count")))
            for row in rows
        ]
        tool_values = [v for v in tool_values if v is not None]
        if tool_values:
            metrics.dump_tool_call_mean = mean(tool_values)
            if metrics.tool_call_mean is None:
                metrics.tool_call_mean = metrics.dump_tool_call_mean
            metrics.tool_call_rate = sum(1 for value in tool_values if value > 0) / len(tool_values)
            metrics.tool_call_total = sum(tool_values)


def _summarize_run(run_dir: Path) -> EvalRunSummary | None:
    latest = _latest_validation_file(run_dir)
    if latest is None:
        return None
    validation_file, validation_row = latest
    record_dir = _record_dir_from_validation_file(validation_file)
    config = _load_config(record_dir)
    metrics = validation_row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    summary = EvalRunSummary(
        run_name=run_dir.name,
        run_dir=run_dir,
        record_dir=record_dir,
        validation_file=validation_file,
        validation_step=validation_row.get("step", ""),
        validation_stage=str(validation_row.get("stage", "")),
        model_path=str(_get_nested(config, "actor_rollout_ref.model.path", "")),
        skill_bank=_infer_skill_bank(run_dir.name, config),
        benchmarks=_extract_metric_benchmarks(metrics),
        overall_success_rate=_as_float(metrics.get("val/success_rate")),
    )
    dump_rows = _load_dump_rows(run_dir, record_dir, config, summary.validation_step)
    if dump_rows:
        _apply_dump_counts(summary, dump_rows)
    return summary


def _mean_metric(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return mean(clean)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _render_markdown(summaries: list[EvalRunSummary], eval_root: Path) -> str:
    lines = [
        "# SkillRL Evaluation Summary",
        "",
        f"- eval_root: `{eval_root}`",
        f"- runs: {len(summaries)}",
        "",
        "## Overall",
    ]

    overall_rows = []
    for summary in summaries:
        mean_score = _mean_metric(m.test_score for m in summary.benchmarks.values())
        mean_tool = _mean_metric(m.tool_call_mean for m in summary.benchmarks.values())
        mean_tool_rate = _mean_metric(m.tool_call_rate for m in summary.benchmarks.values())
        tool_total = sum(m.tool_call_total or 0 for m in summary.benchmarks.values()) or ""
        sample_total = sum(m.sample_count or 0 for m in summary.benchmarks.values()) or ""
        overall_rows.append(
            [
                summary.run_name,
                summary.validation_step,
                summary.validation_stage,
                summary.skill_bank,
                _fmt(mean_score),
                _fmt(summary.overall_success_rate),
                _fmt(mean_tool),
                _fmt(mean_tool_rate),
                _fmt(tool_total),
                sample_total,
                len(summary.benchmarks),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Run",
                "Step",
                "Stage",
                "SkillBank",
                "Mean score",
                "Success rate",
                "Tool calls mean",
                "Tool call rate",
                "Tool calls total",
                "Samples",
                "Benchmarks",
            ],
            overall_rows or [["", "", "", "", "", "", "", "", "", "", ""]],
        )
    )

    lines.extend(["", "## Per Benchmark"])
    benchmark_rows = []
    for summary in summaries:
        for benchmark in sorted(summary.benchmarks):
            metrics = summary.benchmarks[benchmark]
            benchmark_rows.append(
                [
                    summary.run_name,
                    benchmark,
                    metrics.sample_count or "",
                    _fmt(metrics.test_score),
                    _fmt(metrics.success_rate),
                    _fmt(metrics.tool_call_mean),
                    _fmt(metrics.tool_call_rate),
                    _fmt(metrics.tool_call_total),
                ]
            )
    lines.append(
        _markdown_table(
            [
                "Run",
                "Benchmark",
                "Samples",
                "Accuracy",
                "Success rate",
                "Tool calls mean",
                "Tool call rate",
                "Tool calls total",
            ],
            benchmark_rows or [["", "", "", "", "", "", "", ""]],
        )
    )

    lines.extend(["", "## Source Files"])
    file_rows = [
        [
            summary.run_name,
            str(summary.validation_file),
            str(summary.record_dir),
            summary.model_path,
        ]
        for summary in summaries
    ]
    lines.append(
        _markdown_table(
            ["Run", "Validation JSONL", "Record dir", "Model"],
            file_rows or [["", "", "", ""]],
        )
    )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--run-name-regex",
        default="",
        help="Optional regex filter for run directory names.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the generated Markdown to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    run_dirs = list(_iter_run_dirs(eval_root))
    if args.run_name_regex:
        pattern = re.compile(args.run_name_regex)
        run_dirs = [p for p in run_dirs if pattern.search(p.name)]

    summaries = []
    for run_dir in run_dirs:
        summary = _summarize_run(run_dir)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=lambda item: item.run_name)
    markdown = _render_markdown(summaries, eval_root)

    output_md = args.output_md.expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    if args.print:
        print(markdown)
    print(f"Wrote evaluation summary: {output_md}")


if __name__ == "__main__":
    main()
