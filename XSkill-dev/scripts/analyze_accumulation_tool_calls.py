#!/usr/bin/env python3
"""Analyze tool-call frequency in an XSkill accumulation output directory.

The script is read-only. It scans sample rollout folders and counts structured
tool-call events in ``traj.jsonl`` / ``traj.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "output" / "xskill_accum" / "qwen3vl8b_mixed_train_core_seed42"
KNOWN_BENCHMARKS = (
    "visualtoolbench",
    "tirbench",
    "mmsearch_plus",
    "agentvista",
    "mmbrowsecomp",
)
SKIP_DIR_NAMES = {
    "__pycache__",
    "api_timings",
    "eval_dumps",
    "eval_runs",
    "logs",
    "memory_bank",
    "snapshots",
}
KNOWN_TOOLS = ("web_search", "image_search", "visit", "code_interpreter", "zoom")


@dataclass
class RolloutToolStats:
    sample_id: str
    benchmark: str
    rollout_id: str
    rollout_dir: str
    has_traj: bool
    completed: bool
    success: bool
    accuracy_score: float | None
    assistant_turns: int
    tool_calls: int
    tool_names: str
    tool_errors: int
    tool_error_names: str
    tool_error_types: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    try:
        value = read_json(path)
    except Exception:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        records = value.get("records") or value.get("trajectory") or value.get("events")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
        return [value]
    return []


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalized_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or "unknown"


def normalize_benchmark(value: Any) -> str:
    return normalized_id(value).lower().replace("-", "_")


def benchmark_from_sample(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "unknown"
    for key in ("benchmark_name", "benchmark", "data_source", "source"):
        value = sample.get(key)
        if value:
            return normalize_benchmark(value)
    return "unknown"


def build_data_lookup(data_file: Path | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not data_file or not data_file.exists():
        return lookup
    records = read_json(data_file)
    if not isinstance(records, list):
        return lookup
    for idx, sample in enumerate(records):
        if not isinstance(sample, dict):
            continue
        benchmark = benchmark_from_sample(sample)
        raw_ids = [
            sample.get("doc_id"),
            sample.get("id"),
            sample.get("question_id"),
            sample.get("original_doc_id"),
            sample.get("original_id"),
            f"sample_{idx}",
        ]
        ids = {normalized_id(item) for item in raw_ids if item is not None}
        if benchmark != "unknown":
            ids.update(f"{benchmark}_{item}" for item in list(ids) if not str(item).startswith(f"{benchmark}_"))
        for sample_id in ids:
            lookup[sample_id] = sample
    return lookup


def benchmark_from_sample_id(sample_id: str) -> str:
    lowered = sample_id.lower()
    for benchmark in KNOWN_BENCHMARKS:
        if lowered == benchmark or lowered.startswith(f"{benchmark}_"):
            return benchmark
    return "unknown"


def infer_benchmark(sample_id: str, sample_dir: Path, metrics: dict[str, Any] | None, lookup: dict[str, dict[str, Any]]) -> str:
    for payload in (metrics,):
        benchmark = benchmark_from_sample(payload)
        if benchmark != "unknown":
            return benchmark
    for path in (sample_dir / "sample.json", sample_dir / "input.json", sample_dir / "data.json"):
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            benchmark = benchmark_from_sample(payload)
            if benchmark != "unknown":
                return benchmark
    if sample_id in lookup:
        return benchmark_from_sample(lookup[sample_id])
    return benchmark_from_sample_id(sample_id)


def find_sample_dirs(run_dir: Path) -> list[Path]:
    sample_dirs = []
    if not run_dir.exists():
        return sample_dirs
    for path in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or path.name in SKIP_DIR_NAMES:
            continue
        if (path / "metrics_sample.json").exists() or (path / "metrics.json").exists():
            sample_dirs.append(path)
            continue
        if any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir()):
            sample_dirs.append(path)
    return sample_dirs


def find_rollout_dirs(sample_dir: Path) -> list[Path]:
    rollout_dirs = sorted(
        [child for child in sample_dir.iterdir() if child.is_dir() and child.name.startswith("rollout_")],
        key=lambda p: p.name,
    )
    if rollout_dirs:
        return rollout_dirs
    if (sample_dir / "traj.jsonl").exists() or (sample_dir / "traj.json").exists() or (sample_dir / "metrics.json").exists():
        return [sample_dir]
    return []


def extract_tool_name(event: dict[str, Any], *, include_text_mentions: bool = False) -> str | None:
    for key in ("tool_name", "name"):
        value = event.get(key)
        if isinstance(value, str) and value in KNOWN_TOOLS:
            return value
    for container_key in ("tool_call", "tool_error", "function_call"):
        value = event.get(container_key)
        if isinstance(value, dict):
            tool_name = value.get("tool_name") or value.get("name") or value.get("tool")
            if isinstance(tool_name, str) and tool_name:
                return tool_name
    response = event.get("response")
    if isinstance(response, dict) and response.get("tool_calls"):
        calls = response.get("tool_calls")
        if isinstance(calls, list) and calls:
            function = calls[0].get("function", {}) if isinstance(calls[0], dict) else {}
            name = function.get("name")
            if isinstance(name, str) and name:
                return name
    if include_text_mentions:
        text = json.dumps(event, ensure_ascii=False).lower()
        for name in KNOWN_TOOLS:
            if name in text:
                return name
    return None


def classify_tool_error(text: str) -> str:
    lowered = str(text or "").lower()
    if "403" in lowered or "forbidden" in lowered:
        return "http_403"
    if "404" in lowered or "not found" in lowered:
        return "http_404"
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return "http_429"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "network is unreachable" in lowered:
        return "network_unreachable"
    if "connection reset" in lowered or "connection aborted" in lowered:
        return "connection_reset"
    if "connection refused" in lowered:
        return "connection_refused"
    if "proxyerror" in lowered or "proxy error" in lowered:
        return "proxy_error"
    if "ssl" in lowered or "certificate" in lowered:
        return "ssl_error"
    if "tool" in lowered and "not found" in lowered:
        return "tool_unavailable"
    if "error" in lowered:
        return "other_error"
    return "unknown_error"


def extract_tool_error(event: dict[str, Any], *, tool_name: str | None = None) -> tuple[str | None, str | None]:
    value = event.get("tool_error")
    if isinstance(value, dict):
        name = value.get("tool_name") or value.get("name") or tool_name
        text = value.get("error") or value.get("result") or value.get("message") or json.dumps(value, ensure_ascii=False)
        return (str(name) if name else tool_name, classify_tool_error(str(text)))
    text_parts = []
    for key in ("error", "result", "observation", "text_output", "content"):
        if event.get(key):
            text_parts.append(str(event.get(key)))
    text = "\n".join(text_parts)
    if text and ("error" in text.lower() or "403" in text or "timeout" in text.lower()):
        return (tool_name, classify_tool_error(text))
    return None, None


def count_trajectory(traj_path: Path, *, include_text_mentions: bool = False) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    turns = 0
    tool_counter: Counter[str] = Counter()
    tool_error_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()
    for event in iter_json_records(traj_path):
        text = str(event.get("text_output") or event.get("response") or event.get("content") or "")
        if event.get("turn_idx") is not None and text.strip():
            turns += 1
        tool_name = extract_tool_name(event, include_text_mentions=include_text_mentions)
        if tool_name:
            tool_counter[tool_name] += 1
        error_tool_name, error_type = extract_tool_error(event, tool_name=tool_name)
        if error_type:
            tool_error_counter[error_tool_name or "unknown"] += 1
            error_type_counter[error_type] += 1
    return turns, tool_counter, tool_error_counter, error_type_counter


def format_counter(counter: Counter[str], *, limit: int = 8) -> str:
    return ", ".join(f"{name}:{count}" for name, count in counter.most_common(limit))


def parse_tool_names(value: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in str(value or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, count = item.rsplit(":", 1)
        try:
            counter[name] += int(count)
        except ValueError:
            continue
    return counter


def analyze_rollout(sample_dir: Path, rollout_dir: Path, lookup: dict[str, dict[str, Any]], threshold: float, *, include_text_mentions: bool) -> RolloutToolStats:
    sample_id = sample_dir.name
    metrics = None
    accuracy = None
    metrics_path = rollout_dir / "metrics.json"
    if metrics_path.exists():
        try:
            payload = read_json(metrics_path)
            if isinstance(payload, dict):
                metrics = payload
                accuracy = as_float(payload.get("accuracy_score"))
        except Exception:
            pass
    traj_path = rollout_dir / "traj.jsonl"
    if not traj_path.exists():
        traj_path = rollout_dir / "traj.json"
    turns, tool_counter, tool_error_counter, error_type_counter = count_trajectory(
        traj_path,
        include_text_mentions=include_text_mentions,
    )
    return RolloutToolStats(
        sample_id=sample_id,
        benchmark=infer_benchmark(sample_id, sample_dir, metrics, lookup),
        rollout_id=rollout_dir.name,
        rollout_dir=str(rollout_dir),
        has_traj=traj_path.exists(),
        completed=accuracy is not None,
        success=accuracy is not None and accuracy >= threshold,
        accuracy_score=accuracy,
        assistant_turns=turns,
        tool_calls=sum(tool_counter.values()),
        tool_names=format_counter(tool_counter, limit=20),
        tool_errors=sum(tool_error_counter.values()),
        tool_error_names=format_counter(tool_error_counter, limit=20),
        tool_error_types=format_counter(error_type_counter, limit=20),
    )


def summarize_rows(rows: list[RolloutToolStats]) -> dict[str, Any]:
    if not rows:
        return {
            "rollouts": 0,
            "tool_calls_total": 0,
            "tool_calls_mean": None,
            "tool_call_rate": None,
        }
    tool_values = [row.tool_calls for row in rows]
    turn_values = [row.assistant_turns for row in rows]
    completed = [row for row in rows if row.completed]
    success = [row for row in rows if row.success]
    counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()
    for row in rows:
        counter.update(parse_tool_names(row.tool_names))
        error_counter.update(parse_tool_names(row.tool_error_names))
        error_type_counter.update(parse_tool_names(row.tool_error_types))
    return {
        "rollouts": len(rows),
        "completed_rollouts": len(completed),
        "successful_rollouts": len(success),
        "tool_calls_total": sum(tool_values),
        "tool_calls_mean": mean(tool_values),
        "tool_call_rate": sum(1 for value in tool_values if value > 0) / len(tool_values),
        "tool_errors_total": sum(row.tool_errors for row in rows),
        "tool_error_rate": sum(1 for row in rows if row.tool_errors > 0) / len(rows),
        "assistant_turns_mean": mean(turn_values),
        "top_tools": format_counter(counter),
        "top_tool_errors": format_counter(error_counter),
        "top_tool_error_types": format_counter(error_type_counter),
    }


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n## {title}")
    if not rows:
        print("(no rows)")
        return
    widths = {
        column: max(len(column), *(len(format_value(row.get(column))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_value(row.get(column)).ljust(widths[column]) for column in columns))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[RolloutToolStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [field.name for field in RolloutToolStats.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-file", type=Path, default=None, help="Optional original train/eval JSON for benchmark lookup.")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--include-text-mentions",
        action="store_true",
        help="Also count plain text mentions of tool names. Default counts only structured tool-call events.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    lookup = build_data_lookup(args.data_file.expanduser().resolve() if args.data_file else None)

    rows: list[RolloutToolStats] = []
    for sample_dir in find_sample_dirs(run_dir):
        for rollout_dir in find_rollout_dirs(sample_dir):
            rows.append(
                analyze_rollout(
                    sample_dir,
                    rollout_dir,
                    lookup,
                    args.threshold,
                    include_text_mentions=args.include_text_mentions,
                )
            )

    overall = summarize_rows(rows)
    print(f"Run dir: {run_dir}")
    print(f"Samples: {len({row.sample_id for row in rows})}")
    print(f"Rollouts: {overall['rollouts']}")
    print(f"Tool calls total: {overall['tool_calls_total']}")
    print(f"Tool calls mean per rollout: {format_value(overall['tool_calls_mean'])}")
    print(f"Tool call rate: {format_value(overall['tool_call_rate'])}")
    print(f"Tool errors total: {overall['tool_errors_total']}")
    print(f"Tool error rate: {format_value(overall['tool_error_rate'])}")
    print(f"Assistant turns mean: {format_value(overall['assistant_turns_mean'])}")
    print(f"Top tools: {overall['top_tools']}")
    print(f"Top tool errors: {overall['top_tool_errors']}")
    print(f"Top tool error types: {overall['top_tool_error_types']}")

    by_benchmark: dict[str, list[RolloutToolStats]] = defaultdict(list)
    for row in rows:
        by_benchmark[row.benchmark].append(row)
    benchmark_rows = []
    for benchmark, group in sorted(by_benchmark.items()):
        summary = summarize_rows(group)
        benchmark_rows.append({"benchmark": benchmark, **summary})
    print_table(
        "By Benchmark",
        benchmark_rows,
        [
            "benchmark",
            "rollouts",
            "completed_rollouts",
            "successful_rollouts",
            "tool_calls_total",
            "tool_calls_mean",
            "tool_call_rate",
            "tool_errors_total",
            "tool_error_rate",
            "assistant_turns_mean",
            "top_tools",
            "top_tool_errors",
            "top_tool_error_types",
        ],
    )

    if args.csv_output:
        write_csv(args.csv_output.expanduser().resolve(), rows)
        print(f"\nWrote rollout CSV: {args.csv_output}")
    if args.json_output:
        payload = {
            "run_dir": str(run_dir),
            "overall": overall,
            "by_benchmark": benchmark_rows,
            "rollouts": [asdict(row) for row in rows],
        }
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary: {output}")


if __name__ == "__main__":
    main()
