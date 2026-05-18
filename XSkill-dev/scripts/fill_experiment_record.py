"""Fill experiment record markdown from XSkill accumulation artifacts.

The script updates selected tables in docs/EXPERIMENT_RECORD_TEMPLATE_CN.md.
It is intentionally read-only with respect to experiment artifacts; only the
target markdown file is modified when --in-place is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_NAME = "qwen3vl8b_mixed_train_core_seed42"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return ""
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}min"
    return f"{value / 3600:.2f}h"


def strip_ticks(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        return value[1:-1]
    return value


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table_row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def find_heading_line(lines: list[str], heading: str) -> int | None:
    target = heading.strip()
    for i, line in enumerate(lines):
        if line.strip() == target:
            return i
    return None


def find_first_table(lines: list[str], start: int) -> tuple[int, int] | None:
    table_start = None
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#") and table_start is None:
            return None
        if stripped.startswith("|"):
            table_start = i
            break
    if table_start is None:
        return None
    table_end = table_start
    while table_end < len(lines) and lines[table_end].strip().startswith("|"):
        table_end += 1
    return table_start, table_end


def update_table_rows(markdown: str, heading: str, updates: dict[str, list[Any]]) -> str:
    """Update rows in the first markdown table after heading.

    Keys are matched against the first cell after stripping backticks.
    The number of columns is preserved where possible.
    """
    lines = markdown.splitlines()
    hidx = find_heading_line(lines, heading)
    if hidx is None:
        return markdown
    bounds = find_first_table(lines, hidx)
    if bounds is None:
        return markdown
    start, end = bounds
    for i in range(start + 2, end):
        cells = split_table_row(lines[i])
        if not cells:
            continue
        key = strip_ticks(cells[0])
        if key in updates:
            new_cells = [cells[0]] + [str(x) for x in updates[key]]
            if len(new_cells) < len(cells):
                new_cells.extend(cells[len(new_cells):])
            lines[i] = make_table_row(new_cells[: len(cells)])
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def replace_table(markdown: str, heading: str, header: list[str], rows: list[list[Any]]) -> str:
    lines = markdown.splitlines()
    hidx = find_heading_line(lines, heading)
    if hidx is None:
        return markdown
    bounds = find_first_table(lines, hidx)
    if bounds is None:
        return markdown
    start, end = bounds
    new_table = [
        make_table_row(header),
        make_table_row(["---"] * len(header)),
    ]
    new_table.extend(make_table_row(row) for row in rows)
    lines[start:end] = new_table
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def normalized_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or "unknown"


def output_sample_ids(sample: dict[str, Any], idx: int) -> set[str]:
    raw = normalized_id(sample.get("doc_id", sample.get("question_id", f"sample_{idx}")))
    benchmark = sample.get("benchmark_name") or sample.get("benchmark") or sample.get("source")
    ids = {raw}
    if benchmark:
        prefix = normalized_id(benchmark).lower()
        raw_lower = raw.lower()
        if raw_lower == prefix or raw_lower.startswith(f"{prefix}_"):
            ids.add(raw)
        else:
            ids.add(f"{prefix}_{raw}")
    return ids


def build_sample_lookup(data_path: Path) -> tuple[int | None, dict[str, dict[str, Any]], Counter]:
    if not data_path.exists():
        return None, {}, Counter()
    data = read_json(data_path)
    if not isinstance(data, list):
        return None, {}, Counter()
    lookup: dict[str, dict[str, Any]] = {}
    benchmark_counts: Counter = Counter()
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        benchmark = str(sample.get("benchmark_name") or sample.get("benchmark") or "unknown")
        benchmark_counts[benchmark] += 1
        for sid in output_sample_ids(sample, idx):
            lookup.setdefault(sid, sample)
    return len(data), lookup, benchmark_counts


def iter_sample_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted([p for p in output_dir.iterdir() if p.is_dir()])


def load_sample_summaries(output_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    summaries = []
    for sample_dir in iter_sample_dirs(output_dir):
        summary_path = sample_dir / "metrics_sample.json"
        if not summary_path.exists():
            continue
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        if isinstance(summary, dict):
            summaries.append((sample_dir, summary))
    return summaries


def count_exp_items(output_dir: Path) -> int:
    total = 0
    for path in output_dir.glob("*/exp_items.json"):
        try:
            items = read_json(path)
        except Exception:
            continue
        if isinstance(items, list):
            total += len(items)
        elif isinstance(items, dict):
            for key in ("experiences", "items", "ops"):
                if isinstance(items.get(key), list):
                    total += len(items[key])
                    break
    return total


def count_experience_library(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("experiences", "items", "data"):
            if isinstance(data.get(key), list):
                return len(data[key])
        return len(data)
    return None


def count_skill_words(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = re.findall(r"\S+", text)
    return len(words)


def file_size_text(path: Path) -> str:
    if not path.exists():
        return "not found"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def rel_text(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def classify_final_answer(answer: str) -> str | None:
    text = (answer or "").lower()
    if "could not parse model response" in text:
        return "`Error: Could not parse model response`"
    if "reached max token limit" in text:
        return "`Error: Reached max token limit`"
    if "image not found" in text or "need " in text and "image" in text and "none provided" in text:
        return "图片找不到"
    if "api" in text and ("failed" in text or "timeout" in text or "exception" in text):
        return "API timeout / 连接失败"
    if "tool" in text and ("failed" in text or "error" in text):
        return "工具调用失败"
    return None


def collect_failure_stats(output_dir: Path, log_path: Path | None) -> dict[str, dict[str, Any]]:
    labels = [
        "`Error: Could not parse model response`",
        "`Error: Reached max token limit`",
        "输出中途截断",
        "API timeout / 连接失败",
        "工具调用失败",
        "图片找不到",
        "verifier 判分异常",
        "experience embedding 失败",
        "retrieval 无结果",
    ]
    stats = {label: {"count": 0, "examples": []} for label in labels}

    for metrics_path in output_dir.glob("*/rollout_*/metrics.json"):
        sample_id = metrics_path.parent.parent.name
        try:
            metrics = read_json(metrics_path)
        except Exception:
            continue
        label = classify_final_answer(str(metrics.get("final_answer", "")))
        if label:
            stats[label]["count"] += 1
            if len(stats[label]["examples"]) < 5:
                stats[label]["examples"].append(sample_id)

    if log_path and log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lower = line.lower()
                if "embedding failed" in lower or "failed to merge experiences" in lower:
                    label = "experience embedding 失败"
                elif "no relevant experiences found" in lower:
                    label = "retrieval 无结果"
                elif "error: image not found" in lower:
                    label = "图片找不到"
                elif "verifier" in lower and ("error" in lower or "failed" in lower):
                    label = "verifier 判分异常"
                else:
                    continue
                stats[label]["count"] += 1
    return stats


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[int(pos)]
    return values[low] * (high - pos) + values[high] * (pos - low)


def get_first(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def summarize_timings(api_timings_path: Path) -> tuple[list[list[Any]], list[list[Any]]]:
    rows = read_jsonl(api_timings_path)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        kind = str(get_first(row, ("kind", "request_kind", "type", "name", "tool"), "unknown"))
        endpoint = str(get_first(row, ("endpoint", "url", "end_point"), "unknown"))
        by_kind[kind].append(row)
        by_endpoint[endpoint].append(row)

    preferred_kinds = [
        "reasoning",
        "verifier",
        "experience",
        "image caption",
        "web_search",
        "image_search",
        "visit",
        "code_interpreter / zoom",
    ]

    def latency(row: dict[str, Any]) -> float | None:
        value = get_first(row, ("latency", "latency_seconds", "elapsed", "elapsed_seconds", "duration"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def failed(row: dict[str, Any]) -> bool:
        status = get_first(row, ("status", "status_code", "ok", "success"), "")
        if isinstance(status, bool):
            return not status
        status_text = str(status).lower()
        if status_text in {"ok", "success", "200"}:
            return False
        if status_text and status_text not in {"none", ""}:
            try:
                return int(status_text) >= 400
            except ValueError:
                return status_text in {"error", "failed", "timeout"}
        return bool(row.get("error"))

    def token_avg(rows_for_group: list[dict[str, Any]], names: tuple[str, ...]) -> str:
        vals = []
        for row in rows_for_group:
            value = get_first(row, names)
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                pass
        return fmt_float(mean(vals), 1) if vals else ""

    kind_rows = []
    all_kinds = list(dict.fromkeys(preferred_kinds + sorted(by_kind)))
    for kind in all_kinds:
        group = by_kind.get(kind, [])
        latencies = [x for x in (latency(row) for row in group) if x is not None]
        kind_rows.append(
            [
                kind,
                len(group) if group else "",
                fmt_seconds(mean(latencies)) if latencies else "",
                fmt_seconds(quantile(latencies, 0.5)) if latencies else "",
                fmt_seconds(quantile(latencies, 0.95)) if latencies else "",
                sum(1 for row in group if failed(row)) if group else "",
                token_avg(group, ("prompt_tokens", "input_tokens")),
                token_avg(group, ("completion_tokens", "output_tokens")),
            ]
        )

    endpoint_rows = []
    for endpoint, group in sorted(by_endpoint.items()):
        latencies = [x for x in (latency(row) for row in group) if x is not None]
        unhealthy = sum(1 for row in group if str(row.get("event", "")).lower() == "unhealthy")
        endpoint_rows.append(
            [
                endpoint,
                len(group),
                fmt_seconds(mean(latencies)) if latencies else "",
                sum(1 for row in group if failed(row)),
                unhealthy,
                "",
            ]
        )
    if not endpoint_rows:
        endpoint_rows = [["", "", "", "", "", ""]]
    return kind_rows, endpoint_rows


def average_summary_values(summaries: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for summary in summaries:
        value = summary.get(key)
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            pass
    return mean(vals) if vals else None


def benchmark_from_sample(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "unknown"
    return str(sample.get("benchmark_name") or sample.get("benchmark") or "unknown")


def collect_benchmark_rows(
    summaries: list[tuple[Path, dict[str, Any]]],
    sample_lookup: dict[str, dict[str, Any]],
) -> list[list[Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_dir, summary in summaries:
        sid = str(summary.get("question_id") or sample_dir.name)
        sample = sample_lookup.get(sid) or sample_lookup.get(sample_dir.name)
        groups[benchmark_from_sample(sample)].append(summary)
    rows = []
    preferred = ["visualtoolbench", "tirbench", "mmsearch_plus", "agentvista", "mmbrowsecomp"]
    for benchmark in list(dict.fromkeys(preferred + sorted(groups))):
        group = groups.get(benchmark, [])
        if not group:
            display = {
                "visualtoolbench": "VisualToolBench",
                "tirbench": "TIR-Bench",
                "mmsearch_plus": "MMSearch-Plus",
                "agentvista": "AgentVista",
                "mmbrowsecomp": "MMBrowseComp",
            }.get(benchmark, benchmark)
            rows.append([display, "", "", "", "", "", "", ""])
            continue
        k_values = [int(x.get("num_rollouts", 0) or 0) for x in group]
        k = max(k_values) if k_values else 2
        display = {
            "visualtoolbench": "VisualToolBench",
            "tirbench": "TIR-Bench",
            "mmsearch_plus": "MMSearch-Plus",
            "agentvista": "AgentVista",
            "mmbrowsecomp": "MMBrowseComp",
        }.get(benchmark, benchmark)
        rows.append(
            [
                display,
                len(group),
                fmt_float(average_summary_values(group, "pass@1")),
                fmt_float(average_summary_values(group, "average@1")),
                fmt_float(average_summary_values(group, f"pass@{k}")),
                fmt_float(average_summary_values(group, f"average@{k}")),
                "",
                "",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill XSkill experiment record markdown from saved artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--record-md", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--api-timings", default=None)
    parser.add_argument("--in-place", action="store_true", help="Overwrite --record-md. Without this, print to stdout.")
    parser.add_argument("--output-md", default=None, help="Write filled markdown to a new file.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    exp_name = args.exp_name
    record_md = Path(args.record_md) if args.record_md else project_root / "docs" / "EXPERIMENT_RECORD_TEMPLATE_CN.md"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "output" / "xskill_accum" / exp_name
    memory_dir = Path(args.memory_dir) if args.memory_dir else project_root / "memory_bank" / "xskill_accum" / exp_name
    log_path = Path(args.log_path) if args.log_path else project_root / "logs" / "xskill_accum" / f"{exp_name}.log"
    data_path = Path(args.data_path) if args.data_path else project_root / "benchmark" / "_mixed_protocol" / "train_core.json"
    api_timings = Path(args.api_timings) if args.api_timings else output_dir / "api_timings.jsonl"

    markdown = record_md.read_text(encoding="utf-8")

    total_expected, sample_lookup, _benchmark_counts = build_sample_lookup(data_path)
    summary_pairs = load_sample_summaries(output_dir)
    summaries = [summary for _, summary in summary_pairs]
    processed = len(summaries)
    total_rollouts = sum(int(x.get("num_rollouts", 0) or 0) for x in summaries)
    max_k = max([int(x.get("num_rollouts", 0) or 0) for x in summaries] or [2])
    failed = max(total_expected - processed, 0) if total_expected is not None and output_dir.exists() else ""

    fallback_memory_dir = project_root / "memory_bank" / "test"
    exp_library = first_existing_path(memory_dir / "experiences.json", fallback_memory_dir / "experiences.json")
    skill_file = first_existing_path(memory_dir / "SKILL.md", fallback_memory_dir / "SKILL.md")
    skillrl_bank = first_existing_path(memory_dir / "skillrl_skill_bank.json", fallback_memory_dir / "skillrl_skill_bank.json")
    generated_exp_count = count_exp_items(output_dir)
    library_count = count_experience_library(exp_library)
    skill_words = count_skill_words(skill_file)

    overall_updates = {
        "处理样本数": [processed],
        "成功样本数": [sum(1 for x in summaries if float(x.get(f"pass@{int(x.get('num_rollouts', 0) or max_k)}", 0) or 0) > 0)],
        "失败样本数": [failed],
        "总 rollout 数": [total_rollouts],
        "平均每样本耗时": [""],
        "总耗时": [""],
        "pass@1": [fmt_float(average_summary_values(summaries, "pass@1"))],
        "average@1": [fmt_float(average_summary_values(summaries, "average@1"))],
        "pass@2 / pass@k": [fmt_float(average_summary_values(summaries, f"pass@{max_k}"))],
        "average@2 / average@k": [fmt_float(average_summary_values(summaries, f"average@{max_k}"))],
        "生成 experience 数": [generated_exp_count],
        "最终 experience library 数": [library_count if library_count is not None else ""],
        "最终 skill 字数": [skill_words if skill_words is not None else ""],
    }
    markdown = update_table_rows(markdown, "### 5.2 Accumulation 总体结果", overall_updates)

    benchmark_rows = collect_benchmark_rows(summary_pairs, sample_lookup)
    markdown = replace_table(
        markdown,
        "### 5.3 Accumulation 分 Benchmark 结果",
        ["Benchmark", "样本数", "pass@1", "average@1", "pass@k", "average@k", "平均耗时", "主要失败原因"],
        benchmark_rows,
    )

    failure_stats = collect_failure_stats(output_dir, log_path)
    denominator = total_rollouts or 1
    failure_rows = []
    for label, stats in failure_stats.items():
        count = stats["count"]
        examples = ", ".join(stats["examples"])
        failure_rows.append([label, count, fmt_float(count / denominator, 4), examples, ""])
    markdown = replace_table(
        markdown,
        "### 5.4 失败类型统计",
        ["失败类型", "数量", "占比", "代表样本", "备注"],
        failure_rows,
    )

    timing_rows, endpoint_rows = summarize_timings(api_timings)
    markdown = replace_table(
        markdown,
        "### 5.5 API Timing 与吞吐",
        ["请求类型", "请求数", "平均耗时", "P50", "P95", "失败数", "平均 prompt tokens", "平均 completion tokens"],
        timing_rows,
    )

    markdown = replace_table(
        markdown,
        "Endpoint 使用情况：",
        ["Endpoint", "请求数", "平均耗时", "失败数", "unhealthy 次数", "备注"],
        endpoint_rows,
    )

    exp_count_text = "" if library_count is None else str(library_count)
    skill_words_text = "" if skill_words is None else f"{skill_words} words"
    memory_updates = {
        "experiences.json": [f"`{rel_text(exp_library, project_root)}`", "; ".join(x for x in [exp_count_text, file_size_text(exp_library)] if x), "XSkill accumulation 输出"],
        "SKILL.md": [f"`{rel_text(skill_file, project_root)}`", "; ".join(x for x in [skill_words_text, file_size_text(skill_file)] if x), "XSkill accumulation 输出"],
        "skillrl_skill_bank.json": [f"`{rel_text(skillrl_bank, project_root)}`", file_size_text(skillrl_bank), "给 SkillRL/GRPO 使用"],
    }
    markdown = update_table_rows(markdown, "## 6. Memory Bank 记录", memory_updates)

    if args.in_place:
        record_md.write_text(markdown, encoding="utf-8")
        print(f"Updated {record_md}")
    elif args.output_md:
        target = Path(args.output_md)
        target.write_text(markdown, encoding="utf-8")
        print(f"Wrote {target}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
