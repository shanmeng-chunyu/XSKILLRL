#!/usr/bin/env python3
"""Summarize an XSkill accumulation run into Markdown/JSON/CSV reports.

The script is read-only unless output paths are provided. It scans the
accumulation output directory, optional original data split, optional memory
bank, and optional API timing log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "qwen3vl8b_mixed_train_core_seed42"
SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
KNOWN_BENCHMARKS = (
    "visualtoolbench",
    "tirbench",
    "mmsearch_plus",
    "agentvista",
    "mmbrowsecomp",
)
SKIP_OUTPUT_DIRS = {
    "__pycache__",
    "api_timings",
    "eval_dumps",
    "eval_runs",
    "logs",
    "memory_bank",
    "snapshots",
}
FAILURE_PATTERNS = (
    ("image_missing", ("image not found", "no images found", "failed to load image", "filenotfounderror")),
    ("max_tokens", ("max token", "maximum context", "longer than the maximum model length")),
    ("api_timeout_or_network", ("all api attempts failed", "api attempts failed", "api timeout", "connection reset", "network")),
    ("max_turns", ("reached max turns", "without a definitive answer")),
    ("tool_error", ("tool_error", "received valid output (type: error)", "tool call failed")),
    ("verifier_error", ("evaluation failed", "verifier", "judge")),
    ("parse_error", ("could not parse model response", "failed to parse model response")),
    ("error_final_answer", ("final_answer\": \"error", "error:")),
)


@dataclass
class RolloutRow:
    sample_id: str
    benchmark: str
    rollout_id: str
    rollout_dir: str
    completed: bool
    accuracy_score: float | None
    success: bool
    final_answer: str
    failure_type: str
    assistant_turns: int
    tool_calls: int
    tool_names: str
    error: str


@dataclass
class SampleRow:
    sample_id: str
    benchmark: str
    sample_dir: str
    completed_rollouts: int
    expected_rollouts: int
    pass_at_1: float | None
    average_at_1: float | None
    pass_at_k: float | None
    average_at_k: float | None
    rollout_accuracy_mean: float | None
    best_accuracy: float | None
    failure_types: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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


def normalized_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or "unknown"


def normalize_benchmark(value: Any) -> str:
    return normalized_id(value).lower().replace("-", "_")


def sample_from_record(record: dict[str, Any]) -> dict[str, Any]:
    env_kwargs = record.get("env_kwargs")
    if isinstance(env_kwargs, dict):
        return env_kwargs
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        sample = extra_info.get("sample")
        if isinstance(sample, dict):
            return sample
    return record


def benchmark_from_sample(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "unknown"
    for key in ("benchmark_name", "benchmark", "data_source", "source"):
        value = sample.get(key)
        if value:
            return normalize_benchmark(value)
    return "unknown"


def output_sample_ids(sample: dict[str, Any], idx: int) -> set[str]:
    raw = normalized_id(sample.get("doc_id", sample.get("question_id", sample.get("id", f"sample_{idx}"))))
    benchmark = benchmark_from_sample(sample)
    ids = {raw}
    if benchmark != "unknown":
        raw_lower = raw.lower()
        if raw_lower == benchmark or raw_lower.startswith(f"{benchmark}_"):
            ids.add(raw)
        else:
            ids.add(f"{benchmark}_{raw}")
    return ids


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "samples", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []
    if suffix == ".jsonl":
        return [item for item in iter_jsonl(path)]
    if suffix == ".parquet":
        import pandas as pd

        return [item for item in pd.read_parquet(path).to_dict(orient="records") if isinstance(item, dict)]
    raise ValueError(f"Unsupported data file type: {path}")


def build_sample_lookup(data_files: list[str]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    lookup: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for item in data_files:
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.exists():
            print(f"[WARN] data file not found: {path}")
            continue
        for idx, record in enumerate(load_records(path)):
            sample = sample_from_record(record)
            benchmark = benchmark_from_sample(sample)
            counts[benchmark] += 1
            for sample_id in output_sample_ids(sample, idx):
                lookup.setdefault(sample_id, sample)
    return lookup, counts


def discover_sample_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    sample_dirs = []
    for path in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_OUTPUT_DIRS:
            continue
        if (
            (path / SAMPLE_SUMMARY_FILENAME).exists()
            or (path / "metrics.json").exists()
            or any(child.is_dir() and child.name.startswith("rollout_") for child in path.iterdir())
        ):
            sample_dirs.append(path)
    return sample_dirs


def rollout_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("_", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return 10**9, path.name


def discover_rollout_dirs(sample_dir: Path) -> list[Path]:
    rollout_dirs = sorted(
        [path for path in sample_dir.iterdir() if path.is_dir() and path.name.startswith("rollout_")],
        key=rollout_sort_key,
    )
    if rollout_dirs:
        return rollout_dirs
    if (sample_dir / "metrics.json").exists() or (sample_dir / "traj.jsonl").exists():
        return [sample_dir]
    return []


def infer_benchmark(sample_id: str, sample_dir: Path, metrics: dict[str, Any] | None, lookup: dict[str, dict[str, Any]]) -> str:
    for source in (metrics or {},):
        benchmark = benchmark_from_sample(source)
        if benchmark != "unknown":
            return benchmark

    sample = lookup.get(sample_id) or lookup.get(sample_dir.name)
    benchmark = benchmark_from_sample(sample)
    if benchmark != "unknown":
        return benchmark

    for filename in ("sample.json", "metadata.json", "input_sample.json", "request.json"):
        path = sample_dir / filename
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

    lowered = sample_id.lower()
    for prefix in KNOWN_BENCHMARKS:
        if lowered == prefix or lowered.startswith(f"{prefix}_"):
            return prefix
    return "unknown"


def extract_tool_name(event: dict[str, Any]) -> str | None:
    for key in ("tool_name", "name"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    for container_key in ("tool_call", "tool_error"):
        value = event.get(container_key)
        if isinstance(value, dict):
            tool_name = value.get("tool_name") or value.get("name")
            if isinstance(tool_name, str) and tool_name:
                return tool_name
    text = json.dumps(event, ensure_ascii=False).lower()
    for name in ("web_search", "image_search", "visit", "code_interpreter", "zoom"):
        if name in text:
            return name
    return None


def count_traj(traj_path: Path) -> tuple[int, Counter[str], str]:
    turns = 0
    tool_names: Counter[str] = Counter()
    snippets: list[str] = []
    if not traj_path.exists():
        return turns, tool_names, ""
    for event in iter_jsonl(traj_path):
        text = str(event.get("text_output") or event.get("error") or "")
        if event.get("turn_idx") is not None and text.strip():
            turns += 1
        tool_name = extract_tool_name(event)
        if tool_name:
            tool_names[tool_name] += 1
        if len(snippets) < 6 and text:
            snippets.append(text[:500])
        elif len(snippets) < 6 and tool_name:
            snippets.append(json.dumps(event, ensure_ascii=False)[:500])
    return turns, tool_names, "\n".join(snippets)


def classify_failure(metrics: dict[str, Any] | None, traj_snippet: str, error: str) -> str:
    final_answer = str((metrics or {}).get("final_answer") or "").strip().lower()
    trajectory_analysis = str((metrics or {}).get("trajectory_analysis") or "").strip().lower()
    error_text = str(error or "").strip().lower()
    primary_text = " ".join(part for part in (error_text, final_answer, trajectory_analysis) if part)
    text = " ".join(
        str(part or "")
        for part in (
            primary_text,
            traj_snippet,
        )
    ).lower()
    if not text:
        return ""

    # The trajectory often contains parser-correction helper messages even when
    # the actual terminal failure is max turns, API failure, or a wrong answer.
    # Classify from terminal fields first and avoid treating helper corrections
    # as parse failures by themselves.
    for label, markers in FAILURE_PATTERNS:
        if label == "parse_error":
            if any(marker in primary_text for marker in markers):
                return label
            continue
        if any(marker in primary_text for marker in markers):
            return label

    if final_answer and not final_answer.startswith("error:"):
        return "incorrect_answer"

    for label, markers in FAILURE_PATTERNS:
        if label == "parse_error":
            continue
        if any(marker in text for marker in markers):
            return label
    if "error" in primary_text:
        return "other_error"
    return "incorrect_answer"


def load_summary_accuracies(sample_dir: Path) -> tuple[list[float], int]:
    path = sample_dir / SAMPLE_SUMMARY_FILENAME
    if not path.exists():
        return [], 0
    try:
        payload = read_json(path)
    except Exception:
        return [], 0
    if not isinstance(payload, dict):
        return [], 0
    raw = payload.get("accuracies")
    accuracies = [score for score in (as_float(value) for value in raw or []) if score is not None]
    expected = int(as_float(payload.get("num_rollouts")) or len(accuracies))
    return accuracies, expected


def analyze_rollout(sample_id: str, sample_dir: Path, rollout_dir: Path, threshold: float, lookup: dict[str, dict[str, Any]]) -> RolloutRow:
    metrics_path = rollout_dir / "metrics.json"
    traj_path = rollout_dir / "traj.jsonl"
    turns, tool_counter, traj_snippet = count_traj(traj_path)
    error = ""
    metrics: dict[str, Any] | None = None
    accuracy = None
    final_answer = ""
    completed = False

    if metrics_path.exists():
        try:
            payload = read_json(metrics_path)
            if isinstance(payload, dict):
                metrics = payload
                accuracy = as_float(payload.get("accuracy_score"))
                final_answer = str(payload.get("final_answer") or "")
                completed = accuracy is not None
            else:
                error = "metrics.json is not an object"
        except Exception as exc:
            error = f"failed to read metrics.json: {exc}"
    else:
        error = "missing metrics.json"

    benchmark = infer_benchmark(sample_id, sample_dir, metrics, lookup)
    success = accuracy is not None and accuracy >= threshold
    failure_type = "" if success else classify_failure(metrics, traj_snippet, error)
    return RolloutRow(
        sample_id=sample_id,
        benchmark=benchmark,
        rollout_id=rollout_dir.name,
        rollout_dir=str(rollout_dir),
        completed=completed,
        accuracy_score=accuracy,
        success=success,
        final_answer=final_answer,
        failure_type=failure_type,
        assistant_turns=turns,
        tool_calls=sum(tool_counter.values()),
        tool_names=",".join(f"{name}:{count}" for name, count in sorted(tool_counter.items())),
        error=error,
    )


def analyze_sample(sample_dir: Path, rollouts: list[RolloutRow], k: int | None) -> SampleRow:
    sample_id = sample_dir.name
    benchmark = rollouts[0].benchmark if rollouts else "unknown"
    summary_accuracies, summary_expected = load_summary_accuracies(sample_dir)
    accuracies = [row.accuracy_score for row in rollouts if row.accuracy_score is not None]
    if not accuracies and summary_accuracies:
        accuracies = summary_accuracies
    expected = summary_expected or len(rollouts) or len(accuracies)
    effective_k = k or len(accuracies) or expected or 1
    pass1, avg1 = metric_at_k(accuracies, 1)
    passk, avgk = metric_at_k(accuracies, effective_k)
    failures = sorted({row.failure_type for row in rollouts if row.failure_type})
    return SampleRow(
        sample_id=sample_id,
        benchmark=benchmark,
        sample_dir=str(sample_dir),
        completed_rollouts=len(accuracies),
        expected_rollouts=expected,
        pass_at_1=pass1,
        average_at_1=avg1,
        pass_at_k=passk,
        average_at_k=avgk,
        rollout_accuracy_mean=mean(accuracies) if accuracies else None,
        best_accuracy=max(accuracies) if accuracies else None,
        failure_types=",".join(failures),
    )


def metric_at_k(accuracies: list[float], k: int) -> tuple[float | None, float | None]:
    if not accuracies:
        return None, None
    prefix = accuracies[: max(1, min(k, len(accuracies)))]
    return (1.0 if any(score >= 1.0 for score in prefix) else 0.0), mean(prefix)


def format_counter(counter: Counter[str], limit: int = 5) -> str:
    return ", ".join(
        f"{name}:{count}"
        for name, count in counter.most_common(limit)
        if name
    )


def parse_tool_names(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in str(text or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, raw_count = item.rsplit(":", 1)
        try:
            counter[name.strip()] += int(raw_count)
        except ValueError:
            continue
    return counter


def summarize_by_benchmark(samples: list[SampleRow], rollouts: list[RolloutRow], expected_counts: Counter[str]) -> list[dict[str, Any]]:
    sample_groups: dict[str, list[SampleRow]] = defaultdict(list)
    rollout_groups: dict[str, list[RolloutRow]] = defaultdict(list)
    for row in samples:
        sample_groups[row.benchmark].append(row)
    for row in rollouts:
        rollout_groups[row.benchmark].append(row)

    rows: list[dict[str, Any]] = []
    for benchmark in sorted(set(sample_groups) | set(expected_counts) | set(rollout_groups)):
        sg = sample_groups.get(benchmark, [])
        rg = rollout_groups.get(benchmark, [])
        completed_samples = [row for row in sg if row.completed_rollouts > 0]
        pass1_values = [row.pass_at_1 for row in completed_samples if row.pass_at_1 is not None]
        avg1_values = [row.average_at_1 for row in completed_samples if row.average_at_1 is not None]
        passk_values = [row.pass_at_k for row in completed_samples if row.pass_at_k is not None]
        avgk_values = [row.average_at_k for row in completed_samples if row.average_at_k is not None]
        rollout_scores = [row.accuracy_score for row in rg if row.accuracy_score is not None]
        expected = expected_counts.get(benchmark, 0)
        tool_counter: Counter[str] = Counter()
        for row in rg:
            tool_counter.update(parse_tool_names(row.tool_names))
        failed_rollouts = [row for row in rg if not row.success]
        failure_counter = Counter(row.failure_type or "unknown" for row in failed_rollouts)
        successful_samples = sum(1 for row in completed_samples if (row.best_accuracy or 0) >= 1.0)
        completed_rollouts = sum(1 for row in rg if row.completed)
        successful_rollouts = sum(1 for row in rg if row.success)
        rows.append(
            {
                "benchmark": benchmark,
                "expected_samples": expected or "",
                "output_samples": len(sg),
                "completed_samples": len(completed_samples),
                "missing_samples": max(expected - len(sg), 0) if expected else "",
                "sample_success_rate": (successful_samples / len(completed_samples)) if completed_samples else None,
                "rollout_success_rate": (successful_rollouts / completed_rollouts) if completed_rollouts else None,
                "pass@1": mean(pass1_values) if pass1_values else None,
                "average@1": mean(avg1_values) if avg1_values else None,
                "pass@k": mean(passk_values) if passk_values else None,
                "average@k": mean(avgk_values) if avgk_values else None,
                "rollout_accuracy_mean": mean(rollout_scores) if rollout_scores else None,
                "successful_samples": successful_samples,
                "total_rollouts": len(rg),
                "completed_rollouts": completed_rollouts,
                "successful_rollouts": successful_rollouts,
                "failed_rollouts": len(failed_rollouts),
                "assistant_turns_mean": mean([row.assistant_turns for row in rg]) if rg else None,
                "tool_calls_total": sum(row.tool_calls for row in rg),
                "tool_calls_mean": mean([row.tool_calls for row in rg]) if rg else None,
                "tool_call_rate": (sum(1 for row in rg if row.tool_calls > 0) / len(rg)) if rg else None,
                "top_tools": format_counter(tool_counter),
                "top_failure_types": format_counter(failure_counter),
            }
        )
    return rows


def summarize_failures(rollouts: list[RolloutRow]) -> list[dict[str, Any]]:
    groups: dict[str, list[RolloutRow]] = defaultdict(list)
    for row in rollouts:
        if row.success:
            continue
        groups[row.failure_type or "unknown"].append(row)
    total = len([row for row in rollouts if not row.success]) or 1
    rows = []
    for label, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        rows.append(
            {
                "failure_type": label,
                "count": len(group),
                "ratio_of_failed_rollouts": len(group) / total,
                "examples": ", ".join(row.sample_id for row in group[:5]),
            }
        )
    return rows


def summarize_failures_by_benchmark(rollouts: list[RolloutRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RolloutRow]] = defaultdict(list)
    failed_by_benchmark: Counter[str] = Counter()
    for row in rollouts:
        if row.success:
            continue
        label = row.failure_type or "unknown"
        groups[(row.benchmark, label)].append(row)
        failed_by_benchmark[row.benchmark] += 1

    rows = []
    for (benchmark, label), group in sorted(groups.items(), key=lambda item: (item[0][0], -len(item[1]), item[0][1])):
        denom = failed_by_benchmark[benchmark] or 1
        rows.append(
            {
                "benchmark": benchmark,
                "failure_type": label,
                "count": len(group),
                "ratio_of_benchmark_failed_rollouts": len(group) / denom,
                "examples": ", ".join(row.sample_id for row in group[:5]),
            }
        )
    return rows


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return ordered[idx]


def summarize_api_timings(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path or not path.exists():
        return [], []
    rows = list(iter_jsonl(path))
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row.get("kind") or row.get("request_kind") or "unknown")].append(row)
        by_endpoint[str(row.get("endpoint") or "unknown")].append(row)

    def summarize_group(name_key: str, name: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = [value for value in (as_float(row.get("latency_sec") or row.get("latency")) for row in group) if value is not None]
        failures = [
            row for row in group
            if row.get("error_type") or (as_float(row.get("status_code")) or 200) >= 400
        ]
        prompt_tokens = [value for value in (as_float(row.get("prompt_tokens")) for row in group) if value is not None]
        completion_tokens = [value for value in (as_float(row.get("completion_tokens")) for row in group) if value is not None]
        return {
            name_key: name,
            "requests": len(group),
            "failures": len(failures),
            "latency_mean_sec": mean(latencies) if latencies else None,
            "latency_p50_sec": percentile(latencies, 0.50),
            "latency_p95_sec": percentile(latencies, 0.95),
            "prompt_tokens_mean": mean(prompt_tokens) if prompt_tokens else None,
            "completion_tokens_mean": mean(completion_tokens) if completion_tokens else None,
        }

    kind_rows = [summarize_group("kind", name, group) for name, group in sorted(by_kind.items())]
    endpoint_rows = [summarize_group("endpoint", name, group) for name, group in sorted(by_endpoint.items())]
    return kind_rows, endpoint_rows


def count_experience_items(path: Path | None) -> int | None:
    if not path or not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("experiences", "items", "ops"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                return len(value)
        # Backward compatibility: some libraries are saved directly as
        # {"E0": "...", "E1": "..."} without a wrapper key.
        return len(payload)
    return None


def count_skill_words(path: Path | None) -> int | None:
    if not path or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"\S+", text))


def count_skill_bank(path: Path | None) -> int | None:
    if not path or not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        # SkillRL bank format:
        # {
        #   "general_skills": [...],
        #   "task_specific_skills": {"task": [...]},
        #   "common_mistakes": [...]
        # }
        # Count actual entries, not the number of top-level sections.
        total = 0
        recognized_skillrl_bank = False

        general = payload.get("general_skills")
        if isinstance(general, list):
            total += len(general)
            recognized_skillrl_bank = True

        task_specific = payload.get("task_specific_skills")
        if isinstance(task_specific, dict):
            total += sum(len(skills) for skills in task_specific.values() if isinstance(skills, list))
            recognized_skillrl_bank = True
        elif isinstance(task_specific, list):
            total += len(task_specific)
            recognized_skillrl_bank = True

        common_mistakes = payload.get("common_mistakes")
        if isinstance(common_mistakes, list):
            total += len(common_mistakes)
            recognized_skillrl_bank = True

        if recognized_skillrl_bank:
            return total

        for key in ("skills", "items"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                return len(value)
        return len(payload)
    return None


def memory_summary(memory_dir: Path | None) -> dict[str, Any]:
    if not memory_dir:
        return {}
    exp_path = memory_dir / "experiences.json"
    skill_path = memory_dir / "SKILL.md"
    skill_bank_path = memory_dir / "skillrl_skill_bank.json"
    return {
        "memory_dir": str(memory_dir),
        "experiences_json": str(exp_path) if exp_path.exists() else "",
        "experience_count": count_experience_items(exp_path),
        "skill_md": str(skill_path) if skill_path.exists() else "",
        "skill_word_count": count_skill_words(skill_path),
        "skillrl_skill_bank_json": str(skill_bank_path) if skill_bank_path.exists() else "",
        "skillrl_skill_count": count_skill_bank(skill_bank_path),
    }


def overall_summary(samples: list[SampleRow], rollouts: list[RolloutRow], expected_counts: Counter[str]) -> dict[str, Any]:
    completed_samples = [row for row in samples if row.completed_rollouts > 0]
    pass1_values = [row.pass_at_1 for row in completed_samples if row.pass_at_1 is not None]
    avg1_values = [row.average_at_1 for row in completed_samples if row.average_at_1 is not None]
    passk_values = [row.pass_at_k for row in completed_samples if row.pass_at_k is not None]
    avgk_values = [row.average_at_k for row in completed_samples if row.average_at_k is not None]
    rollout_scores = [row.accuracy_score for row in rollouts if row.accuracy_score is not None]
    expected_total = sum(expected_counts.values()) if expected_counts else None
    return {
        "expected_samples": expected_total,
        "output_samples": len(samples),
        "completed_samples": len(completed_samples),
        "missing_samples": max(expected_total - len(samples), 0) if expected_total is not None else None,
        "total_rollouts": len(rollouts),
        "completed_rollouts": sum(1 for row in rollouts if row.completed),
        "successful_rollouts": sum(1 for row in rollouts if row.success),
        "successful_samples": sum(1 for row in completed_samples if (row.best_accuracy or 0) >= 1.0),
        "pass@1": mean(pass1_values) if pass1_values else None,
        "average@1": mean(avg1_values) if avg1_values else None,
        "pass@k": mean(passk_values) if passk_values else None,
        "average@k": mean(avgk_values) if avgk_values else None,
        "rollout_accuracy_mean": mean(rollout_scores) if rollout_scores else None,
        "assistant_turns_mean": mean([row.assistant_turns for row in rollouts]) if rollouts else None,
        "tool_calls_total": sum(row.tool_calls for row in rollouts),
    }


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


def make_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    lines = [
        "# XSkill Accumulation Summary",
        "",
        "## Run",
        "",
        f"- output_dir: `{payload['output_dir']}`",
        f"- data_files: `{', '.join(payload['data_files']) if payload['data_files'] else ''}`",
        f"- k: `{payload['k'] or 'all completed rollouts per sample'}`",
        "",
        "## Overall",
        "",
        markdown_table([overall], [
            "expected_samples",
            "output_samples",
            "completed_samples",
            "missing_samples",
            "pass@1",
            "average@1",
            "pass@k",
            "average@k",
            "rollout_accuracy_mean",
            "successful_samples",
            "total_rollouts",
            "successful_rollouts",
            "tool_calls_total",
        ]),
        "",
        "## By Benchmark",
        "",
        markdown_table(payload["by_benchmark"], [
            "benchmark",
            "expected_samples",
            "output_samples",
            "completed_samples",
            "missing_samples",
            "sample_success_rate",
            "rollout_success_rate",
            "pass@1",
            "average@1",
            "pass@k",
            "average@k",
            "rollout_accuracy_mean",
            "successful_samples",
            "total_rollouts",
            "successful_rollouts",
            "failed_rollouts",
            "assistant_turns_mean",
            "tool_calls_total",
            "tool_calls_mean",
            "tool_call_rate",
            "top_tools",
            "top_failure_types",
        ]),
        "",
        "## Failure Types",
        "",
        markdown_table(payload["failure_types"], ["failure_type", "count", "ratio_of_failed_rollouts", "examples"]),
        "",
        "## Failure Types By Benchmark",
        "",
        markdown_table(payload["failure_types_by_benchmark"], [
            "benchmark",
            "failure_type",
            "count",
            "ratio_of_benchmark_failed_rollouts",
            "examples",
        ]),
        "",
        "## Memory",
        "",
        markdown_table([payload["memory"]], [
            "memory_dir",
            "experience_count",
            "skill_word_count",
            "skillrl_skill_count",
            "experiences_json",
            "skill_md",
            "skillrl_skill_bank_json",
        ]),
    ]
    if payload["api_timing_by_kind"]:
        lines.extend([
            "",
            "## API Timing By Kind",
            "",
            markdown_table(payload["api_timing_by_kind"], [
                "kind",
                "requests",
                "failures",
                "latency_mean_sec",
                "latency_p50_sec",
                "latency_p95_sec",
                "prompt_tokens_mean",
                "completion_tokens_mean",
            ]),
        ])
    if payload["api_timing_by_endpoint"]:
        lines.extend([
            "",
            "## API Timing By Endpoint",
            "",
            markdown_table(payload["api_timing_by_endpoint"], [
                "endpoint",
                "requests",
                "failures",
                "latency_mean_sec",
                "latency_p50_sec",
                "latency_p95_sec",
            ]),
        ])
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "output" / "accumulation_reports" / run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "xskill_accum" / DEFAULT_RUN_NAME),
        help="Accumulation output directory.",
    )
    parser.add_argument(
        "--data-file",
        action="append",
        default=[],
        help="Original data file for benchmark mapping and expected counts. Supports json/jsonl/parquet. Can be repeated.",
    )
    parser.add_argument("--memory-dir", default=None, help="Memory bank directory, e.g. memory_bank/test.")
    parser.add_argument("--api-timings", default=None, help="api_timings.jsonl path. Defaults to output-dir/api_timings.jsonl.")
    parser.add_argument("--k", type=int, default=None, help="Use this k for pass@k/average@k. Default: all completed rollouts per sample.")
    parser.add_argument("--success-threshold", type=float, default=1.0, help="Accuracy score threshold for a successful rollout.")
    parser.add_argument("--output-md", default=None, help="Write Markdown summary to this path.")
    parser.add_argument("--output-json", default=None, help="Write full JSON summary to this path.")
    parser.add_argument("--benchmark-csv-output", default=None, help="Write benchmark summary CSV.")
    parser.add_argument("--benchmark-failure-csv-output", default=None, help="Write benchmark-by-failure summary CSV.")
    parser.add_argument("--sample-csv-output", default=None, help="Write sample summary CSV.")
    parser.add_argument("--rollout-csv-output", default=None, help="Write rollout detail CSV.")
    parser.add_argument(
        "--write-default-reports",
        action="store_true",
        help="Write Markdown/JSON/CSV files under output/accumulation_reports/<run_name>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    run_name = output_dir.name
    data_files = args.data_file or []
    if not data_files:
        default_data = PROJECT_ROOT / "benchmark" / "_mixed_protocol" / "train_core.json"
        if default_data.exists():
            data_files = [str(default_data)]
    api_timings = Path(args.api_timings).expanduser() if args.api_timings else output_dir / "api_timings.jsonl"
    memory_dir = Path(args.memory_dir).expanduser() if args.memory_dir else PROJECT_ROOT / "memory_bank" / "xskill_accum" / run_name
    if not memory_dir.exists():
        fallback = PROJECT_ROOT / "memory_bank" / "test"
        memory_dir = fallback if fallback.exists() else memory_dir

    lookup, expected_counts = build_sample_lookup(data_files)
    sample_dirs = discover_sample_dirs(output_dir)
    rollout_rows: list[RolloutRow] = []
    sample_rows: list[SampleRow] = []
    for sample_dir in sample_dirs:
        rollouts = [
            analyze_rollout(sample_dir.name, sample_dir, rollout_dir, args.success_threshold, lookup)
            for rollout_dir in discover_rollout_dirs(sample_dir)
        ]
        rollout_rows.extend(rollouts)
        sample_rows.append(analyze_sample(sample_dir, rollouts, args.k))

    by_benchmark = summarize_by_benchmark(sample_rows, rollout_rows, expected_counts)
    failure_rows = summarize_failures(rollout_rows)
    failure_by_benchmark_rows = summarize_failures_by_benchmark(rollout_rows)
    api_kind_rows, api_endpoint_rows = summarize_api_timings(api_timings)
    payload = {
        "output_dir": str(output_dir),
        "data_files": data_files,
        "memory_dir": str(memory_dir),
        "api_timings": str(api_timings) if api_timings.exists() else "",
        "k": args.k,
        "success_threshold": args.success_threshold,
        "overall": overall_summary(sample_rows, rollout_rows, expected_counts),
        "by_benchmark": by_benchmark,
        "failure_types": failure_rows,
        "failure_types_by_benchmark": failure_by_benchmark_rows,
        "memory": memory_summary(memory_dir),
        "api_timing_by_kind": api_kind_rows,
        "api_timing_by_endpoint": api_endpoint_rows,
        "samples": [asdict(row) for row in sample_rows],
        "rollouts": [asdict(row) for row in rollout_rows],
    }

    markdown = make_markdown(payload)
    print(markdown)

    output_md = Path(args.output_md).expanduser() if args.output_md else None
    output_json = Path(args.output_json).expanduser() if args.output_json else None
    benchmark_csv = Path(args.benchmark_csv_output).expanduser() if args.benchmark_csv_output else None
    benchmark_failure_csv = Path(args.benchmark_failure_csv_output).expanduser() if args.benchmark_failure_csv_output else None
    sample_csv = Path(args.sample_csv_output).expanduser() if args.sample_csv_output else None
    rollout_csv = Path(args.rollout_csv_output).expanduser() if args.rollout_csv_output else None

    if args.write_default_reports:
        report_dir = default_output_dir(run_name)
        output_md = output_md or report_dir / "accumulation_summary.md"
        output_json = output_json or report_dir / "accumulation_summary.json"
        benchmark_csv = benchmark_csv or report_dir / "benchmark_summary.csv"
        benchmark_failure_csv = benchmark_failure_csv or report_dir / "benchmark_failure_summary.csv"
        sample_csv = sample_csv or report_dir / "sample_summary.csv"
        rollout_csv = rollout_csv or report_dir / "rollout_details.csv"

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"Wrote Markdown report: {output_md}")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {output_json}")
    if benchmark_csv:
        write_csv(benchmark_csv, by_benchmark)
        print(f"Wrote benchmark CSV: {benchmark_csv}")
    if benchmark_failure_csv:
        write_csv(benchmark_failure_csv, failure_by_benchmark_rows)
        print(f"Wrote benchmark failure CSV: {benchmark_failure_csv}")
    if sample_csv:
        write_csv(sample_csv, [asdict(row) for row in sample_rows])
        print(f"Wrote sample CSV: {sample_csv}")
    if rollout_csv:
        write_csv(rollout_csv, [asdict(row) for row in rollout_rows])
        print(f"Wrote rollout CSV: {rollout_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
