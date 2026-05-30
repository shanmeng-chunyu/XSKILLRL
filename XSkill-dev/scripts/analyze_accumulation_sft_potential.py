"""Estimate successful rollout count and trajectory-SFT potential.

This script is read-only.  It scans an XSkill accumulation output directory
whose layout is usually:

    output/xskill_accum/<run_name>/<sample_id>/rollout_0/metrics.json
    output/xskill_accum/<run_name>/<sample_id>/rollout_0/traj.jsonl

For single-rollout runs it also accepts:

    output/xskill_accum/<run_name>/<sample_id>/metrics.json
    output/xskill_accum/<run_name>/<sample_id>/traj.jsonl

The estimated SFT sample count follows the SkillRL-style step imitation idea:
each assistant action turn in a successful rollout can become one SFT sample.
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


DEFAULT_ACCUM_DIR = "/sata/luzy/XSKILLRL/XSkill-dev/output/xskill_accum/qwen3vl8b_mixed_train_core_seed42"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "benchmark" / "_mixed_protocol" / "train_core.json"


@dataclass
class RolloutStats:
    sample_id: str
    benchmark: str
    rollout_id: str
    rollout_dir: str
    completed: bool
    success: bool
    accuracy_score: float | None
    sft_samples: int
    assistant_turns: int
    tool_calls: int
    final_answer: str
    ground_truth: str
    error: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def infer_benchmark_from_sample(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "unknown"
    for key in ("benchmark_name", "benchmark", "source", "data_source"):
        value = sample.get(key)
        if value:
            return str(value)
    return "unknown"


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = read_json_any(path)
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


def build_sample_lookup(data_files: list[str]) -> tuple[dict[str, dict[str, Any]], Counter]:
    lookup: dict[str, dict[str, Any]] = {}
    benchmark_counts: Counter = Counter()
    for data_file in data_files:
        if not data_file:
            continue
        path = Path(data_file).expanduser()
        if not path.exists():
            print(f"Warning: data file not found, benchmark grouping may be incomplete: {path}")
            continue
        for idx, record in enumerate(load_records(path)):
            sample = sample_from_record(record)
            benchmark = infer_benchmark_from_sample(sample)
            benchmark_counts[benchmark] += 1
            for sid in output_sample_ids(sample, idx):
                lookup.setdefault(sid, sample)
    return lookup, benchmark_counts


def load_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    for filename in (
        "sample.json",
        "metadata.json",
        "input_sample.json",
        "request.json",
        "metrics_sample.json",
    ):
        path = sample_dir / filename
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def infer_benchmark(
    sample_id: str,
    sample_dir: Path,
    metrics: dict[str, Any] | None,
    sample_lookup: dict[str, dict[str, Any]],
) -> str:
    for source in (metrics or {}, load_sample_metadata(sample_dir)):
        benchmark = infer_benchmark_from_sample(source)
        if benchmark != "unknown":
            return benchmark

    sample = sample_lookup.get(sample_id) or sample_lookup.get(sample_dir.name)
    if sample:
        return infer_benchmark_from_sample(sample)

    lowered = sample_id.lower()
    known_prefixes = (
        "agentvista",
        "mmbrowsecomp",
        "mmsearch_plus",
        "tirbench",
        "visualtoolbench",
    )
    for prefix in known_prefixes:
        if lowered.startswith(prefix):
            return prefix
    return "unknown"


def discover_sample_dirs(accum_dir: Path) -> list[Path]:
    if not accum_dir.exists():
        raise FileNotFoundError(f"Accumulation directory not found: {accum_dir}")
    sample_dirs = []
    for path in sorted(accum_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if path.name in {"logs", "memory_bank", "api_timings", "_remote_images"}:
            continue
        sample_dirs.append(path)
    return sample_dirs


def discover_rollout_dirs(sample_dir: Path) -> list[Path]:
    rollout_dirs = sorted(
        [path for path in sample_dir.iterdir() if path.is_dir() and path.name.startswith("rollout_")],
        key=lambda p: rollout_sort_key(p.name),
    )
    if rollout_dirs:
        return rollout_dirs
    if (sample_dir / "metrics.json").exists() or (sample_dir / "traj.jsonl").exists():
        return [sample_dir]
    return []


def rollout_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    try:
        return (int(suffix), name)
    except ValueError:
        return (10**9, name)


def count_trajectory_steps(traj_path: Path) -> tuple[int, int]:
    if not traj_path.exists():
        return 0, 0
    assistant_turns = 0
    tool_calls = 0
    for event in iter_jsonl(traj_path):
        if "text_output" in event and event.get("turn_idx") is not None:
            text = str(event.get("text_output") or "").strip()
            if text:
                assistant_turns += 1
        if "tool_call" in event:
            tool_calls += 1
    return assistant_turns, tool_calls


def analyze_rollout(
    sample_id: str,
    sample_dir: Path,
    rollout_dir: Path,
    threshold: float,
    sample_lookup: dict[str, dict[str, Any]],
) -> RolloutStats:
    metrics_path = rollout_dir / "metrics.json"
    traj_path = rollout_dir / "traj.jsonl"
    if not metrics_path.exists():
        assistant_turns, tool_calls = count_trajectory_steps(traj_path)
        benchmark = infer_benchmark(sample_id, sample_dir, None, sample_lookup)
        return RolloutStats(
            sample_id=sample_id,
            benchmark=benchmark,
            rollout_id=rollout_dir.name,
            rollout_dir=str(rollout_dir),
            completed=False,
            success=False,
            accuracy_score=None,
            sft_samples=0,
            assistant_turns=assistant_turns,
            tool_calls=tool_calls,
            final_answer="",
            ground_truth="",
            error="missing metrics.json",
        )

    try:
        metrics = read_json(metrics_path)
    except Exception as exc:
        assistant_turns, tool_calls = count_trajectory_steps(traj_path)
        benchmark = infer_benchmark(sample_id, sample_dir, None, sample_lookup)
        return RolloutStats(
            sample_id=sample_id,
            benchmark=benchmark,
            rollout_id=rollout_dir.name,
            rollout_dir=str(rollout_dir),
            completed=False,
            success=False,
            accuracy_score=None,
            sft_samples=0,
            assistant_turns=assistant_turns,
            tool_calls=tool_calls,
            final_answer="",
            ground_truth="",
            error=f"failed to read metrics.json: {exc}",
        )

    accuracy = as_float(metrics.get("accuracy_score"))
    benchmark = infer_benchmark(sample_id, sample_dir, metrics, sample_lookup)
    success = accuracy is not None and accuracy >= threshold
    assistant_turns, tool_calls = count_trajectory_steps(traj_path)
    sft_samples = assistant_turns if success else 0
    if success and sft_samples == 0:
        # Fallback for older runs where traj.jsonl is incomplete but metrics says
        # the rollout succeeded.  Such a rollout can still form at least one
        # final-answer SFT pair if final_answer exists.
        sft_samples = 1 if metrics.get("final_answer") else 0
    return RolloutStats(
        sample_id=sample_id,
        benchmark=benchmark,
        rollout_id=rollout_dir.name,
        rollout_dir=str(rollout_dir),
        completed=True,
        success=success,
        accuracy_score=accuracy,
        sft_samples=sft_samples,
        assistant_turns=assistant_turns,
        tool_calls=tool_calls,
        final_answer=str(metrics.get("final_answer") or ""),
        ground_truth=str(metrics.get("ground_truth") or ""),
        error="",
    )


def summarize(sample_dirs: list[Path], rollouts: list[RolloutStats]) -> dict[str, Any]:
    by_sample: dict[str, list[RolloutStats]] = {}
    for rollout in rollouts:
        by_sample.setdefault(rollout.sample_id, []).append(rollout)

    completed_rollouts = [r for r in rollouts if r.completed]
    successful_rollouts = [r for r in rollouts if r.success]
    samples_with_completed = [sid for sid, group in by_sample.items() if any(r.completed for r in group)]
    samples_with_success = [sid for sid, group in by_sample.items() if any(r.success for r in group)]
    completed_accuracies = [r.accuracy_score for r in completed_rollouts if r.accuracy_score is not None]
    success_sft_counts = [r.sft_samples for r in successful_rollouts]
    success_turn_counts = [r.assistant_turns for r in successful_rollouts]

    max_rollouts_per_sample = max((len(group) for group in by_sample.values()), default=0)
    pass_at_k_key = f"pass@{max_rollouts_per_sample}" if max_rollouts_per_sample else "pass@k"

    return {
        "total_sample_dirs": len(sample_dirs),
        "samples_with_rollouts": len(by_sample),
        "samples_with_completed_rollouts": len(samples_with_completed),
        "samples_with_success": len(samples_with_success),
        "total_rollout_dirs": len(rollouts),
        "completed_rollouts": len(completed_rollouts),
        "successful_rollouts": len(successful_rollouts),
        "estimated_sft_samples_from_successful_rollouts": sum(r.sft_samples for r in successful_rollouts),
        "estimated_sft_samples_per_successful_rollout_mean": round(mean(success_sft_counts), 4) if success_sft_counts else 0.0,
        "assistant_turns_per_successful_rollout_mean": round(mean(success_turn_counts), 4) if success_turn_counts else 0.0,
        "tool_calls_in_successful_rollouts": sum(r.tool_calls for r in successful_rollouts),
        "overall_rollout_accuracy_mean": round(mean(completed_accuracies), 6) if completed_accuracies else 0.0,
        pass_at_k_key: round(len(samples_with_success) / len(by_sample), 6) if by_sample else 0.0,
        "rollouts_per_sample_max": max_rollouts_per_sample,
    }


def summarize_by_benchmark(rollouts: list[RolloutStats], benchmark_counts: Counter) -> list[dict[str, Any]]:
    groups: dict[str, list[RolloutStats]] = defaultdict(list)
    for rollout in rollouts:
        groups[rollout.benchmark or "unknown"].append(rollout)

    rows = []
    for benchmark in sorted(set(groups) | set(benchmark_counts)):
        group = groups.get(benchmark, [])
        completed = [r for r in group if r.completed]
        successful = [r for r in group if r.success]
        sample_ids = {r.sample_id for r in group}
        completed_sample_ids = {r.sample_id for r in completed}
        success_sample_ids = {r.sample_id for r in successful}
        accuracies = [r.accuracy_score for r in completed if r.accuracy_score is not None]
        max_rollouts_per_sample = max(
            (sum(1 for r in group if r.sample_id == sid) for sid in sample_ids),
            default=0,
        )
        input_samples = benchmark_counts.get(benchmark, 0)
        completed_rollout_count = len(completed)
        successful_rollout_count = len(successful)
        rows.append(
            {
                "benchmark": benchmark,
                "input_samples": input_samples or "",
                "samples": len(sample_ids),
                "missing_samples": max(input_samples - len(sample_ids), 0) if input_samples else "",
                "completed_samples": len(completed_sample_ids),
                "success_samples": len(success_sample_ids),
                "sample_completion_rate": round(len(completed_sample_ids) / input_samples, 6) if input_samples else "",
                "sample_success_rate": round(len(success_sample_ids) / len(sample_ids), 6) if sample_ids else 0.0,
                "rollouts": len(group),
                "completed_rollouts": len(completed),
                "successful_rollouts": len(successful),
                "rollout_completion_rate": round(completed_rollout_count / len(group), 6) if group else 0.0,
                "rollout_success_rate": round(successful_rollout_count / completed_rollout_count, 6) if completed_rollout_count else 0.0,
                "estimated_sft_samples": sum(r.sft_samples for r in successful),
                "assistant_turns_success_mean": round(mean([r.assistant_turns for r in successful]), 4) if successful else 0.0,
                "sft_samples_per_success_mean": round(mean([r.sft_samples for r in successful]), 4) if successful else 0.0,
                "tool_calls_total": sum(r.tool_calls for r in group),
                "tool_calls_in_successful_rollouts": sum(r.tool_calls for r in successful),
                "rollout_accuracy_mean": round(mean(accuracies), 6) if accuracies else 0.0,
                "pass@k": round(len(success_sample_ids) / len(sample_ids), 6) if sample_ids else 0.0,
                "rollouts_per_sample_max": max_rollouts_per_sample,
            }
        )
    return rows


def write_csv(path: Path, rollouts: list[RolloutStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rollouts[0]).keys()) if rollouts else list(RolloutStats.__dataclass_fields__.keys()))
        writer.writeheader()
        for rollout in rollouts:
            writer.writerow(asdict(rollout))


def print_benchmark_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n=== By Benchmark ===")
    headers = [
        "benchmark",
        "input_samples",
        "samples",
        "missing_samples",
        "success_samples",
        "sample_success_rate",
        "successful_rollouts",
        "rollout_success_rate",
        "estimated_sft_samples",
        "rollout_accuracy_mean",
        "pass@k",
        "tool_calls_total",
        "tool_calls_in_successful_rollouts",
    ]
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def print_summary(
    summary: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    success_examples: list[RolloutStats],
    failure_examples: list[RolloutStats],
) -> None:
    print("\n=== XSkill Accumulation SFT Potential ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print_benchmark_summary(benchmark_rows)

    if success_examples:
        print("\nSuccessful rollout examples:")
        for rollout in success_examples:
            print(
                f"  - {rollout.benchmark}/{rollout.sample_id}/{rollout.rollout_id}: "
                f"acc={rollout.accuracy_score}, turns={rollout.assistant_turns}, "
                f"sft_samples={rollout.sft_samples}"
            )

    if failure_examples:
        print("\nIncomplete/error rollout examples:")
        for rollout in failure_examples:
            print(f"  - {rollout.sample_id}/{rollout.rollout_id}: {rollout.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accum-dir",
        default=DEFAULT_ACCUM_DIR,
        help="XSkill accumulation run directory to scan.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Minimum accuracy_score treated as a successful trajectory.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write summary and rollout details as JSON.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write per-rollout details as CSV.",
    )
    parser.add_argument(
        "--data-file",
        action="append",
        default=[],
        help=(
            "Optional original benchmark/GRPO data file used to map sample IDs to benchmark names. "
            "Can be repeated. Supports json, jsonl, parquet."
        ),
    )
    parser.add_argument(
        "--benchmark-csv-output",
        default=None,
        help="Optional path to write per-benchmark summary as CSV.",
    )
    parser.add_argument("--show-examples", type=int, default=10)
    args = parser.parse_args()

    accum_dir = Path(args.accum_dir).expanduser()
    data_files = list(args.data_file or [])
    if not data_files and DEFAULT_DATA_FILE.exists():
        data_files = [str(DEFAULT_DATA_FILE)]
    sample_lookup, benchmark_counts = build_sample_lookup(data_files)
    sample_dirs = discover_sample_dirs(accum_dir)
    rollouts: list[RolloutStats] = []
    for sample_dir in sample_dirs:
        rollout_dirs = discover_rollout_dirs(sample_dir)
        for rollout_dir in rollout_dirs:
            rollouts.append(
                analyze_rollout(
                    sample_dir.name,
                    sample_dir,
                    rollout_dir,
                    args.success_threshold,
                    sample_lookup,
                )
            )

    summary = summarize(sample_dirs, rollouts)
    benchmark_rows = summarize_by_benchmark(rollouts, benchmark_counts)
    if benchmark_counts:
        summary["input_data_benchmark_counts"] = dict(sorted(benchmark_counts.items()))
    successful = [r for r in rollouts if r.success]
    errored = [r for r in rollouts if r.error]
    print_summary(summary, benchmark_rows, successful[: args.show_examples], errored[: args.show_examples])

    if args.json_output:
        output_path = Path(args.json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "accum_dir": str(accum_dir),
            "success_threshold": args.success_threshold,
            "summary": summary,
            "by_benchmark": benchmark_rows,
            "rollouts": [asdict(rollout) for rollout in rollouts],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {output_path}")

    if args.csv_output:
        output_path = Path(args.csv_output).expanduser()
        write_csv(output_path, rollouts)
        print(f"Wrote CSV report: {output_path}")

    if args.benchmark_csv_output:
        output_path = Path(args.benchmark_csv_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(benchmark_rows[0].keys()) if benchmark_rows else [
                "benchmark",
                "samples",
                "success_samples",
                "successful_rollouts",
                "estimated_sft_samples",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in benchmark_rows:
                writer.writerow(row)
        print(f"Wrote benchmark CSV report: {output_path}")


if __name__ == "__main__":
    main()
