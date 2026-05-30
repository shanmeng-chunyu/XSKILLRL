#!/usr/bin/env python3
"""Report per-benchmark accuracy from an XSkill output directory.

This script is read-only. It scans a run output directory with either
multi-rollout sample folders:

    output/xskill_accum/<run>/<sample_id>/metrics_sample.json
    output/xskill_accum/<run>/<sample_id>/rollout_0/metrics.json

or single-rollout sample folders:

    output/xskill_accum/<run>/<sample_id>/metrics.json

It prints overall and per-benchmark pass@k / average@k style metrics. If an
original benchmark data file is provided, sample IDs are mapped to their
``benchmark_name`` from that file; otherwise the script falls back to benchmark
prefixes in output directory names.
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


SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
SKIP_DIR_NAMES = {
    "__pycache__",
    "api_timings",
    "eval_dumps",
    "eval_runs",
    "logs",
    "memory_bank",
    "snapshots",
}
KNOWN_BENCHMARK_PREFIXES = (
    "visualtoolbench",
    "tirbench",
    "mmsearch_plus",
    "agentvista",
    "mmbrowsecomp",
)


@dataclass
class SampleAccuracy:
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
    accuracies: list[float]
    error: str


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


def normalize_benchmark(value: Any) -> str:
    text = normalized_id(value).lower()
    return text.replace("-", "_")


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
    for data_file in data_files:
        path = Path(data_file).expanduser()
        if not path.exists():
            print(f"[WARN] data file not found, benchmark mapping may be incomplete: {path}")
            continue
        for idx, record in enumerate(load_records(path)):
            sample = sample_from_record(record)
            benchmark = benchmark_from_sample(sample)
            counts[benchmark] += 1
            for sample_id in output_sample_ids(sample, idx):
                lookup.setdefault(sample_id, sample)
    return lookup, counts


def infer_benchmark(sample_id: str, sample_dir: Path, sample_lookup: dict[str, dict[str, Any]]) -> str:
    sample = sample_lookup.get(sample_id) or sample_lookup.get(sample_dir.name)
    benchmark = benchmark_from_sample(sample)
    if benchmark != "unknown":
        return benchmark

    # Some outputs include metadata in sample-level files.
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
    for prefix in KNOWN_BENCHMARK_PREFIXES:
        if lowered == prefix or lowered.startswith(f"{prefix}_"):
            return prefix
    return "unknown"


def discover_sample_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    sample_dirs: list[Path] = []
    for path in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIR_NAMES:
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


def load_rollout_accuracies(sample_dir: Path) -> tuple[list[float], int, str]:
    rollout_dirs = sorted(
        [path for path in sample_dir.iterdir() if path.is_dir() and path.name.startswith("rollout_")],
        key=rollout_sort_key,
    )
    if not rollout_dirs and (sample_dir / "metrics.json").exists():
        rollout_dirs = [sample_dir]

    accuracies: list[float] = []
    errors: list[str] = []
    for rollout_dir in rollout_dirs:
        metrics_path = rollout_dir / "metrics.json"
        if not metrics_path.exists():
            errors.append(f"{rollout_dir.name}: missing metrics.json")
            continue
        try:
            payload = read_json(metrics_path)
        except Exception as exc:
            errors.append(f"{rollout_dir.name}: failed to read metrics.json: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{rollout_dir.name}: metrics.json is not an object")
            continue
        score = as_float(payload.get("accuracy_score"))
        if score is None:
            errors.append(f"{rollout_dir.name}: missing accuracy_score")
            continue
        accuracies.append(score)
    return accuracies, len(rollout_dirs), "; ".join(errors)


def metric_at_k(accuracies: list[float], k: int) -> tuple[float | None, float | None]:
    if not accuracies:
        return None, None
    prefix = accuracies[: max(1, min(k, len(accuracies)))]
    pass_at_k = 1.0 if any(score >= 1.0 for score in prefix) else 0.0
    average_at_k = mean(prefix)
    return pass_at_k, average_at_k


def analyze_sample(
    sample_dir: Path,
    *,
    k: int | None,
    sample_lookup: dict[str, dict[str, Any]],
) -> SampleAccuracy:
    sample_id = sample_dir.name
    benchmark = infer_benchmark(sample_id, sample_dir, sample_lookup)
    summary_path = sample_dir / SAMPLE_SUMMARY_FILENAME
    accuracies: list[float] = []
    expected_rollouts = 0
    error = ""

    if summary_path.exists():
        try:
            summary = read_json(summary_path)
        except Exception as exc:
            summary = {}
            error = f"failed to read {SAMPLE_SUMMARY_FILENAME}: {exc}"
        if isinstance(summary, dict):
            raw_accuracies = summary.get("accuracies")
            if isinstance(raw_accuracies, list):
                accuracies = [score for score in (as_float(value) for value in raw_accuracies) if score is not None]
            expected_rollouts = int(as_float(summary.get("num_rollouts")) or len(accuracies))

    if not accuracies:
        accuracies, expected_rollouts, rollout_error = load_rollout_accuracies(sample_dir)
        error = "; ".join(part for part in (error, rollout_error) if part)

    completed = len(accuracies)
    effective_k = k or completed or expected_rollouts or 1
    pass_at_1, average_at_1 = metric_at_k(accuracies, 1)
    pass_at_k, average_at_k = metric_at_k(accuracies, effective_k)
    return SampleAccuracy(
        sample_id=sample_id,
        benchmark=benchmark,
        sample_dir=str(sample_dir),
        completed_rollouts=completed,
        expected_rollouts=expected_rollouts,
        pass_at_1=pass_at_1,
        average_at_1=average_at_1,
        pass_at_k=pass_at_k,
        average_at_k=average_at_k,
        rollout_accuracy_mean=mean(accuracies) if accuracies else None,
        best_accuracy=max(accuracies) if accuracies else None,
        accuracies=accuracies,
        error=error,
    )


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_rows(samples: list[SampleAccuracy], data_counts: Counter[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[SampleAccuracy]] = defaultdict(list)
    for sample in samples:
        groups[sample.benchmark or "unknown"].append(sample)

    rows: list[dict[str, Any]] = []
    for benchmark in sorted(set(groups) | set(data_counts)):
        group = groups.get(benchmark, [])
        completed = [sample for sample in group if sample.completed_rollouts > 0]
        pass1_values = [sample.pass_at_1 for sample in completed if sample.pass_at_1 is not None]
        avg1_values = [sample.average_at_1 for sample in completed if sample.average_at_1 is not None]
        passk_values = [sample.pass_at_k for sample in completed if sample.pass_at_k is not None]
        avgk_values = [sample.average_at_k for sample in completed if sample.average_at_k is not None]
        rollout_scores = [
            score
            for sample in completed
            for score in sample.accuracies
        ]
        expected = data_counts.get(benchmark)
        rows.append(
            {
                "benchmark": benchmark,
                "expected_samples": expected if expected else "",
                "output_samples": len(group),
                "completed_samples": len(completed),
                "missing_samples": max(expected - len(group), 0) if expected else "",
                "pass@1": mean(pass1_values) if pass1_values else None,
                "average@1": mean(avg1_values) if avg1_values else None,
                "pass@k": mean(passk_values) if passk_values else None,
                "average@k": mean(avgk_values) if avgk_values else None,
                "rollout_accuracy_mean": mean(rollout_scores) if rollout_scores else None,
                "successful_samples": sum(1 for sample in completed if (sample.best_accuracy or 0.0) >= 1.0),
                "total_rollouts": sum(sample.completed_rollouts for sample in completed),
                "successful_rollouts": sum(1 for sample in completed for score in sample.accuracies if score >= 1.0),
                "errored_samples": sum(1 for sample in group if sample.error),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "benchmark",
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
        "errored_samples",
    ]
    widths = {
        header: max(len(header), *(len(fmt(row.get(header))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(fmt(row.get(header)).ljust(widths[header]) for header in headers))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="XSkill output run directory to scan.")
    parser.add_argument(
        "--data-file",
        action="append",
        default=[],
        help="Optional original data file for benchmark mapping and expected counts. Supports json/jsonl/parquet. Can be repeated.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Report pass@k/average@k using this k. Default: all completed rollouts per sample.",
    )
    parser.add_argument("--json-output", help="Optional path to write full JSON report.")
    parser.add_argument("--benchmark-csv-output", help="Optional path to write per-benchmark CSV report.")
    parser.add_argument("--sample-csv-output", help="Optional path to write per-sample CSV report.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    sample_lookup, data_counts = build_sample_lookup(args.data_file)
    sample_dirs = discover_sample_dirs(output_dir)
    samples = [
        analyze_sample(sample_dir, k=args.k, sample_lookup=sample_lookup)
        for sample_dir in sample_dirs
    ]
    rows = summarize_rows(samples, data_counts)

    print("=== XSkill Output Benchmark Accuracy ===")
    print(f"output_dir: {output_dir}")
    print(f"sample_dirs: {len(sample_dirs)}")
    print(f"k: {args.k if args.k is not None else 'all completed rollouts per sample'}")
    print()
    print_table(rows)

    if args.json_output:
        path = Path(args.json_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "output_dir": str(output_dir),
            "k": args.k,
            "by_benchmark": rows,
            "samples": [asdict(sample) for sample in samples],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {path}")

    if args.benchmark_csv_output:
        path = Path(args.benchmark_csv_output).expanduser()
        write_csv(path, rows)
        print(f"Wrote benchmark CSV report: {path}")

    if args.sample_csv_output:
        path = Path(args.sample_csv_output).expanduser()
        sample_rows = [asdict(sample) for sample in samples]
        for row in sample_rows:
            row["accuracies"] = json.dumps(row["accuracies"], ensure_ascii=False)
        write_csv(path, sample_rows)
        print(f"Wrote sample CSV report: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
