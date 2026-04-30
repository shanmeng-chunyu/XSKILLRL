"""Aggregate benchmark result summaries into a macro-average table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_METRIC_CANDIDATES = ("score", "accuracy", "overall_score")


def load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_metric(payload: Dict, metric_name: str | None) -> tuple[str, float]:
    if metric_name:
        return metric_name, float(payload[metric_name])
    for candidate in DEFAULT_METRIC_CANDIDATES:
        if candidate in payload:
            return candidate, float(payload[candidate])
    raise KeyError(f"Could not find a metric in payload keys: {sorted(payload.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows: List[Dict] = []
    for path_str in args.input_json:
        path = Path(path_str)
        payload = load_payload(path)
        metric_name, metric_value = resolve_metric(payload, args.metric)
        benchmark_name = payload.get("benchmark_name") or path.stem
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "source": str(path),
            }
        )

    macro_average = sum(row["metric_value"] for row in rows) / len(rows)
    summary = {
        "benchmarks": rows,
        "macro_average": macro_average,
    }

    for row in rows:
        print(f"{row['benchmark_name']}: {row['metric_name']}={row['metric_value']:.4f}")
    print(f"macro_average: {macro_average:.4f}")

    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
