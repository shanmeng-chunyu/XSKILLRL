"""Export mixed benchmark samples to JSONL or Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.dataset import load_records_from_spec
from xskill_rl.benchmark_protocol import write_json


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(path: Path, records) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for parquet export") from exc

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet export") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-spec", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mixing-strategy", choices=["concat", "sqrt_size"], default="concat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, manifest = load_records_from_spec(
        args.input_spec,
        mixing_strategy=args.mixing_strategy,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    if output_path.suffix.lower() == ".parquet":
        write_parquet(output_path, records)
    else:
        write_jsonl(output_path, records)

    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)


if __name__ == "__main__":
    main()
