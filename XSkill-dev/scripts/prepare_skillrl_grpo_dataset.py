"""Export XSkill samples in SkillRL/verl GRPO dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import write_json
from xskill_rl.skillrl.verl_export import (
    DEFAULT_SYSTEM_PROMPT,
    load_verl_records_from_spec,
    write_jsonl,
    write_parquet,
)


def _read_system_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SkillRL/verl-compatible GRPO data from XSkill samples."
    )
    parser.add_argument("--input-spec", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mixing-strategy", choices=["concat", "sqrt_size"], default="concat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skill-bank-json", default=None)
    parser.add_argument("--skill-retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--system-prompt-file", default=None)
    args = parser.parse_args()

    records, manifest = load_verl_records_from_spec(
        args.input_spec,
        mixing_strategy=args.mixing_strategy,
        seed=args.seed,
        skills_json_path=args.skill_bank_json,
        retrieval_mode=args.skill_retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        top_k=args.top_k,
        task_specific_top_k=args.task_specific_top_k,
        system_prompt=_read_system_prompt(args.system_prompt_file),
    )

    output_path = Path(args.output_path)
    if output_path.suffix.lower() == ".parquet":
        write_parquet(output_path, records)
    else:
        write_jsonl(output_path, records)
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)


if __name__ == "__main__":
    main()
