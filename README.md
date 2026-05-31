# XSKILLRL

This repository is organized as a monorepo-style research workspace:

```text
.
+-- XSkill-dev/   # XSkill-based visual reasoning, benchmark preparation, and RL export utilities
+-- SkillRL/      # SkillRL/verl trainer code with an XSkill environment entry point
```

Use `XSkill-dev/` to prepare benchmark splits, run XSkill accumulation, export SFT/GRPO records, and build launch scripts. Use `SkillRL/` to run the generated SFT, GRPO, and validation commands.

The XSkill inference and memory path remains under `XSkill-dev/eval/`. The SkillRL integration is added as an external RL bridge so XSkill baseline functionality is kept separate from parameter training.

## Quick Start

Commands below assume the current directory is the repository root:

```bash
cd XSkill-dev
```

Prepare or copy local resources that are intentionally not tracked by Git:

- model weights, for example `../models/Qwen3-VL-8B-Instruct`
- benchmark images, for example `../images` or `benchmark/...`
- generated memory banks under `memory_bank/`
- generated parquet/jsonl files under `output/`
- checkpoints under `../SkillRL/checkpoints/`

The main workflow is:

1. Prepare benchmark splits under `benchmark/`.
2. Build `benchmark/_mixed_protocol/train_core.json` and `global_val.json`.
3. Run XSkill accumulation with `eval/run_exskill_train.sh`.
4. Convert accumulated skills to `memory_bank/test/skillrl_skill_bank.json`.
5. Export SFT or GRPO data under `output/`.
6. Generate portable launch scripts under `output/sft_runs/`, `output/rl_runs/`, or `output/eval_runs/`.
7. Run the generated scripts from the repository root after activating the target environment.

Detailed commands are in [XSkill-dev/docs/XSKILL_SKILLRL_USAGE_CN.md](XSkill-dev/docs/XSKILL_SKILLRL_USAGE_CN.md).

## Current Integration Notes

- XSkill accumulation now appends benchmark `source` URLs into the task prompt when present. The model is instructed to use the `visit` tool on those URLs when relevant, which is important for MMBrowseComp samples that provide useful external references without local images.
- Accumulation scoring now evaluates only the parsed `final_answer`. Rollouts whose `final_answer` starts with `Error`, remains a tool call, or reaches the max-turn limit without a concrete answer are forced to `0.0` instead of letting the LLM judge mark the trajectory as correct.
- Cleanup utilities for reruns are under `XSkill-dev/scripts/`, including:
  - `cleanup_error_scored_samples.py` for removing old samples where an error final answer was incorrectly scored as `1.0`.
  - `cleanup_mmbrowse_source_outputs.py` for removing MMBrowseComp source-URL samples after prompt-format changes.

Runtime outputs, generated data, memory banks, checkpoints, and downloaded benchmark assets remain intentionally ignored by Git.
