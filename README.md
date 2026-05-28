# XSKILLRL

This repository is organized as a two-part research workspace:

```text
.
+-- XSkill-dev/   # XSkill-based visual reasoning, benchmark preparation, and RL export utilities
+-- SkillRL/      # SkillRL/verl trainer code with an XSkill environment entry point
```

Use `XSkill-dev/` to prepare benchmark splits, export SkillRL/GRPO records, and build the launch command. Use `SkillRL/` on the remote training server to run the generated `python -m verl.trainer.main_ppo ...` command.

The XSkill inference and memory path remains under `XSkill-dev/eval/`. The SkillRL integration is added as an external RL bridge so XSkill baseline functionality is kept separate from parameter training.

## Current Integration Notes

- XSkill accumulation now appends benchmark `source` URLs into the task prompt when present. The model is instructed to use the `visit` tool on those URLs when relevant, which is important for MMBrowseComp samples that provide useful external references without local images.
- Accumulation scoring now evaluates only the parsed `final_answer`. Rollouts whose `final_answer` starts with `Error`, remains a tool call, or reaches the max-turn limit without a concrete answer are forced to `0.0` instead of letting the LLM judge mark the trajectory as correct.
- Cleanup utilities for reruns are under `XSkill-dev/scripts/`, including:
  - `cleanup_error_scored_samples.py` for removing old samples where an error final answer was incorrectly scored as `1.0`.
  - `cleanup_mmbrowse_source_outputs.py` for removing MMBrowseComp source-URL samples after prompt-format changes.

Runtime outputs, generated data, memory banks, checkpoints, and downloaded benchmark assets remain intentionally ignored by git.
