# AGENTS.md

This repository is now a monorepo-style workspace.

- `XSkill-dev/` contains the XSkill-derived project, mixed benchmark protocol, SkillRL/GRPO dataset export, SkillBank conversion, command builder, and the XSkill visual QA bridge used by SkillRL.
- `SkillRL/` contains the SkillRL/verl training stack. Its environment factory has an `xskill` branch that imports `xskill_rl.skillrl_bridge` from `env.xskill.repo_root` or `XSKILL_REPO_ROOT`.

For detailed project background and XSkill-side constraints, read `XSkill-dev/AGENTS.md` first.

Do not commit runtime outputs, downloaded benchmark raw data, model checkpoints, `wandb/`, `logs/`, `output/`, `memory_bank/`, Python caches, or generated Parquet files.
