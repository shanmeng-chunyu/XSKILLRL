# XSKILLRL

This repository is organized as a two-part research workspace:

```text
.
+-- XSkill-dev/   # XSkill-based visual reasoning, benchmark preparation, and RL export utilities
+-- SkillRL/      # SkillRL/verl trainer code with an XSkill environment entry point
```

Use `XSkill-dev/` to prepare benchmark splits, export SkillRL/GRPO records, and build the launch command. Use `SkillRL/` on the remote training server to run the generated `python -m verl.trainer.main_ppo ...` command.

The XSkill inference and memory path remains under `XSkill-dev/eval/`. The SkillRL integration is added as an external RL bridge so XSkill baseline functionality is kept separate from parameter training.
