# SkillRL GRPO Integration

本仓库现在新增了一个轻量的 `xskill_rl.skillrl` 适配层，用来把 XSkill mixed-benchmark 数据接到 SkillRL 风格的 GRPO 训练流程中。

重要边界：

- 这里接入的是 SkillRL 兼容的数据、SkillBank、GRPO 配置和命令生成层。
- 不把 SkillRL/verl 的完整分布式 trainer vendoring 到本仓库。
- 不修改 `eval/` 下的 XSkill inference / memory 主链路。
- 真正训练仍建议在外部 SkillRL/verl 环境中运行。

## 对齐 SkillRL 的部分

SkillRL 论文和官方仓库中的关键约定在本仓库中对应如下：

- `algorithm.adv_estimator=grpo`
- group-relative advantage: `A_i = (R_i - mean(R)) / std(R)`
- SkillBank JSON:
  - `general_skills`
  - `task_specific_skills`
  - `common_mistakes`
- skills-only memory 配置：
  - `env.use_skills_only_memory=True`
  - `env.skills_only_memory.skills_json_path`
  - `env.skills_only_memory.retrieval_mode`
  - `env.skills_only_memory.top_k`
  - `env.skills_only_memory.enable_dynamic_update`

实现位置：

- `xskill_rl/skillrl/grpo.py`
- `xskill_rl/skillrl/skill_bank.py`
- `xskill_rl/skillrl/verl_export.py`
- `xskill_rl/skillrl/config.py`
- `xskill_rl/skillrl/dynamic_update.py`

## 1. 准备 SkillBank

如果已有 XSkill accumulation 产生的 `SKILL.md`，可以先转换成 SkillRL JSON：

```powershell
py -3 scripts/convert_xskill_skill_to_skillrl_bank.py `
  --input-skill-md memory_bank\test\SKILL.md `
  --output-json memory_bank\test\skillrl_skill_bank.json
```

输出 JSON 与 SkillRL 官方 `SkillsOnlyMemory` 格式一致。

也可以直接手写或由外部 teacher 生成：

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Check Visual Evidence",
      "principle": "Ground each answer in visible image evidence before using external tools.",
      "when_to_apply": "For all visual reasoning questions."
    }
  ],
  "task_specific_skills": {},
  "common_mistakes": []
}
```

## 2. 导出 SkillRL/verl GRPO 数据

```powershell
py -3 scripts/prepare_skillrl_grpo_dataset.py `
  --input-spec benchmark\_mixed_protocol\train_core.json `
  --output-path output\skillrl_train.parquet `
  --skill-bank-json memory_bank\test\skillrl_skill_bank.json `
  --skill-retrieval-mode template `
  --top-k 6
```

验证集：

```powershell
py -3 scripts/prepare_skillrl_grpo_dataset.py `
  --input-spec benchmark\_mixed_protocol\global_val.json `
  --output-path output\skillrl_val.parquet `
  --skill-bank-json memory_bank\test\skillrl_skill_bank.json `
  --skill-retrieval-mode template `
  --top-k 6
```

导出记录包含：

- `data_source`
- `prompt`
- `images`
- `reward_model`
- `extra_info`

其中 `prompt` 是 verl `RLHFDataset` 可读的 chat messages；如果提供 SkillBank，会把检索到的 skills 注入 system prompt 的 `## Retrieved Relevant Experience` 段。

## 3. 生成 SkillRL GRPO 命令

```powershell
py -3 scripts/build_skillrl_grpo_command.py `
  --model-path path\to\sft_model `
  --train-file output\skillrl_train.parquet `
  --val-file output\skillrl_val.parquet `
  --skill-bank-json memory_bank\test\skillrl_skill_bank.json `
  --group-size 8 `
  --n-gpus-per-node 8
```

脚本会输出类似 SkillRL 官方 `examples/grpo_trainer/*_skills.sh` 的 Hydra override 命令。

注意：官方 SkillRL 的 `verl.trainer.main_ppo` 依赖其 `agent_system.environments.make_envs`。如果要完全复用官方 online environment rollout，需要在外部 SkillRL/verl 环境注册 XSkill visual reasoning environment，或使用已经预注入 skill 的 Parquet 数据接入支持单轮 completion reward 的 GRPO trainer。

## 4. 动态 SkillBank 更新

`xskill_rl.skillrl.dynamic_update` 提供了 SkillRL 论文里的失败样本选择和 prompt 构造：

- `select_failed_trajectories`
- `build_failure_skill_prompt`
- `parse_new_skills`

这部分不默认调用任何 API。后续可以把 XSkill/GRPO 验证失败 trajectory 交给 teacher model，再把返回的新 skills merge 到 SkillBank。

## 5. 不影响 XSkill 的约束

本次接入不改变：

- `eval/infer_api.py`
- `eval/run_exskill_train.sh`
- `eval/run_exskill_inference.sh`
- `eval/exskill/`

XSkill baseline 仍按原方式做 accumulation 和 inference。SkillRL/GRPO 相关入口只在 `xskill_rl/skillrl/` 和新增 `scripts/` 中。
