# XSkill + SkillRL Runtime Skill Retrieval and SFT

本文档记录当前推荐流程：先做 SFT 冷启动，再用 GRPO 训练；GRPO 阶段不再依赖 parquet 中写死的 skill prompt，而是在 XSkill 环境 `reset()` 时按当前 SkillBank 动态检索。

## 1. 关键变化

| 模块 | 行为 |
| --- | --- |
| `xskill_rl/skillrl_bridge.py` | 当 `env.use_skills_only_memory=True` 时，每个 rollout/validation 样本按 `problem` 动态检索 SkillBank，并拼接到 prompt。 |
| GRPO parquet | 推荐导出无 skill prompt 的 parquet，只保存题目、答案、图片和 env kwargs。 |
| Validation skill update | `trainer.test_freq` 触发 validation 后，如果开启 `env.skills_only_memory.enable_dynamic_update=True`，会从失败 validation trajectory 生成新 skill，加入训练环境的 SkillBank，并保存 `updated_skills_step*.json`。 |
| SFT | 新增 multimodal SFT 数据导出与启动脚本生成器，用于先训练输出格式、答案约束和 skill 使用风格；默认保留并读取图片。 |
| Tool / multi-turn RL | XSkill 环境现在默认开启工具协议和真实工具执行；GRPO 默认 `env.max_steps=20`，模型可在一次 rollout 内多轮调用 `web_search`、`visit`、`image_search`、`code_interpreter`、`zoom`，最后用 `<answer>...</answer>` 作答。 |

注意：validation 生成的新 skill 默认只加入后续训练环境，不回灌到本轮 validation 环境，避免验证集泄漏。

## 2. 重新导出无 Skill 的 GRPO 数据

导出 GRPO parquet 时不要传 `--skill-bank-json`：

```bash
cd XSkill-dev

python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec configs/benchmarks/mixed_train_core.json \
  --output-path output/rl_data/skillrl_train.parquet \
  --mixing-strategy concat \
  --seed 42

python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec configs/benchmarks/mixed_val_core.json \
  --output-path output/rl_data/skillrl_val.parquet \
  --mixing-strategy concat \
  --seed 42
```

SkillBank 只在 GRPO 启动脚本生成时传入。

## 3. 生成 SFT 数据

SFT trainer 当前支持 multimodal SFT：导出的 parquet 默认保留 `images` 字段，训练时通过 Qwen3-VL `AutoProcessor` 读取图片并生成 `pixel_values` / `image_grid_thw`。对于 TIRBench 这类题干主要在图片里的样本，即使文本 `problem` 为空，也会保留图片并使用通用图上答题提示，避免纯文本 SFT 造成“无图无题干 -> 答案”的坏监督。

```bash
cd XSkill-dev

python scripts/prepare_skillrl_sft_dataset.py \
  --input-file output/rl_data/skillrl_train.parquet \
  --output-path output/sft_data/xskill_sft_train.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --top-k 6 \
  --response-format answer_tag

python scripts/prepare_skillrl_sft_dataset.py \
  --input-file output/rl_data/skillrl_val.parquet \
  --output-path output/sft_data/xskill_sft_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --top-k 6 \
  --response-format answer_tag
```

如果需要临时回到纯文本 SFT，可以在导出时加入 `--no-include-images`，并在生成训练脚本时加入 `--no-enable-multimodal`。一般不建议对当前混合视觉 benchmark 使用纯文本 SFT。

SFT 是离线监督训练，不会像 GRPO rollout 那样在训练中实时执行工具。当前 SFT 导出默认把工具调用协议写入 prompt，让模型学习可用工具格式；如果要做原 SkillRL 那种“工具调用轨迹 SFT”，需要额外把 accumulation/evaluation 轨迹中的 assistant tool call、tool observation 和最终答案导出为监督样本。

## 4. 生成并运行 SFT 脚本

```bash
cd XSkill-dev
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct

python scripts/build_skillrl_sft_command.py \
  --model-path "$MODEL_PATH" \
  --train-file output/sft_data/xskill_sft_train.parquet \
  --val-file output/sft_data/xskill_sft_val.parquet \
  --nproc-per-node 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --train-batch-size 64 \
  --micro-batch-size-per-gpu 1 \
  --max-length 8192 \
  --enable-multimodal \
  --total-epochs 1 \
  --output-script output/sft_runs/qwen3vl8b_sft.sh
```

提交到计算节点时直接运行生成的脚本即可：

```bash
bash XSkill-dev/output/sft_runs/qwen3vl8b_sft.sh
```

SFT checkpoint 默认保存在：

```text
SkillRL/checkpoints/xskill_sft/qwen3vl8b_sft/global_step_*
```

## 5. 生成 Runtime Skill GRPO 脚本

GRPO 的 `--model-path` 可以指向原始模型，也可以指向 SFT 后的 checkpoint：

默认 GRPO 超参已按论文配置对齐：学习率 `1e-6`、训练 batch size `16`、group size `8`、梯度累加步数 `4`。在当前 verl actor 中，梯度累加步数由 `ppo_mini_batch_size // ppo_micro_batch_size_per_gpu` 决定，因此下面使用 `16 // 4 = 4`。

```bash
cd XSkill-dev
export MODEL_PATH=/path/to/SFT/checkpoint/or/Qwen3-VL-8B-Instruct

python scripts/build_skillrl_grpo_command.py \
  --model-path "$MODEL_PATH" \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --retrieval-mode template \
  --top-k 6 \
  --enable-dynamic-update \
  --n-gpus-per-node 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --train-batch-size 16 \
  --val-batch-size 16 \
  --ppo-mini-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --group-size 8 \
  --max-steps 20 \
  --enable-tools \
  --enabled-tools web_search,visit,image_search,code_interpreter,zoom \
  --max-prompt-length 16000 \
  --max-response-length 2048 \
  --rollout-max-model-len 24576 \
  --rollout-max-num-batched-tokens 24576 \
  --rollout-gpu-memory-utilization 0.5 \
  --model-dtype bfloat16 \
  --output-script output/rl_runs/qwen3vl8b_grpo_runtime_skill.sh
```

运行：

```bash
bash XSkill-dev/output/rl_runs/qwen3vl8b_grpo_runtime_skill.sh
```

## 6. 如何确认不是离线写死 Skill

检查 GRPO parquet：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("output/rl_data/skillrl_train.parquet")
print(df.iloc[0]["prompt"])
PY
```

如果里面没有 `## Retrieved Relevant Experience`，说明数据是无 skill prompt 的。

训练日志中应出现：

```text
[XSkillVisualQAEnvironment] Runtime skill retrieval enabled for train: ...
[XSkillVisualQAEnvironment] Runtime skill retrieval enabled for val: ...
```

rollout prompt 中应在运行时出现：

```text
## Retrieved Relevant Experience
```

如果开启了 dynamic update，validation 后可能出现：

```text
[SkillUpdate] Low success tasks: ...
[SkillUpdate] Added ... new skills to training envs
[SkillUpdate] Saved updated skill bank to .../updated_skills_step*.json
```

如果没有配置 SkillUpdater 所需的外部 LLM 环境变量，会跳过 skill 更新，但训练不会因此中断。
