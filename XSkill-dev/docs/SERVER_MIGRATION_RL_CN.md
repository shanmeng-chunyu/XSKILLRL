# XSKILLRL 新服务器迁移与 RL 训练准备

本文档假设新服务器只拿到 `XSKILLRL/` 仓库代码。以下目录通常不会进入 git，需要在新服务器重新准备或从旧服务器手动复制：

- `XSkill-dev/benchmark/`
- `XSkill-dev/output/`
- `XSkill-dev/memory_bank/`
- `XSkill-dev/logs/`
- `SkillRL/checkpoints/`
- 本地模型目录，例如 `/path/to/models/Qwen3-VL-8B-Instruct`

推荐新服务器目录结构保持为：

```text
XSKILLRL/
  XSkill-dev/
  SkillRL/
```

## 1. 创建环境

### 1.1 XSkill 环境

用于 benchmark 准备、XSkill accumulation/inference、SkillBank 转换、RL parquet 导出。

```bash
cd /path/to/XSKILLRL/XSkill-dev
conda create -n xskill python=3.10 -y
conda activate xskill
pip install -r requirements.txt
```

如果需要使用本地 embedding：

```bash
export EXPERIENCE_EMBEDDING_BACKEND=local
export EXPERIENCE_EMBEDDING_MODEL=/path/to/models/bge-m3
```

### 1.2 SkillRL 环境

用于 GRPO/verl 训练。建议独立于 XSkill 环境。

```bash
cd /path/to/XSKILLRL/SkillRL
conda create -n skillrl python=3.10 -y
conda activate skillrl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如果 `flash-attn` 安装失败，通常需要先确认 CUDA、PyTorch 和编译工具链匹配；也可以按服务器环境安装对应 wheel。不要安装 `huggingface-hub>=1.0`，当前 transformers/SkillRL 组合要求 `<1.0`。

## 2. 模型准备

建议把 Qwen3-VL 模型提前下载到新服务器本地，例如：

```text
/path/to/models/Qwen3-VL-8B-Instruct
```

之后所有脚本都使用这个本地路径：

```bash
MODEL_PATH=/path/to/models/Qwen3-VL-8B-Instruct
```

SkillRL 训练使用内部 vLLM rollout，不需要再单独启动 OpenAI-compatible vLLM serve。XSkill accumulation/inference 阶段仍需要你另外启动 vLLM serve。

## 3. 数据准备

如果你没有从旧服务器复制 `benchmark/`、`output/`、`memory_bank/`，需要重新生成。

### 3.1 Benchmark split

先把原始 benchmark 数据放到 `XSkill-dev/benchmark/` 下，然后按项目已有 prepare 脚本生成各 benchmark 的 `train/test` 和 `_mixed_protocol`。最终至少需要：

```text
XSkill-dev/benchmark/_mixed_protocol/train_core.json
XSkill-dev/benchmark/_mixed_protocol/global_val.json
```

如果旧服务器已经有这些文件，可以直接复制整个：

```text
XSkill-dev/benchmark/
```

### 3.2 本地化远程图片

RL 训练不建议实时访问 GitHub raw 图片。先下载远程图片并重写 JSON：

```bash
cd /path/to/XSKILLRL/XSkill-dev
conda activate xskill

python scripts/localize_remote_images.py \
  --input-json benchmark/_mixed_protocol/train_core.json \
  --output-json output/rl_data/train_core_local_images.json \
  --input-json benchmark/_mixed_protocol/global_val.json \
  --output-json output/rl_data/global_val_local_images.json \
  --cache-dir benchmark/_remote_images \
  --path-prefix benchmark/_remote_images \
  --manifest-path output/rl_data/remote_image_manifest.json \
  --retries 8 \
  --timeout 30 \
  --sleep 2.0
```

迁移时如果复制了 `benchmark/_remote_images/`，脚本会复用缓存，不会重复下载。

### 3.3 SkillBank 准备

如果已有 XSkill accumulation 产物：

```text
memory_bank/test/SKILL.md
```

转换成 SkillRL 使用的 JSON：

```bash
python scripts/convert_xskill_skill_to_skillrl_bank.py \
  --input-skill-md memory_bank/test/SKILL.md \
  --output-json memory_bank/test/skillrl_skill_bank.json \
  --max-skills 24
```

如果没有 `memory_bank/test/SKILL.md`，需要先运行 XSkill accumulation，或者临时不传 `--skill-bank-json` 跑 no-memory GRPO baseline。

### 3.4 导出 SkillRL parquet

```bash
python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec output/rl_data/train_core_local_images.json \
  --output-path output/rl_data/skillrl_train.parquet \
  --mixing-strategy concat \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --top-k 6

python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec output/rl_data/global_val_local_images.json \
  --output-path output/rl_data/skillrl_val.parquet \
  --mixing-strategy concat \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --top-k 6
```

## 4. 迁移路径检查

训练前先检查路径是否可解析：

```bash
python scripts/check_rl_migration_paths.py \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --sample-limit 500
```

如果检查失败，优先修复缺失文件或图片路径，不要直接启动长时间训练。

## 5. 生成可迁移 GRPO 脚本

脚本默认会在运行时自动推断：

- `XSKILL_REPO_ROOT`
- `SKILLRL_REPO_ROOT`
- `XSKILL_IMAGE_ROOT`

8 卡 A800 示例：

```bash
cd /path/to/XSKILLRL/XSkill-dev
conda activate xskill

python scripts/build_skillrl_grpo_command.py \
  --model-path /path/to/models/Qwen3-VL-8B-Instruct \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --n-gpus-per-node 8 \
  --group-size 4 \
  --train-batch-size 64 \
  --val-batch-size 64 \
  --rollout-max-model-len 16384 \
  --rollout-max-num-batched-tokens 16384 \
  --actor-ppo-max-token-len-per-gpu 32768 \
  --rollout-log-prob-max-token-len-per-gpu 32768 \
  --ref-log-prob-max-token-len-per-gpu 32768 \
  --output-script output/rl_runs/qwen3vl8b_grpo.sh
```

如果模型路径在不同服务器不同，只需要重新生成脚本并改 `--model-path`。

## 6. 启动训练

```bash
cd /path/to/XSKILLRL/SkillRL
conda activate skillrl

ray stop --force
HYDRA_FULL_ERROR=1 bash ../XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh
```

如需手动指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HYDRA_FULL_ERROR=1 bash ../XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh
```

## 7. 常见迁移问题

### 图片找不到

如果报：

```text
FileNotFoundError: benchmark/_remote_images/...
```

先运行：

```bash
python scripts/check_rl_migration_paths.py --sample-limit 1000
```

确认 `benchmark/_remote_images/` 是否复制或重新下载完成。

### vLLM 版本不兼容

如果报：

```text
No module named 'vllm.lora.models'
```

通常是安装到了过新的 vLLM。按 `SkillRL/requirements.txt` 安装，使用 `vllm==0.11.0`。

### huggingface-hub 版本不兼容

如果报：

```text
huggingface-hub>=0.23.2,<1.0 is required
```

执行：

```bash
pip install 'huggingface-hub>=0.23.2,<1.0'
```

### 多图样本过长

如果报 image tokens 和 image features 不一致，通常是视觉 token 被截断。可以先降低图片 token：

```bash
export XSKILL_RL_IMAGE_MAX_PIXELS=786432
export XSKILL_RL_IMAGE_MIN_MAX_PIXELS=262144
export XSKILL_RL_IMAGE_MIN_PIXELS=16384
```

或者提高：

```bash
data.max_prompt_length
actor_rollout_ref.rollout.max_model_len
actor_rollout_ref.rollout.max_num_batched_tokens
```

