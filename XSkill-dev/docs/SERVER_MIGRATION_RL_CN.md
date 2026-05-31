# XSKILLRL 新服务器迁移与 RL 训练准备

本文档面向“新服务器只迁移 Git 仓库内容”的场景。环境如果已经准备好，仍然需要额外准备模型、benchmark 数据、图片缓存、SkillBank/MemoryBank、RL Parquet 数据和训练脚本。

仓库默认不会提交以下运行产物：

| 类型 | 典型路径 | 是否需要重新准备 |
| --- | --- | --- |
| Qwen3-VL 模型权重 | `../models/Qwen3-VL-8B-Instruct` 或任意本地路径 | 是 |
| embedding 模型权重 | `models/bge-m3`、`../models/bge-m3` 或 HF cache | 使用本地经验检索时需要 |
| benchmark 原始/切分数据 | `XSkill-dev/benchmark/` | 是 |
| 远程图片本地缓存 | `XSkill-dev/benchmark/_remote_images/` | 推荐准备 |
| accumulation 记忆库 | `XSkill-dev/memory_bank/test/` | 使用 SkillBank 时需要 |
| RL 训练数据 | `XSkill-dev/output/rl_data/*.parquet` | 是 |
| RL 启动脚本 | `XSkill-dev/output/rl_runs/*.sh` | 是 |
| 训练日志和 checkpoint | `SkillRL/checkpoints/`、`logs/`、`wandb/` | 不需要，除非续训 |

## 1. 克隆仓库

```bash
git clone <your-repo-url> XSKILLRL
cd XSKILLRL
```

仓库结构应为：

```text
XSKILLRL/
  XSkill-dev/
  SkillRL/
```

后续命令默认在 `XSKILLRL/XSkill-dev` 下执行，RL 训练启动时再切到 `XSKILLRL/SkillRL`。

## 2. 准备本地模型

如果已经下载好 Qwen3-VL，后续脚本直接传本地路径即可：

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

如果还没有下载，可以在网络可用的机器上执行：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /path/to/Qwen3-VL-8B-Instruct \
  --local-dir-use-symlinks False
```

如果使用本地经验检索 embedding，建议提前下载：

```bash
hf download BAAI/bge-m3 \
  --local-dir /path/to/bge-m3 \
  --local-dir-use-symlinks False
```

然后在 accumulation 或检索相关流程中设置：

```bash
export EXPERIENCE_EMBEDDING_BACKEND=local
export EXPERIENCE_EMBEDDING_MODEL=/path/to/bge-m3
```

## 3. 准备 benchmark 数据

Git 仓库不包含 benchmark 原始数据和运行期切分文件。新服务器需要二选一：

| 方式 | 适用场景 |
| --- | --- |
| 从旧服务器复制 `XSkill-dev/benchmark/` | 最推荐，能保证实验数据完全一致 |
| 在新服务器重新运行 benchmark 准备脚本 | 适合重新开始实验 |

RL 至少需要这些切分文件：

```text
XSkill-dev/benchmark/_mixed_protocol/train_core.json
XSkill-dev/benchmark/_mixed_protocol/global_val.json
```

如果旧服务器已有完整 benchmark，直接复制：

```bash
rsync -av old_server:/path/to/XSKILLRL/XSkill-dev/benchmark/ ./benchmark/
```

## 4. 提前下载远程图片

不要在 RL 训练时实时访问 GitHub/raw URL 图片。网络抖动会导致 `Connection reset by peer`、`FileNotFoundError` 或图像 token 对不齐。

如果你从旧服务器复制了已经下载好的图片，推荐放到仓库同级目录：

```text
XSKILLRL/
  images/
    agentvista/
      agentvista_000002_field_images_0.png
  XSkill-dev/
  SkillRL/
```

训练脚本会优先把 `XSKILL_IMAGE_ROOT` 指向 `XSKILLRL/images`。即使 parquet 里还保留旧服务器绝对路径，例如 `/data/luzy/xskill/images/agentvista/a.png`，当前路径解析也会尝试用 `images/` 后面的相对路径重定位到 `XSKILLRL/images/agentvista/a.png`。

在 `XSkill-dev` 目录执行：

```bash
mkdir -p output/rl_data benchmark/_remote_images

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

这一步会把远程图片下载到 `benchmark/_remote_images/`，并生成使用本地图片路径的新 JSON。后续导出 RL 数据时优先使用 `*_local_images.json`。

## 5. 准备 MemoryBank / SkillBank

如果只迁移 Git 内容，`memory_bank/` 不会存在。你有三种选择：

| 方案 | 说明 |
| --- | --- |
| 复制旧服务器 `XSkill-dev/memory_bank/test/` | 推荐用于复现实验或继续原实验 |
| 重新跑 XSkill accumulation | 推荐用于新实验 |
| 不使用 SkillBank，跑 no-memory GRPO baseline | 适合排查 RL 训练链路 |

如果旧服务器已有 `memory_bank/test/SKILL.md`，复制后转换为 SkillRL 可读取的 JSON：

```bash
mkdir -p memory_bank/test

python scripts/convert_xskill_skill_to_skillrl_bank.py \
  --input-skill-md memory_bank/test/SKILL.md \
  --output-json memory_bank/test/skillrl_skill_bank.json \
  --max-skills 24
```

转换后应存在：

```text
XSkill-dev/memory_bank/test/skillrl_skill_bank.json
```

## 6. 导出 RL Parquet

有 SkillBank 时：

```bash
mkdir -p output/rl_data

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

无 SkillBank baseline 时，去掉 `--skill-bank-json`：

```bash
python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec output/rl_data/train_core_local_images.json \
  --output-path output/rl_data/skillrl_train.parquet \
  --mixing-strategy concat

python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec output/rl_data/global_val_local_images.json \
  --output-path output/rl_data/skillrl_val.parquet \
  --mixing-strategy concat
```

## 7. 迁移路径检查

导出 Parquet 后，先运行检查脚本：

```bash
python scripts/check_rl_migration_paths.py \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --sample-limit 500
```

如果跑无 SkillBank baseline：

```bash
python scripts/check_rl_migration_paths.py \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --sample-limit 500
```

检查通过后再开始 RL。若这里发现图片缺失，先重新运行第 4 步，不要直接训练。

## 8. 生成 RL 启动脚本

8 卡 A800 示例：

```bash
mkdir -p output/rl_runs

python scripts/build_skillrl_grpo_command.py \
  --model-path "$MODEL_PATH" \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --n-gpus-per-node 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --train-batch-size 64 \
  --val-batch-size 64 \
  --ppo-mini-batch-size 64 \
  --ppo-micro-batch-size-per-gpu 1 \
  --group-size 4 \
  --max-prompt-length 6000 \
  --max-response-length 2048 \
  --rollout-max-model-len 24576 \
  --rollout-max-num-batched-tokens 24576 \
  --rollout-gpu-memory-utilization 0.5 \
  --model-dtype bfloat16 \
  --output-script output/rl_runs/qwen3vl8b_grpo.sh
```

如果跑无 SkillBank baseline，去掉 `--skill-bank-json memory_bank/test/skillrl_skill_bank.json`。

脚本会自动写入可迁移路径：

```bash
export XSKILL_REPO_ROOT=...
export SKILLRL_REPO_ROOT=...
export XSKILL_IMAGE_ROOT=...
```

因此移动到新服务器后，应重新生成启动脚本，不建议直接复用旧服务器 `output/rl_runs/*.sh`。

## 9. 启动 RL

```bash
cd ../SkillRL
ray stop --force
HYDRA_FULL_ERROR=1 bash ../XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh
```

训练使用哪些 GPU 由启动脚本中的 `CUDA_VISIBLE_DEVICES` 和 Hydra 参数 `trainer.n_gpus_per_node` 决定。

## 10. 迁移后最小检查清单

开始训练前确认：

| 检查项 | 期望结果 |
| --- | --- |
| `echo $MODEL_PATH` | 指向本地 Qwen3-VL 目录 |
| `ls benchmark/_mixed_protocol/*.json` | 能看到 train/val 切分 |
| `ls benchmark/_remote_images` | 有已下载图片 |
| `ls output/rl_data/*.parquet` | 有 train/val parquet |
| `python scripts/check_rl_migration_paths.py ...` | 无缺图、坏路径、空图异常 |
| `ls memory_bank/test/skillrl_skill_bank.json` | 使用 SkillBank 时存在 |
| `bash output/rl_runs/qwen3vl8b_grpo.sh --help` | 不适用，脚本不是 CLI；直接从 `SkillRL` 目录运行 |

如果只想验证链路，建议先使用较小数据和较短训练：

```bash
python scripts/build_skillrl_grpo_command.py \
  --model-path "$MODEL_PATH" \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --n-gpus-per-node 2 \
  --cuda-visible-devices 0,1 \
  --train-batch-size 16 \
  --val-batch-size 16 \
  --ppo-mini-batch-size 16 \
  --ppo-micro-batch-size-per-gpu 1 \
  --group-size 2 \
  --max-response-length 512 \
  --total-epochs 1 \
  --output-script output/rl_runs/smoke_grpo.sh

cd ../SkillRL
HYDRA_FULL_ERROR=1 bash ../XSkill-dev/output/rl_runs/smoke_grpo.sh
```

Smoke test 能跑通后，再切回正式配置。
