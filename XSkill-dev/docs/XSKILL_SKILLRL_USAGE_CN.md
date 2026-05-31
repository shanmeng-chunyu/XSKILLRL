# XSkill + SkillRL 使用说明

本文档默认仓库结构如下：

```text
XSKILLRL/
├── XSkill-dev/
└── SkillRL/
```

除非特别说明，命令中的相对路径都以 `XSKILLRL/XSkill-dev` 为工作目录。运行 SkillRL 训练或评估脚本时，生成的 `.sh` 会自动切换到 `XSKILLRL/SkillRL`。

## 1. 不进入 Git 的本地资源

以下内容需要在服务器本地准备，但不要提交到 Git：

| 资源 | 推荐位置 | 说明 |
| --- | --- | --- |
| Qwen3-VL 模型 | `../models/Qwen3-VL-8B-Instruct` 或任意本地路径 | 通过脚本参数 `--model-path` / `--base-model-path` 指定 |
| BGE embedding 模型 | `models/bge-m3` 或任意本地路径 | 本地 experience/skill embedding 时使用 |
| benchmark 原始图片 | `../images/` 或 `benchmark/` | 路径解析会优先尝试 `XSKILL_IMAGE_ROOT` |
| 运行输出 | `output/` | rollout、报告、导出的 parquet/jsonl |
| memory bank | `memory_bank/` | accumulation 生成的 experience/skill |
| checkpoints | `../SkillRL/checkpoints/` | SFT/RL/评估保存结果 |

`.gitignore` 已忽略 `output/`、`memory_bank/`、checkpoint、parquet、缓存和下载数据。

## 2. Benchmark 准备

单个 benchmark 先标准化成 `all.json/train.json/test.json`。示例：

```bash
python scripts/prepare_tirbench.py \
  --input-json path/to/tirbench.json \
  --image-root benchmark/TIR-bench/data \
  --output-dir benchmark/TIR-Bench
```

其它准备脚本位于 `scripts/`：

- `prepare_visualtoolbench.py`
- `prepare_tirbench.py`
- `prepare_mmsearch_plus.py`
- `prepare_agentvista.py`
- `prepare_mmbrowsecomp.py`

构造统一 mixed protocol：

```bash
python scripts/prepare_mixed_train_pool.py \
  --train-json benchmark/MM-BrowseComp/train.json \
  --train-json benchmark/TIR-Bench/train.json \
  --train-json benchmark/MMSearch-Plus/train.json \
  --train-json benchmark/AgentVista/train.json \
  --train-json benchmark/VisualToolBench/train.json \
  --output-dir benchmark/_mixed_protocol
```

主要输出：

- `benchmark/_mixed_protocol/train_core.json`
- `benchmark/_mixed_protocol/global_val.json`
- `benchmark/_mixed_protocol/mixing_manifest.json`

如果修改了样本 ID 或图片路径，建议重新生成 mixed protocol，保证后续 accumulation、SFT、RL 看到的是同一套样本。

## 3. XSkill Accumulation

先启动 OpenAI-compatible VLM 服务，例如本地 vLLM。模型路径可以是本地路径或 Hugging Face 名称：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ../models/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host 0.0.0.0 \
  --port 8002 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image":16}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

在 `eval/run_exskill_train.sh` 中重点确认：

```bash
export DATA_PATH="benchmark/_mixed_protocol/train_core.json"
export OUTPUT_DIR="output/xskill_accum/qwen3vl8b_mixed_train_core_seed42"
export LOG_OUTPUT_DIR="logs/xskill_accum/qwen3vl8b_mixed_train_core_seed42"

export LOCAL_VLM_ENDPOINTS="http://127.0.0.1:8002/v1/chat/completions"
export REASONING_END_POINTS="${LOCAL_VLM_ENDPOINTS}"
export VERIFIER_END_POINTS="${LOCAL_VLM_ENDPOINTS}"
export EXPERIENCE_END_POINTS="${LOCAL_VLM_ENDPOINTS}"
export IMAGE_SEARCH_CAPTION_ENDPOINTS="${LOCAL_VLM_ENDPOINTS}"

export SEARCH_API_PROVIDER="bocha"
export IMAGE_SEARCH_PROVIDER="bocha"
export BOCHA_API_KEY="your_key"
export VISIT_BACKEND="local"

export EXPERIENCE_EMBEDDING_BACKEND="local"
export EXPERIENCE_EMBEDDING_MODEL="models/bge-m3"
```

运行：

```bash
bash eval/run_exskill_train.sh
```

Accumulation 支持跳过已完成样本。若中途中断，重新运行同一 `OUTPUT_DIR` 会跳过已有完整 rollout 的样本；如果删除了某些样本目录，这些样本会被重新跑。

## 4. Accumulation 报告与 SkillBank

汇总 accumulation 结果：

```bash
python scripts/summarize_accumulation_run.py \
  --output-dir output/xskill_accum/qwen3vl8b_mixed_train_core_seed42 \
  --data-file benchmark/_mixed_protocol/train_core.json \
  --memory-dir memory_bank/test \
  --api-timings output/xskill_accum/qwen3vl8b_mixed_train_core_seed42/api_timings.jsonl \
  --k 4 \
  --write-default-reports
```

报告默认写入：

```text
output/accumulation_reports/qwen3vl8b_mixed_train_core_seed42/
```

将 XSkill skill 转成 SkillRL 可检索 SkillBank：

```bash
python scripts/convert_xskill_skill_to_skillrl_bank.py \
  --input-skill-md memory_bank/test/SKILL.md \
  --output-json memory_bank/test/skillrl_skill_bank.json
```

如果需要检查工具调用次数：

```bash
python scripts/analyze_accumulation_tool_calls.py \
  --output-dir output/xskill_accum/qwen3vl8b_mixed_train_core_seed42
```

## 5. 导出训练和评估数据

导出 SkillRL/verl GRPO 训练数据：

```bash
python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec benchmark/_mixed_protocol/train_core.json \
  --output-path output/rl_data/skillrl_train.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode none
```

导出 global validation 数据：

```bash
python scripts/prepare_skillrl_grpo_dataset.py \
  --input-spec benchmark/_mixed_protocol/global_val.json \
  --output-path output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode none
```

说明：

- 推荐训练和评估数据不要把 skill prompt 写死进 parquet。
- 是否开启 skill 检索由运行脚本里的 `env.use_skills_only_memory` 决定。
- 这样 `with_skill` / `no_skill` 对比只改变 runtime SkillBank 开关，数据本身保持一致。

## 6. SFT

当前仓库提供的 SFT 导出脚本会把标准化样本转成 multimodal SFT parquet，并保留图片字段。它默认使用样本里的 `solution/answer` 作为监督回答；如果要严格复现 SkillRL 论文里的“只从成功 rollout 轨迹提取 SFT 样本”，需要先基于 accumulation 成功轨迹另行筛选输入文件。

```bash
python scripts/prepare_skillrl_sft_dataset.py \
  --input-spec benchmark/_mixed_protocol/train_core.json \
  --output-path output/sft_data/xskill_sft_train.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --include-images
```

验证集也按同样方式导出：

```bash
python scripts/prepare_skillrl_sft_dataset.py \
  --input-spec benchmark/_mixed_protocol/global_val.json \
  --output-path output/sft_data/xskill_sft_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --skill-retrieval-mode template \
  --include-images
```

生成 SFT 脚本：

```bash
python scripts/build_skillrl_sft_command.py \
  --model-path ../models/Qwen3-VL-8B-Instruct \
  --train-file output/sft_data/xskill_sft_train.parquet \
  --val-file output/sft_data/xskill_sft_val.parquet \
  --output-dir ../SkillRL/checkpoints/xskill_sft/qwen3vl8b_sft \
  --output-script output/sft_runs/qwen3vl8b_sft.sh \
  --cuda-visible-devices 0,1,2,3
```

运行：

```bash
bash output/sft_runs/qwen3vl8b_sft.sh
```

SFT checkpoint 通常位于：

```text
../SkillRL/checkpoints/xskill_sft/qwen3vl8b_sft/global_step_x/
```

用于后续评估或 RL 时，`--model-path` 指向该 checkpoint 目录。

## 7. GRPO/RL

生成 GRPO 启动脚本。默认生成 portable 脚本，不写死 conda 或服务器绝对路径：

```bash
python scripts/build_skillrl_grpo_command.py \
  --model-path ../models/Qwen3-VL-8B-Instruct \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --output-script output/rl_runs/qwen3vl8b_grpo.sh \
  --portable-output-script \
  --no-compute-node-output-script \
  --cuda-visible-devices 0,1,2,3 \
  --n-gpus-per-node 4 \
  --train-batch-size 16 \
  --val-batch-size 16 \
  --ppo-mini-batch-size 16 \
  --ppo-micro-batch-size-per-gpu 1 \
  --group-size 8 \
  --gradient-accumulation-steps 16 \
  --max-steps 10 \
  --max-prompt-length 16000 \
  --max-response-length 2048 \
  --rollout-max-model-len 24576 \
  --rollout-max-num-batched-tokens 24576 \
  --rollout-gpu-memory-utilization 0.5
```

运行前进入仓库根目录并激活环境：

```bash
cd ..
conda activate skillrl
bash XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh
```

如果必须在计算节点脚本里激活 conda，显式使用 compute 模式：

```bash
python scripts/build_skillrl_grpo_command.py \
  ... \
  --output-script output/rl_runs/qwen3vl8b_grpo_compute.sh \
  --compute-node-output-script \
  --no-portable-output-script \
  --compute-conda-init /path/to/conda.sh \
  --compute-conda-env skillrl \
  --compute-monorepo-root /path/to/XSKILLRL
```

`--compute-monorepo-root` 可以不填；生成脚本会默认从自身位置推断 `XSKILLRL/` 根目录。

## 8. Global Val 统一评估

生成 Base / SFT / RL × with SkillBank / no SkillBank 六组评估脚本：

```bash
python scripts/build_skillrl_eval_suite.py \
  --global-val-json benchmark/_mixed_protocol/global_val.json \
  --base-model-path ../models/Qwen3-VL-8B-Instruct \
  --sft-model-path ../SkillRL/checkpoints/xskill_sft/qwen3vl8b_sft/global_step_x \
  --rl-model-path ../SkillRL/checkpoints/xskill_skillrl/xskill_grpo_skills/hf_checkpoint \
  --skill-bank-json memory_bank/test/skillrl_skill_bank.json \
  --cuda-visible-devices 0 \
  --script-mode portable \
  --max-steps 10 \
  --val-rollout-n 4 \
  --output-dir output/eval_runs
```

如果暂时没有 SFT 或 RL 模型，删掉对应 `--sft-model-path` / `--rl-model-path`，脚本会跳过对应组。

运行：

```bash
cd ..
conda activate skillrl
export BOCHA_API_KEY="your_key"
bash XSkill-dev/output/eval_runs/base_with_skill_val.sh
```

评估输出位于：

```text
SkillRL/checkpoints/xskill_eval/<run_name>/
```

关键文件：

- `experiment_records/*/experiment_record.md`: 指标表和验证历史
- `validation_dump/0.jsonl`: 每条 validation trajectory 的摘要
- `validation_trajectories/0.jsonl`: 仅 `val_only=True` 时保存的逐轮轨迹

查看一条详细轨迹：

```bash
python - <<'PY'
import json

path = "SkillRL/checkpoints/xskill_eval/base_with_skill_val/validation_trajectories/0.jsonl"
with open(path, encoding="utf-8") as f:
    row = json.loads(next(f))

print("source:", row["data_source"])
print("score:", row["score"])
print("tool calls:", row["tool_calling"])
for step in row["trajectory"]:
    print("\nSTEP", step.get("step"))
    print("type:", step.get("parsed_action_type"))
    print("tool:", step.get("tool_name", ""))
    print("response:", step.get("response", "")[:500])
    if step.get("tool_observation"):
        print("observation:", step["tool_observation"][:500])
PY
```

汇总多组评估结果：

```bash
python scripts/summarize_skillrl_eval_suite.py \
  --eval-root ../SkillRL/checkpoints/xskill_eval \
  --output-md output/eval_runs/eval_summary.md
```

## 9. 路径迁移检查

迁移到新服务器后，先检查 parquet 和图片路径：

```bash
python scripts/check_rl_migration_paths.py \
  --train-file output/rl_data/skillrl_train.parquet \
  --val-file output/rl_data/skillrl_val.parquet \
  --image-root ../images \
  --sample-limit 1000
```

如果 parquet 里保留了旧服务器的绝对图片路径，只要图片已迁移到 `../images/<benchmark>/...`，当前路径解析会尝试用路径中的 `images/` 后缀进行重定位。

远程图片建议提前下载到本地：

```bash
python scripts/localize_remote_images.py \
  --input-json benchmark/_mixed_protocol/train_core.json \
  --output-json benchmark/_mixed_protocol/train_core.local_images.json \
  --cache-dir benchmark/_remote_images \
  --path-prefix benchmark/_remote_images \
  --manifest-path output/rl_data/remote_image_manifest.json
```

## 10. 常见环境变量

| 变量 | 用途 |
| --- | --- |
| `XSKILL_REPO_ROOT` | `XSkill-dev` 目录，生成脚本会自动推断 |
| `SKILLRL_REPO_ROOT` | `SkillRL` 目录，生成脚本会自动推断 |
| `XSKILL_IMAGE_ROOT` | 图片根目录，默认优先 `../images` |
| `BOCHA_API_KEY` | `web_search` 和 `image_search` |
| `SEARCH_API_PROVIDER=bocha` | 搜索后端 |
| `IMAGE_SEARCH_PROVIDER=bocha` | 图片搜索后端 |
| `VISIT_BACKEND=local` | 本地网页访问 |
| `VISIT_TIMEOUT` / `VISIT_HARD_TIMEOUT` | visit 工具超时 |
| `EXPERIENCE_EMBEDDING_BACKEND=local` | 本地 embedding |
| `EXPERIENCE_EMBEDDING_MODEL` | 本地 embedding 模型路径 |

## 11. 验证命令

修改脚本后建议先做语法检查：

```bash
python -m py_compile \
  scripts/build_skillrl_grpo_command.py \
  scripts/build_skillrl_eval_suite.py \
  scripts/summarize_accumulation_run.py \
  xskill_rl/skillrl_bridge.py \
  xskill_rl/skillrl/config.py
```

在 Windows 本地如果 `python` 指向商店占位符，可以用：

```powershell
py -3 -m py_compile scripts\build_skillrl_grpo_command.py
```
