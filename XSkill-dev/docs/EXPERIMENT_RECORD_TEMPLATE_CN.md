# XSkill + SkillRL 实验记录模板

本文档用于记录 XSkill accumulation、SkillRL/GRPO 训练、验证与消融实验中的关键信息。建议每次正式实验复制一份本文档，按实际运行情况填写，避免只依赖日志文件追溯实验。

## 1. 实验基本信息

| 字段 | 记录 |
|---|---|
| 实验编号 |  |
| 实验名称 |  |
| 负责人 |  |
| 开始时间 |  |
| 结束时间 |  |
| Git commit |  |
| 代码分支 |  |
| 服务器 / 节点 |  |
| GPU 型号与数量 |  |
| CUDA / Driver 版本 |  |
| 备注 |  |

## 2. 实验目标

记录本次实验主要回答的问题，例如：

- 验证 XSkill accumulation 是否提升后续测试准确率。
- 验证 SkillBank 注入对 GRPO 训练的影响。
- 比较不同 `MAX_COMPLETION_TOKENS`、`MAX_TOTAL_TOKENS`、endpoint 数量、rollout 数量对准确率和吞吐的影响。
- 比较纯模型、XSkill memory、SkillRL/GRPO、Hybrid 方法之间的差异。

本次实验目标：

```text

```

## 3. 环境记录

### 3.1 Conda / Python 环境

| 环境 | 用途 | Python | 关键包版本 | 备注 |
|---|---|---:|---|---|
| xskill | accumulation / inference |  | `torch=`, `transformers=`, `sentence-transformers=`, `huggingface-hub<1.0`, `qwen-vl-utils=` | 运行 `eval/run_exskill_train.sh`、`eval/run_exskill_inference.sh` |
| vllm-serve | OpenAI-compatible 推理服务 |  | `vllm=`, `transformers=` | 可与训练环境分离；Qwen3-VL 服务需确认 tool parser 可用 |
| skillrl | GRPO / verl 训练 | Python 3.10（当前服务器检查） | `torch=`, `vllm=0.11.0 推荐`, `ray=`, `transformers=` | 不建议使用 `vllm==0.21.0`，当前 SkillRL/verl 代码依赖旧 vLLM API |

建议记录命令输出：

```bash
python -V
python -m pip freeze | grep -E "torch|vllm|transformers|ray|flash|sentence|huggingface|qwen"
nvidia-smi
```

### 3.2 vLLM 服务配置

| Endpoint | GPU | 模型路径 | 端口 | max-model-len | max-num-seqs | gpu-memory-utilization | dtype | 备注 |
|---|---:|---|---:|---:|---:|---:|---|---|
| `http://127.0.0.1:8002/v1/chat/completions` |  | `/sata/luzy/models/Qwen3-VL-8B-Instruct` 或 `Qwen/Qwen3-VL-8B-Instruct` | 8002 | 32768 / 65536 | 2-4 | 0.7-0.8 | bfloat16 | 当前 XSkill 默认 endpoint pool 第 1 个 |
| `http://127.0.0.1:8003/v1/chat/completions` |  | 同上 | 8003 | 32768 / 65536 | 2-4 | 0.7-0.8 | bfloat16 | 当前 XSkill 默认 endpoint pool 第 2 个 |

启动命令：

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve /sata/luzy/models/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host 0.0.0.0 \
  --port 8002 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image":16}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

吞吐诊断结果：

| Endpoint | 并发 | 样本数 | P50 latency | P95 latency | tokens/s | 失败率 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |

## 4. 数据与 Benchmark

### 4.1 数据来源与 Split

| Benchmark | 原始样本数 | 过滤后样本数 | Train | Global Val | Test | 图片是否完整 | 备注 |
|---|---:|---:|---:|---:|---:|---|---|
| VisualToolBench |  |  | 246 train_core（本地当前文件） |  |  |  |  |
| TIR-Bench |  |  | 327 train_core（本地当前文件） |  |  |  |  |
| MMSearch-Plus |  |  | 236 train_core（本地当前文件） |  |  |  |  |
| AgentVista |  |  | 159 train_core（本地当前文件） |  |  |  |  |
| MMBrowseComp |  |  | 305 train_core（本地当前文件） |  |  |  |  |
| Mixed |  | 1340 merged train（本地当前文件） | 1273 train_core（本地当前文件） | 67（由 1340-1273 推得，若文件存在请核对） |  |  | `benchmark/_mixed_protocol` |

关键文件：

| 文件 | 路径 | SHA256 / 版本 | 备注 |
|---|---|---|---|
| merged_train | `benchmark/_mixed_protocol/merged_train.json` |  | 本地当前统计 1340 条 |
| train_core | `benchmark/_mixed_protocol/train_core.json` |  | XSkill accumulation 默认输入；本地当前统计 1273 条 |
| global_val | `benchmark/_mixed_protocol/global_val.json` |  | 若存在则作为验证/选 checkpoint 使用 |
| test split |  |  |  |
| SkillRL train parquet | `output/rl_data/skillrl_train.parquet` |  | 从 `XSkill-dev` 目录看是相对路径；从 `SkillRL` 目录运行时建议改绝对路径 |
| SkillRL val parquet | `output/rl_data/skillrl_val.parquet` |  | 从 `XSkill-dev` 目录看是相对路径；从 `SkillRL` 目录运行时建议改绝对路径 |

### 4.2 数据异常记录

| 问题类型 | 数量 | 代表样本 | 是否影响实验 | 处理方式 |
|---|---:|---|---|---|
| `images=[]` 的文本型样本 | 454（当前 `train_core.json`） | MMBrowseComp/TIR-Bench 部分样本 | 通常不影响；表示文本/网页搜索型任务 | 记录即可；若 problem 含 `<image>` 才需要修数据 |
| `<image>` 存在但图片缺失 | 0（当前 `train_core.json`） |  | 会严重影响视觉任务 | 需要补图或过滤样本 |
| doc_id 重复 | 80 个重复 ID / 160 条样本（当前 `train_core.json`） | 数字 ID 跨 benchmark 重复 | 可能导致输出目录覆盖/混淆 | 输出目录应使用 benchmark 前缀或唯一 sample id |
| 远程图片不可访问 |  |  |  |  |
| 空 problem / 空 answer | 空 problem: 327；空 answer: 1（当前 `train_core.json`） | TIR-Bench 部分样本可能字段结构不同 | 需要区分真实空字段和字段名不一致 | 抽查原始样本 schema |

## 5. XSkill Accumulation 记录

### 5.1 运行配置

| 参数 | 值 |
|---|---|
| `EXP_NAME` | `qwen3vl8b_mixed_train_core_seed42` |
| `DATA_PATH` | `benchmark/_mixed_protocol/train_core.json` |
| `IMAGE_DIR` | `benchmark` |
| `OUTPUT_DIR` | `output/xskill_accum/${EXP_NAME}` |
| `LOG_OUTPUT_DIR` | `logs/xskill_accum/${EXP_NAME}` |
| `LOCAL_VLM_ENDPOINTS` | `http://127.0.0.1:8002/v1/chat/completions,http://127.0.0.1:8003/v1/chat/completions` |
| `NUM_WORKERS` | `8` |
| `ROLLOUTS_PER_SAMPLE` | `2` |
| `MAX_TURNS` | `20` |
| `MAX_TOTAL_TOKENS` | `65536` |
| `MAX_COMPLETION_TOKENS` | `12288` |
| `MAX_IMAGES` | `100` |
| `TEMPERATURE` / `TOP_P` | `0.6` / `1.0` |
| `EXPERIENCE_LARGE_BATCH` | `32` |
| `EXPERIENCE_MAX_OPS` | `3` |
| `EXPERIENCE_MAX_ITEMS` | `120` |
| `EXPERIENCE_RETRIEVAL_TOP_K` | `3` |
| `SKILL_MAX_LENGTH` | `1000` |
| `ENABLED_TOOLS` | `web_search, image_search, visit, code_interpreter, zoom` |
| `IMAGE_SEARCH_MAX_CALLS` | `0` |
| `WEB_SEARCH_MAX_CALLS` | `3` |
| `SEARCH_API_PROVIDER` | `bocha` |
| `IMAGE_SEARCH_PROVIDER` | `bocha` |
| `VISIT_BACKEND` | `local` |
| `EXPERIENCE_EMBEDDING_BACKEND` | `local` |
| `EXPERIENCE_EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `EXPERIENCE_EMBEDDING_DEVICE` | `cuda` |

完整启动命令：

```bash
cd /sata/luzy/XSKILLRL/XSkill-dev
LOCAL_VLM_ENDPOINTS="http://127.0.0.1:8002/v1/chat/completions,http://127.0.0.1:8003/v1/chat/completions" \
BOCHA_API_KEY="..." \
bash eval/run_exskill_train.sh
```

### 5.2 Accumulation 总体结果

| 指标 | 值 |
|---|---:|
| 处理样本数 | 1193 |
| 成功样本数 | 50 |
| 失败样本数 | 80 |
| 总 rollout 数 | 2386 |
| 平均每样本耗时 |  |
| 总耗时 |  |
| pass@1 | 0.0285 |
| average@1 | 0.0285 |
| pass@2 / pass@k | 0.0419 |
| average@2 / average@k | 0.0281 |
| 生成 experience 数 | 1775 |
| 最终 experience library 数 |  |
| 最终 skill 字数 |  |

### 5.3 Accumulation 分 Benchmark 结果

| Benchmark | 样本数 | pass@1 | average@1 | pass@k | average@k | 平均耗时 | 主要失败原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VisualToolBench | 246 | 0.0122 | 0.0122 | 0.0325 | 0.0203 |  |  |
| TIR-Bench | 247 | 0.0081 | 0.0081 | 0.0121 | 0.0081 |  |  |
| MMSearch-Plus | 236 | 0.0042 | 0.0042 | 0.0042 | 0.0042 |  |  |
| AgentVista | 159 | 0.1069 | 0.1069 | 0.1447 | 0.0975 |  |  |
| MMBrowseComp | 305 | 0.0361 | 0.0361 | 0.0492 | 0.0328 |  |  |

### 5.4 失败类型统计

| 失败类型 | 数量 | 占比 | 代表样本 | 备注 |
| --- | --- | --- | --- | --- |
| `Error: Could not parse model response` | 1185 | 0.4966 | 6883f999d3f384b2a9a3e61b, 6883f999d3f384b2a9a3e61b, 212, 212, 954 |  |
| `Error: Reached max token limit` | 0 | 0.0000 |  |  |
| 输出中途截断 | 0 | 0.0000 |  |  |
| API timeout / 连接失败 | 1016 | 0.4258 | 1102, 68757ac94c044bac23c9bdcc, 68757ac94c044bac23c9bdcc, 68757ac94c044bac23c9bdd9, 68757ac94c044bac23c9bd93 |  |
| 工具调用失败 | 2 | 0.0008 | mmsearch_plus_000011, mmsearch_plus_000217 |  |
| 图片找不到 | 381 | 0.1597 |  |  |
| verifier 判分异常 | 29 | 0.0122 |  |  |
| experience embedding 失败 | 0 | 0.0000 |  |  |
| retrieval 无结果 | 16 | 0.0067 |  |  |

### 5.5 API Timing 与吞吐

| 请求类型 | 请求数 | 平均耗时 | P50 | P95 | 失败数 | 平均 prompt tokens | 平均 completion tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reasoning | 29481 |  |  |  | 25 | 6245.8 | 205.0 |
| verifier | 2572 |  |  |  | 0 | 685.6 | 108.8 |
| experience | 9014 |  |  |  | 145 | 5241.6 | 942.1 |
| image caption |  |  |  |  |  |  |  |
| web_search |  |  |  |  |  |  |  |
| image_search |  |  |  |  |  |  |  |
| visit |  |  |  |  |  |  |  |
| code_interpreter / zoom |  |  |  |  |  |  |  |

Endpoint 使用情况：

| Endpoint | 请求数 | 平均耗时 | 失败数 | unhealthy 次数 | 备注 |
| --- | --- | --- | --- | --- | --- |
| http://127.0.0.1:8002/v1/chat/completions | 19356 |  | 71 | 0 |  |
| http://127.0.0.1:8003/v1/chat/completions | 21711 |  | 99 | 0 |  |

## 6. Memory Bank 记录

| 文件 | 路径 | 数量 / 大小 | 备注 |
|---|---|---:|---|
| experiences.json | `memory_bank/test/experiences.json` |  | XSkill accumulation 输出；填充脚本会优先读 `memory_bank/xskill_accum/${EXP_NAME}`，不存在时回退到 `memory_bank/test` |
| SKILL.md | `memory_bank/test/SKILL.md` |  | XSkill accumulation 输出；填充脚本会优先读 `memory_bank/xskill_accum/${EXP_NAME}`，不存在时回退到 `memory_bank/test` |
| skillrl_skill_bank.json | `memory_bank/xskill_accum/qwen3vl8b_mixed_train_core_seed42/skillrl_skill_bank.json` | 12.6 KB | 给 SkillRL/GRPO 使用 |
| embedding cache / local model | `BAAI/bge-m3` 或本地模型目录 |  | 离线服务器建议写本地绝对路径 |

质量抽查：

| 抽查项 | 结果 | 备注 |
|---|---|---|
| experience 是否包含具体视觉线索 |  |  |
| skill 是否过长或重复 |  |  |
| retrieval top-k 是否与任务相关 |  |  |
| 是否存在空 embedding 或 fallback experience |  |  |

## 7. SkillRL / GRPO 训练记录

### 7.1 训练配置

| 参数 | 值 |
|---|---|
| 训练脚本 | `../XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh` |
| `actor_rollout_ref.model.path` | `Qwen/Qwen3-VL-8B-Instruct`（也可改为本地模型绝对路径） |
| `data.train_files` | 推荐 `/sata/luzy/XSKILLRL/XSkill-dev/output/rl_data/skillrl_train.parquet` |
| `data.val_files` | 推荐 `/sata/luzy/XSKILLRL/XSkill-dev/output/rl_data/skillrl_val.parquet` |
| `data.train_batch_size` | `64` |
| `data.val_batch_size` | `64` |
| `data.max_prompt_length` | `6000` |
| `data.max_response_length` | `1024` |
| `data.filter_overlong_prompts` | `True` |
| `data.truncation` | `left` |
| `data.return_raw_chat` | `True` |
| `algorithm.adv_estimator` | `grpo` |
| `env.env_name` | `xskill_visual` |
| `env.rollout.n` | `4`（当前运行脚本观测值；配置生成器默认 `8`） |
| `env.max_steps` | `1` |
| `env.xskill.reward_mode` | `contains` |
| `env.xskill.repo_root` | `/sata/luzy/XSKILLRL/XSkill-dev` |
| `env.xskill.image_root` | 推荐 `/sata/luzy/XSKILLRL/XSkill-dev/benchmark` |
| `trainer.n_gpus_per_node` | `4`（当前运行脚本观测值；配置生成器默认 `8`） |
| `trainer.nnodes` | `1` |
| `trainer.total_epochs` | `150` |
| `trainer.save_freq` | `10` |
| `trainer.test_freq` | `5` |
| `trainer.project_name` | `xskill_skillrl` |
| `trainer.experiment_name` | `xskill_grpo_skills` |
| actor lr | `1e-6` |
| actor batch | `ppo_mini_batch_size=64`, `ppo_micro_batch_size_per_gpu=4` |
| rollout engine | `vllm`, `tensor_model_parallel_size=1`, `gpu_memory_utilization=0.7` |
| ref logprob | `log_prob_micro_batch_size_per_gpu=8`, `ref.fsdp_config.param_offload=True` |
| KL 配置 | `actor.use_kl_loss=True`, `kl_loss_coef=0.01`, `kl_loss_type=low_var_kl`, `algorithm.use_kl_in_reward=False` |
| invalid action penalty | `use_invalid_action_penalty=True`, `invalid_action_penalty_coef=0.1` |
| SkillBank 路径 | 推荐 `/sata/luzy/XSKILLRL/XSkill-dev/memory_bank/xskill_accum/qwen3vl8b_mixed_train_core_seed42/skillrl_skill_bank.json` |
| Skill memory 配置 | `retrieval_mode=template`, `top_k=6`, `enable_dynamic_update=False`, `update_threshold=0.4`, `max_new_skills=3` |

完整启动命令：

```bash
cd /sata/luzy/XSKILLRL/SkillRL
ray stop -f
HYDRA_FULL_ERROR=1 bash ../XSkill-dev/output/rl_runs/qwen3vl8b_grpo.sh
```

### 7.2 训练曲线记录

| Epoch / Step | reward mean | reward std | success rate | response length | KL | entropy | actor loss | grad norm | GPU memory | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |  |  |  |

### 7.3 Checkpoint 记录

| Checkpoint | Step / Epoch | Val score | 选择原因 | 路径 | 备注 |
|---|---:|---:|---|---|---|
|  |  |  |  |  |  |

### 7.4 训练异常记录

| 时间 | 异常 | 影响 | 处理方式 | 是否需要重跑 |
|---|---|---|---|---|
|  |  |  |  |  |

## 8. 验证 / 测试记录

### 8.1 Inference 配置

| 参数 | 值 |
|---|---|
| 模型 / checkpoint | `REASONING_MODEL_NAME`，默认需手动填写；通常为 `qwen3-vl-8b` 或训练后 checkpoint |
| 测试数据 | `benchmark/VisualProbe_Test/val.json`（当前 `run_exskill_inference.sh` smoke-test 默认） |
| 图片目录 | `benchmark` |
| 输出目录 | `output/test_exskill_1` |
| 日志目录 | `logs/test_exskill_1.log` |
| `MAX_TOTAL_TOKENS` | `32768` |
| `MAX_COMPLETION_TOKENS` | `8192`（可通过环境变量覆盖） |
| `MAX_TURNS` | `20` |
| `MAX_IMAGES` | `100` |
| `NUM_WORKERS` | `8` |
| `ROLLOUTS_PER_SAMPLE` | `2` |
| `IMAGE_SEARCH_MAX_CALLS` | `5` |
| `WEB_SEARCH_MAX_CALLS` | `7` |
| 是否启用 skill | 是：`--skill-enable --skill-inference` |
| 是否启用 experience | 是：`--experience-enable --experience-retrieval` |
| retrieval top-k | `3` |
| retrieval decomposition/rewrite | 是：`--experience-retrieval-decomposition --experience-retrieval-rewrite` |
| verifier / judge 模型 | `VERIFIER_MODEL_NAME`，默认需手动填写；可与 reasoning endpoint 相同 |

完整命令：

```bash
cd /sata/luzy/XSKILLRL/XSkill-dev
MAX_COMPLETION_TOKENS=8192 bash eval/run_exskill_inference.sh
```

### 8.2 测试结果

| Benchmark | 样本数 | Accuracy | pass@1 | average@1 | pass@k | average@k | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| VisualToolBench |  |  |  |  |  |  |  |
| TIR-Bench |  |  |  |  |  |  |  |
| MMSearch-Plus |  |  |  |  |  |  |  |
| AgentVista |  |  |  |  |  |  |  |
| MMBrowseComp |  |  |  |  |  |  |  |
| Macro Average |  |  |  |  |  |  |  |

### 8.3 Case Study

| 样本 ID | Benchmark | 预测 | 标准答案 | 是否正确 | 失败/成功原因 | 备注 |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

重点记录：

- 输出是否被长度截断。
- 是否正确调用工具。
- retrieval 的 experience/skill 是否相关。
- 错误是模型知识不足、视觉识别失败、搜索失败、工具失败，还是 verifier 判分问题。

## 9. 消融实验矩阵

| 实验编号 | 方法 | Skill | Experience | Retrieval | GRPO | Rollout n | Token 配置 | Endpoint 数 | 结果 | 备注 |
|---|---|---|---|---|---|---:|---|---:|---:|---|
| A0 | Base VLM | off | off | off | off | 2 | train: `65536/12288` 或 test: `32768/8192` | 1-2 |  | 去掉 `--skill-*` 和 `--experience-*` |
| A1 | XSkill skill only | on | off | on/off | off | 2 | 同上 | 1-2 |  | 保留 skill，关闭 experience |
| A2 | XSkill experience only | off | on | on | off | 2 | 同上 | 1-2 |  | 关闭 skill，保留 experience retrieval |
| A3 | XSkill full memory | on | on | on | off | 2 | 同上 | 1-2 |  | 当前 XSkill 默认主设置 |
| A4 | SkillRL GRPO no memory | off | off | off | on | 4 | `max_prompt=6000`, `max_response=1024` | 训练 GPU=4 |  | GRPO 环境关闭 SkillBank |
| A5 | SkillRL GRPO + SkillBank | on | off | on | on | 4 | `max_prompt=6000`, `max_response=1024` | 训练 GPU=4 |  | 当前 GRPO 脚本主设置 |
| A6 | Hybrid full | on | on | on | on | 4 | 训练后再按 XSkill inference token 设置测试 | 训练 GPU=4 / 推理 endpoint=1-2 |  | GRPO checkpoint + XSkill memory |

## 10. 结论记录

### 10.1 主要发现

```text

```

### 10.2 当前最佳配置

```text

```

### 10.3 剩余问题

```text

```

### 10.4 下一步计划

```text

```

## 11. 复现实验 Checklist

- [ ] Git commit 已记录。
- [ ] 所有数据 split 路径和版本已记录。
- [ ] vLLM 启动命令已记录。
- [ ] Conda 环境和关键包版本已记录。
- [ ] XSkill accumulation 命令已记录。
- [ ] SkillRL/GRPO 训练命令已记录。
- [ ] checkpoint 路径已记录。
- [ ] 测试命令和结果路径已记录。
- [ ] 失败样本和异常日志已归档。
- [ ] 最终结果表格已与日志核对。
