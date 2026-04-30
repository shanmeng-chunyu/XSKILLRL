# AGENTS.md

本文件面向后续接手本仓库的 agent。目标是快速说明项目背景、论文背景、当前工程边界和推荐操作方式。

## 1. 项目定位

本仓库是一个以官方 `XSkill` 代码为基础的研究工程，用于推进 `XSkill + SkillRL` 方向的多模态智能体视觉推理实验。

当前仓库的实际状态：

- 保留了官方 `XSkill` 的推理、工具调用、Experience/Skill memory 构建与检索逻辑。
- 增加了本地 mixed-benchmark 数据准备工具，用于统一多个视觉推理 benchmark 的训练/测试协议。
- 增加了 RL 侧数据导出工具，可导出 JSONL 或 Parquet。
- 还没有内置完整的 PPO/GRPO trainer、分布式 rollout、在线 skill bank 更新或模型参数更新代码。

本项目当前更准确的工程边界是：

```text
XSkill inference + memory framework
+ mixed benchmark split/export utilities
+ RL trainer input preparation
```

而不是：

```text
complete XSkill + SkillRL training system
```

## 2. 研究背景

### 2.1 XSkill

论文：`XSkill: Continual Learning from Experience and Skills in Multimodal Agents`

核心思想：

- 多模态 agent 在不同任务 episode 中积累可复用知识。
- Memory 分为两层：
  - `Experience`: 动作级、局部的经验提示，通常来自具体 trajectory。
  - `Skill`: 任务级、结构化的流程、工具使用模板和注意事项。
- XSkill 是 training-free 方法，不更新基座模型参数。
- 流程分两阶段：
  - `Phase I Accumulation`: 在训练/积累样本上运行 agent，将轨迹蒸馏为 experiences 和 skills。
  - `Phase II Inference`: 在测试样本上检索、改写并注入相关 memory，辅助多步视觉推理。

官方 XSkill 关注的 benchmark 包括：

- `VisualToolBench`
- `TIR-Bench`
- `MMSearch-Plus`
- `AgentVista`
- `MMBrowseComp`

注意：XSkill 虽然不训练参数，但不是完全不使用数据。它会使用 accumulation split 构建 memory，再在 held-out test 上评估。

### 2.2 SkillRL

论文：`SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning`

核心思想：

- 将原始轨迹 memory 压缩成更凝练的 skill bank。
- 训练流程大致是：
  - 初始 memory / SkillBank 生成
  - cold-start SFT
  - skill-augmented GRPO/RL
  - 根据失败样本递归更新 SkillBank
- SkillRL 的重点是让模型在 RL 中学习如何利用 skill，而不是仅靠静态 prompt 注入。

本项目长期目标是借鉴 SkillRL 的思路，让开源 VLM 学会在视觉推理场景中检索和使用合适的 skill。

### 2.3 当前选择的融合方向

前期讨论中选择的是 `策略 B: XSkill as Base`：

- 以 XSkill 的多模态输入、工具执行、memory 机制和 benchmark 处理为基础。
- 将 RL trainer 作为外部训练层接入。
- 优先保持 XSkill benchmark 和 tool-use 逻辑稳定。

当前仓库已经完成的是数据协议和导出层，不包含完整训练闭环。

## 3. 当前目录结构

```text
.
├── assets/                         # README 用图与框架图
├── benchmark/                      # 官方 smoke-test benchmark
│   ├── Tool_Test/
│   └── VisualProbe_Test/
├── docs/                           # 本地工程说明
├── eval/                           # 官方 XSkill 推理、工具、memory 主链路
├── scripts/                        # benchmark 准备、混合训练池、导出与结果聚合
├── xskill_rl/                      # 本地 mixed benchmark 与 RL 数据导出工具库
├── README.md
├── requirements.txt
└── AGENTS.md
```

重要目录说明：

- `eval/infer_api.py`: XSkill 主入口。
- `eval/run_exskill_train.sh`: accumulation 阶段参考脚本，会开启 experience/skill 更新。
- `eval/run_exskill_inference.sh`: inference 阶段参考脚本，只检索并使用已有 memory。
- `eval/tools/`: `web_search`, `image_search`, `visit`, `code_interpreter`, `zoom` 等工具实现。
- `eval/exskill/`: experience 生成、检索、critique、skill 构建逻辑。
- `xskill_rl/benchmark_protocol.py`: split、manifest、global_val、sqrt-size expansion 的共享实现。
- `xskill_rl/dataset.py`: 将标准化样本转成 RL 侧记录格式。

## 4. Benchmark 协议

当前项目采用自定义但固定的 mixed-benchmark 协议。目标不是逐样本复现 XSkill 论文 split，而是在同一协议下公平重跑所有方法。

正式 split：

- 每个 benchmark 只生成 `train/test`。
- 固定 `seed=42`。
- `train=80%`。
- `test=20%`。
- 不为每个 benchmark 单独切 dev。

全局验证集：

- 将所有 benchmark 的 `train.json` 合并为 `merged_train`。
- 从 `merged_train` 中抽取 `5%` 作为 `global_val`。
- `global_val` 只用于 early stopping、checkpoint 选择、reward/训练稳定性比较。
- 最终正式报数时，超参数冻结后应使用完整 `merged_train` 重新训练，再只在固定 `test` 上评估。

分层字段约定：

- `VisualToolBench`: `prompt_category`
- `TIR-Bench`: `task`
- `MMSearch-Plus`: `category + difficulty`，不足时退化为 `category`
- `AgentVista`: `domain + subdomain`，不足时退化为 `domain`
- `MMBrowseComp`: `category + level`，不足时退化为 `category`

过滤规则：

- `VisualToolBench`: 仅保留 `turncase=single-turn` 且 `eval_focus=hybrid_tool_reasoning` 的样本。
- `TIR-Bench`: 仅保留 `refcoco`, `maze`, `instrument`, `ocr`, `contrast` 五类。
- `MMSearch-Plus`: 使用全量输入。
- `AgentVista`: 使用全量输入。
- `MMBrowseComp`: 使用全量输入。

## 5. 常用脚本

### 5.1 生成单个 benchmark split

```powershell
py -3 scripts/prepare_mmsearch_plus.py `
  --input-json path\to\mmsearch_plus.json `
  --output-dir benchmark\MMSearch-Plus
```

其它 benchmark 脚本：

- `scripts/prepare_visualtoolbench.py`
- `scripts/prepare_tirbench.py`
- `scripts/prepare_mmsearch_plus.py`
- `scripts/prepare_agentvista.py`
- `scripts/prepare_mmbrowsecomp.py`

输出文件：

- `all.json`
- `train.json`
- `test.json`
- `split_manifest_seed42.json`

### 5.2 构造混合训练池

```powershell
py -3 scripts/prepare_mixed_train_pool.py `
  --train-json benchmark\MM-BrowseComp\train.json `
  --train-json benchmark\TIR-Bench\train.json `
  --train-json benchmark\MMSearch-Plus\train.json `
  --output-dir benchmark\_mixed_protocol
```

输出文件：

- `merged_train.json`
- `train_core.json`
- `global_val.json`
- `mixing_manifest.json`

### 5.3 导出 RL 数据

导出 JSONL：

```powershell
py -3 scripts/prepare_xskill_rl_dataset.py `
  --input-spec benchmark\_mixed_protocol\train_core.json `
  --output-path output\train_core.jsonl `
  --mixing-strategy concat
```

导出 Parquet：

```powershell
py -3 scripts/prepare_xskill_rl_dataset.py `
  --input-spec benchmark\_mixed_protocol\train_core.json `
  --output-path output\train_core.parquet `
  --mixing-strategy sqrt_size
```

Parquet 导出需要安装 `pyarrow`。

### 5.4 聚合 benchmark 分数

```powershell
py -3 scripts/aggregate_benchmark_results.py `
  --input-json results\mmbrowsecomp_summary.json `
  --input-json results\tirbench_summary.json
```

脚本会输出每个 benchmark 的分数和 macro average。

## 6. XSkill 运行方式

XSkill 使用 OpenAI-compatible API endpoint。

主要环境变量在以下脚本中配置：

- `eval/run_exskill_train.sh`
- `eval/run_exskill_inference.sh`

需要配置的模型和服务：

- `REASONING_MODEL_NAME`, `REASONING_API_KEY`, `REASONING_END_POINT`
- `VERIFIER_MODEL_NAME`, `VERIFIER_API_KEY`, `VERIFIER_END_POINT`
- `EXPERIENCE_MODEL_NAME`, `EXPERIENCE_API_KEY`, `EXPERIENCE_END_POINT`
- `EXPERIENCE_EMBEDDING_MODEL`, `EXPERIENCE_EMBEDDING_API_KEY`, `EXPERIENCE_EMBEDDING_ENDPOINT`

工具相关：

- `SERPAPI_KEY`: web/image search。
- `JINA_API_KEY`: webpage visit。
- `ENABLED_TOOLS`: 默认示例为 `web_search, visit, code_interpreter`。

运行 accumulation：

```bash
bash eval/run_exskill_train.sh
```

运行 inference：

```bash
bash eval/run_exskill_inference.sh
```

运行产物默认写入：

- `memory_bank/`
- `output/`
- `logs/`

这些目录已在 `.gitignore` 中忽略。

## 7. 预期实验对比

建议后续实验至少保持三组方法在同一 split 和同一评测器下比较：

- `XSkill baseline`: 用统一 train pool 做 accumulation，测试时使用 skill/experience memory。
- `Pure RL baseline`: 使用同一训练池训练，但关闭 skill/experience 检索。
- `Hybrid`: 使用同一训练池训练，并显式允许 skill/experience 检索。

报告结果时建议包含：

- 每个 benchmark 的 test 分数。
- 五个 benchmark 的 macro average。
- mixed-train 协议下的主结果表。

注意：如果后续使用 `global_val` 选 checkpoint，最终正式结果应在超参数冻结后使用完整 `merged_train` 重训一次。

## 8. 重要工程边界

当前仓库没有：

- 完整 SkillRL 代码库。
- PPO/GRPO trainer。
- 本地 Qwen-VL 全参训练脚本。
- 分布式 rollout。
- 在线 skill bank 更新。
- VisualToolBench / MMSearch-Plus / AgentVista 的原始数据下载结果。

当前仓库有：

- 官方 XSkill 基础推理代码。
- 两个官方 smoke-test benchmark：`Tool_Test` 和 `VisualProbe_Test`。
- 统一 benchmark split 工具。
- 混合训练池构造工具。
- RL 数据导出工具。

如需接入真实 RL 训练，推荐把本仓库导出的 JSONL/Parquet 和 manifest 作为稳定输入层，在外部 RL 框架中实现训练循环。

## 9. 开发约束

后续 agent 修改本仓库时应遵守：

- 不要提交 `logs/`, `output/`, `memory_bank/`, `benchmark/*/raw/`, `*.parquet`, `*.part`。
- 不要把大型 benchmark 图片、原始 parquet 或下载缓存提交到 git。
- 修改 benchmark 协议时，同步更新 `xskill_rl/benchmark_protocol.py` 和 `docs/MIXED_BENCHMARK_PROTOCOL_CN.md`。
- 修改导出格式时，同步更新 `xskill_rl/dataset.py` 和 `docs/XSKILL_SKILLRL_USAGE_CN.md`。
- 修改 XSkill 推理主链路前，先阅读 `eval/infer_api.py`, `eval/infer_api_utils.py`, `eval/engine/`, `eval/exskill/`。
- 保持样本基础 schema 兼容：
  - `doc_id`
  - `problem`
  - `images`
  - `solution`
  - `benchmark_name`

建议验证命令：

```powershell
py -3 -m py_compile `
  xskill_rl\__init__.py `
  xskill_rl\benchmark_protocol.py `
  xskill_rl\dataset.py `
  scripts\prepare_mmbrowsecomp.py `
  scripts\prepare_tirbench.py `
  scripts\prepare_visualtoolbench.py `
  scripts\prepare_mmsearch_plus.py `
  scripts\prepare_agentvista.py `
  scripts\prepare_mixed_train_pool.py `
  scripts\prepare_xskill_rl_dataset.py `
  scripts\aggregate_benchmark_results.py
```

如果需要推送到 GitHub，当前远端为：

```text
https://github.com/shanmeng-chunyu/XSKILLRL.git
```

本机此前 `git push` 需要临时代理和 OpenSSL 后端：

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl push
```

## 10. 后续优先任务

建议按以下顺序推进：

1. 下载并规范化 `VisualToolBench`, `MMSearch-Plus`, `AgentVista`。
2. 为五个 benchmark 生成固定 `train/test` 和 `split_manifest_seed42.json`。
3. 构造 `benchmark/_mixed_protocol` 下的 `merged_train`, `train_core`, `global_val`。
4. 导出 RL 侧 JSONL/Parquet。
5. 在外部 RL 框架中接入 cold-start SFT 和 GRPO/PPO。
6. 回到本仓库统一跑 XSkill baseline、Pure RL baseline、Hybrid 的 test 评测。
