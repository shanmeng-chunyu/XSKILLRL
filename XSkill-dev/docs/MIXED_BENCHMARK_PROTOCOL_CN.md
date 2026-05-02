# Mixed Benchmark Protocol

本地协议目标只有两个：

1. 给 `XSkill baseline / Pure RL / Hybrid` 提供同一套可复现 split。
2. 在不浪费太多样本的前提下，为训练阶段留一个很小的 `global_val`。

## 正式 split

- 每个 benchmark 只保留 `train/test`
- 固定 `seed=42`
- `train=80%`
- `test=20%`

分层字段约定：

- `VisualToolBench`: `prompt_category`
- `TIR-Bench`: `task`
- `MMSearch-Plus`: `category + difficulty`，不足时退化为 `category`
- `AgentVista`: `domain + subdomain`，不足时退化为 `domain`
- `MMBrowseComp`: `category + level`，不足时退化为 `category`

实现入口：

- `scripts/prepare_visualtoolbench.py`
- `scripts/prepare_tirbench.py`
- `scripts/prepare_mmsearch_plus.py`
- `scripts/prepare_agentvista.py`
- `scripts/prepare_mmbrowsecomp.py`

共享实现位于 `xskill_rl/benchmark_protocol.py`。

## global_val

正式 benchmark 不切单独 `dev`。训练时从合并后的 `merged_train` 再抽 `5%` 作为 `global_val`：

- `merged_train`: 所有 benchmark 的 `train.json` 直接合并
- `train_core`: `merged_train - global_val`
- `global_val`: 只用于 early stopping / checkpoint 选择 / 稳定性比较

构造脚本：

- `scripts/prepare_mixed_train_pool.py`

输出文件：

- `merged_train.json`
- `train_core.json`
- `global_val.json`
- `mixing_manifest.json`

## 混合采样

默认支持两种导出策略：

- `concat`: 直接拼接
- `sqrt_size`: benchmark 级采样权重满足 `p_d ∝ sqrt(n_d)`

导出脚本：

- `scripts/prepare_xskill_rl_dataset.py`

`sqrt_size` 的做法不是在线采样器，而是把较小 benchmark 复制扩展到 `sqrt(n_d * n_max)` 的规模，再统一导出。
