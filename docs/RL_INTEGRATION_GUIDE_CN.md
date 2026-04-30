# RL Integration Guide

当前仓库里保留的是一层轻量工程接口，不直接携带完整 RL trainer。

边界如下：

- `eval/` 仍然是原始 XSkill 推理与 memory 流程
- `scripts/prepare_*.py` 负责把 benchmark 统一整理成训练可用格式
- `scripts/prepare_mixed_train_pool.py` 负责 `merged_train` 和 `global_val`
- `scripts/prepare_xskill_rl_dataset.py` 负责把样本导出成 RL 侧可消费的 JSONL / Parquet
- `xskill_rl/benchmark_protocol.py` 负责 split 规则和 manifest 生成

也就是说，这个仓库现在解决的是：

1. benchmark 协议固定
2. mixed training 数据组织
3. RL 侧输入数据导出

它不直接提供：

1. PPO / GRPO trainer 本体
2. 分布式 rollout
3. 在线 skill bank 更新

如果要接外部 RL 框架，建议把这里导出的数据和 manifest 视为稳定输入层，而不是把训练逻辑塞回 `eval/` 主链路里。
