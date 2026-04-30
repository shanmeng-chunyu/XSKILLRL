# XSkill + Mixed Training Utilities Usage

这份文档只覆盖当前仓库里已经落地的部分：

- benchmark 规范化
- `train/test` split
- `merged_train/global_val`
- RL 侧 JSONL / Parquet 数据导出

## 1. 准备 benchmark split

以 `MMSearch-Plus` 为例：

```powershell
python scripts/prepare_mmsearch_plus.py `
  --input-json path\\to\\mmsearch_plus.json `
  --output-dir benchmark\\MMSearch-Plus
```

生成：

- `benchmark/MMSearch-Plus/all.json`
- `benchmark/MMSearch-Plus/train.json`
- `benchmark/MMSearch-Plus/test.json`
- `benchmark/MMSearch-Plus/split_manifest_seed42.json`

其它 benchmark 的脚本命名方式一致。

## 2. 构造混合训练池

```powershell
python scripts/prepare_mixed_train_pool.py `
  --train-json benchmark\\MM-BrowseComp\\train.json `
  --train-json benchmark\\TIR-Bench\\train.json `
  --train-json benchmark\\MMSearch-Plus\\train.json `
  --output-dir benchmark\\_mixed_protocol
```

生成：

- `merged_train.json`
- `train_core.json`
- `global_val.json`
- `mixing_manifest.json`

## 3. 导出 RL 数据

导出 JSONL：

```powershell
python scripts/prepare_xskill_rl_dataset.py `
  --input-spec benchmark\\_mixed_protocol\\train_core.json `
  --output-path output\\train_core.jsonl `
  --mixing-strategy concat
```

导出 Parquet：

```powershell
python scripts/prepare_xskill_rl_dataset.py `
  --input-spec benchmark\\_mixed_protocol\\train_core.json `
  --output-path output\\train_core.parquet `
  --mixing-strategy sqrt_size
```

如果导出 Parquet，需要本地环境有 `pyarrow`。

## 4. 结果聚合

```powershell
python scripts/aggregate_benchmark_results.py `
  --input-json results\\mmbrowsecomp_summary.json `
  --input-json results\\tirbench_summary.json
```

输出每个 benchmark 的分数和 macro average。
