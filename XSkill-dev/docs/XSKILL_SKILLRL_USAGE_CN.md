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

## 5. SkillRL/GRPO 兼容导出

把 XSkill `SKILL.md` 转成 SkillRL SkillBank JSON：

```powershell
python scripts/convert_xskill_skill_to_skillrl_bank.py `
  --input-skill-md memory_bank\\test\\SKILL.md `
  --output-json memory_bank\\test\\skillrl_skill_bank.json
```

导出 SkillRL/verl 可读的 GRPO Parquet：

```powershell
python scripts/prepare_skillrl_grpo_dataset.py `
  --input-spec benchmark\\_mixed_protocol\\train_core.json `
  --output-path output\\skillrl_train.parquet `
  --skill-bank-json memory_bank\\test\\skillrl_skill_bank.json `
  --skill-retrieval-mode template `
  --top-k 6
```

生成 SkillRL 风格 GRPO 启动命令：

```powershell
python scripts/build_skillrl_grpo_command.py `
  --model-path path\\to\\sft_model `
  --train-file output\\skillrl_train.parquet `
  --val-file output\\skillrl_val.parquet `
  --skill-bank-json memory_bank\\test\\skillrl_skill_bank.json
```

更完整说明见 `docs/SKILLRL_GRPO_INTEGRATION_CN.md`。

## 6. 搜索、网页访问和 embedding 后端

当前工具后端支持：

- `web_search`: 可用博查 API，配置 `SEARCH_API_PROVIDER=bocha` 和 `BOCHA_API_KEY`。
- `visit`: 默认本地 `requests + trafilatura`，配置 `VISIT_BACKEND=local`。
- experience embedding: 默认本地开源 `sentence-transformers`，配置 `EXPERIENCE_EMBEDDING_BACKEND=local`。

详细配置见 `docs/TOOL_AND_EMBEDDING_BACKENDS_CN.md`。
