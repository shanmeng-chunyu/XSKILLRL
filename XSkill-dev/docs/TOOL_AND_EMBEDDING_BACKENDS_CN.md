# Tool and Embedding Backends

本文说明当前项目中搜索、网页访问和 experience embedding 的后端选择。

## 1. Web Search: Bocha

`web_search` 现在支持两种 provider：

- `bocha`: 博查 Web Search API，使用 `BOCHA_API_KEY`。
- `serper`: 旧的 Serper.dev API，使用 `SERPAPI_KEY`。

推荐配置：

```bash
export SEARCH_API_PROVIDER="bocha"
export BOCHA_API_KEY="你的博查API Key"
```

如果需要回退旧 Serper：

```bash
export SEARCH_API_PROVIDER="serper"
export SERPAPI_KEY="你的Serper Key"
```

注意：`image_search` 仍使用 Serper.dev 的 image/lens API。如果不配置
`SERPAPI_KEY`，不要在 `ENABLED_TOOLS` 中启用 `image_search`。

## 2. Visit: Local Open-Source Extraction

`visit` 默认使用本地开源方式：

- `requests` 获取网页
- `trafilatura` 抽取正文

推荐配置：

```bash
export VISIT_BACKEND="local"
```

如果希望本地抽取失败后再用 Jina Reader：

```bash
export VISIT_BACKEND="auto"
export JINA_API_KEY="你的Jina Key"
```

如果强制只用 Jina：

```bash
export VISIT_BACKEND="jina"
export JINA_API_KEY="你的Jina Key"
```

## 3. Experience Embedding: Local Open-Source Model

Experience retrieval 现在支持：

- `local`: 使用 `sentence-transformers` 本地加载开源 embedding 模型。
- `api`: 使用 OpenAI-compatible `/embeddings` API。

推荐本地配置：

```bash
export EXPERIENCE_EMBEDDING_BACKEND="local"
export EXPERIENCE_EMBEDDING_MODEL="BAAI/bge-m3"
export EXPERIENCE_EMBEDDING_DEVICE="cuda"
export EXPERIENCE_EMBEDDING_API_KEY=""
export EXPERIENCE_EMBEDDING_ENDPOINT=""
```

如果显存紧张，可以把模型换成更小的：

```bash
export EXPERIENCE_EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
```

如果要使用 API embedding：

```bash
export EXPERIENCE_EMBEDDING_BACKEND="api"
export EXPERIENCE_EMBEDDING_MODEL="text-embedding-3-small"
export EXPERIENCE_EMBEDDING_API_KEY="你的API Key"
export EXPERIENCE_EMBEDDING_ENDPOINT="https://api.openai.com/v1"
```

`XSkill-dev/requirements.txt` 已包含 `sentence-transformers`，因此本地 embedding
不需要额外商业 API。
