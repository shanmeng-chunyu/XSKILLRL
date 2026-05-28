# Tool and Embedding Backends

本文说明当前项目中搜索、网页访问和 experience embedding 的后端选择。当前默认路线是“境内 API + 本地开源组件”，避免依赖 Serper、Google Lens、Jina Reader、ImgBB、OpenAI Embedding 等境外 API。

## 1. Web Search / Image Search: Bocha

`web_search` 和 `image_search` 均使用博查 Web Search API：

- `web_search`: 直接把文本 query 发给博查。
- `image_search` 的文本搜索：把 query 发给博查，并默认补充“图片”关键词。
- `image_search` 的反向搜图：先用本地 OpenAI-compatible VLM 服务为图片生成搜索关键词，再调用博查搜索；不会上传图片到公共图床，也不会调用 Google Lens/Serper。

推荐配置：

```bash
export SEARCH_API_PROVIDER="bocha"
export IMAGE_SEARCH_PROVIDER="bocha"
export BOCHA_API_KEY="你的博查API Key"

# 反向搜图复用本地 reasoning VLM；默认会读取 REASONING_*。
export IMAGE_SEARCH_CAPTION_MODEL="$REASONING_MODEL_NAME"
export IMAGE_SEARCH_CAPTION_API_KEY="$REASONING_API_KEY"
export IMAGE_SEARCH_CAPTION_ENDPOINT="$REASONING_END_POINT"
```

## 2. Visit: Local Open-Source Extraction

`visit` 只使用本地开源方式：

- `requests` 获取网页
- `trafilatura` 抽取正文

推荐配置：

```bash
export VISIT_BACKEND="local"
```

不再启用 Jina Reader fallback。若网页自身在服务器网络中不可访问，`visit` 会返回本地抽取失败，而不会转发给第三方 reader API。

## 3. Experience Embedding: Local Open-Source Model

Experience retrieval 默认使用本地开源 embedding：

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

`XSkill-dev/requirements.txt` 已包含 `sentence-transformers`。第一次运行会从模型源下载 embedding 模型；如果服务器不能访问 Hugging Face，可以提前把模型下载到本地目录，并把 `EXPERIENCE_EMBEDDING_MODEL` 设置为该本地路径。

## 4. Strict Domestic / Local Configuration

严格不使用境外 API 时，训练和测试脚本中保持以下设置：

```bash
export SEARCH_API_PROVIDER="bocha"
export IMAGE_SEARCH_PROVIDER="bocha"
export BOCHA_API_KEY="你的博查API Key"

export VISIT_BACKEND="local"

export EXPERIENCE_EMBEDDING_BACKEND="local"
export EXPERIENCE_EMBEDDING_MODEL="BAAI/bge-m3"

export ENABLED_TOOLS="web_search, image_search, visit, code_interpreter"
```

同时不要设置或依赖以下旧变量：

- `SERPAPI_KEY`
- `JINA_API_KEY`
- `IMGBB_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_API_BASE`

## 5. Source URL Samples

Some benchmark records, especially MMBrowseComp, provide useful `source` URLs even when `images` is empty. The current pipeline appends these URLs into the model prompt and expects the model to call `visit` when it needs page content. The model does not browse directly.

`visit` uses local HTTP extraction (`requests` + `trafilatura`) and may optionally summarize page content through the configured OpenAI-compatible reasoning/evaluator endpoint. Warnings such as "No API key provided, using environment variable REASONING_API_KEY or EVALUATOR_API_KEY" mean the optional summarization client is falling back to environment variables; they are not fatal if the endpoint variables are set correctly.

Google search, Google Maps, and similar anti-bot pages can still return HTTP 403 to local `requests` even when the server has general internet access. Prefer source URLs that expose normal pages or use `web_search` first to find fetchable pages.
