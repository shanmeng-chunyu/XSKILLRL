"""
ImageSearch tool using China-accessible backends.

Text image search uses Bocha Web Search. Reverse image search is implemented as
"local VLM caption/keywords -> Bocha search" so it does not upload images to
foreign image hosting services or call external reverse-image APIs.
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from tools.base import BaseTool
from tools.tool_registry import register_tool
from utils.context_utils import pil_to_base64_data_uri


@register_tool("image_search")
class ImageSearch(BaseTool):
    name = "image_search"
    description = """
Search for related images using text query or reverse image search.
- For text-to-image search: specify search_type="text" and provide a query.
- For reverse image search: specify search_type="reverse" and provide image_url.

Reverse image search does not call an external reverse-image API. It first asks
the local reasoning VLM to describe the image as search keywords, then searches
Bocha with those keywords.

Image references for reverse search:
- "original_image" or "original_image_N"
- "tool_image_N"
- "observation_N"
- a local image path
- an http(s) image URL
"""
    parameters = {
        "type": "object",
        "properties": {
            "search_type": {
                "type": "string",
                "enum": ["text", "reverse"],
                "description": "Type of search: 'text' or 'reverse'",
                "default": "text",
            },
            "query": {
                "type": "string",
                "description": "Search query string. Required for text search and optional for reverse search.",
            },
            "image_url": {
                "type": "string",
                "description": "Image reference or URL for reverse search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, config=None):
        super().__init__(config)
        config = config or {}

        self.provider = (
            config.get("provider")
            or os.getenv("IMAGE_SEARCH_PROVIDER")
            or "bocha"
        ).lower()
        if self.provider != "bocha":
            raise ValueError(
                "Only IMAGE_SEARCH_PROVIDER=bocha is supported for the domestic image search backend."
            )

        self.api_key = config.get("api_key") or os.getenv("BOCHA_API_KEY")
        if not self.api_key:
            raise ValueError("BOCHA_API_KEY is required for image_search.")

        self.api_endpoint = (
            config.get("api_endpoint")
            or os.getenv("BOCHA_SEARCH_ENDPOINT")
            or "https://api.bochaai.com/v1/web-search"
        )
        self.max_results_default = int(config.get("max_results", 10))
        self.search_type_default = config.get("search_type", "text")
        self.timeout = int(config.get("timeout", 30))
        self.image_query_suffix = (
            config.get("image_query_suffix")
            or os.getenv("BOCHA_IMAGE_SEARCH_SUFFIX")
            or " 图片"
        )

        self.caption_model = (
            config.get("caption_model")
            or os.getenv("IMAGE_SEARCH_CAPTION_MODEL")
            or os.getenv("REASONING_MODEL_NAME")
        )
        self.caption_api_key = (
            config.get("caption_api_key")
            or os.getenv("IMAGE_SEARCH_CAPTION_API_KEY")
            or os.getenv("REASONING_API_KEY")
            or "EMPTY"
        )
        self.caption_endpoint = (
            config.get("caption_endpoint")
            or os.getenv("IMAGE_SEARCH_CAPTION_ENDPOINT")
            or os.getenv("REASONING_END_POINT")
        )
        self.caption_endpoint = self._normalize_chat_endpoint(self.caption_endpoint)
        self.caption_timeout = int(config.get("caption_timeout", 120))

        self.proxies = None
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            self.proxies = {
                "http": http_proxy,
                "https": https_proxy or http_proxy,
            }
            print(f"[ImageSearch] Using proxy: {self.proxies}")

    def _normalize_chat_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return endpoint
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def call(self, params, **kwargs):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"query": params, "search_type": "text"}

        search_type = str(params.get("search_type", self.search_type_default)).lower()
        max_results = int(params.get("max_results", self.max_results_default))

        if search_type in ("image", "reverse"):
            return self._reverse_image_search(
                params,
                max_results=max_results,
                image_map=kwargs.get("image_map") or {},
                save_dir=kwargs.get("save_dir"),
            )
        if search_type == "text":
            query = params.get("query", "")
            if not query:
                return "Error: No query provided for text image search"
            return self._bocha_image_search(query, max_results=max_results)

        return f"Error: Invalid search_type '{search_type}'. Must be 'text' or 'reverse'"

    def _reverse_image_search(
        self,
        params: Dict[str, Any],
        *,
        max_results: int,
        image_map: Dict[str, Image.Image],
        save_dir: Optional[str],
    ) -> str:
        image_ref = params.get("image_url", "")
        if not image_ref:
            return "Error: No image_url provided for reverse image search"

        image = self._resolve_image(image_ref, image_map=image_map, save_dir=save_dir)
        if image is None:
            return (
                f"Error: Could not resolve image reference '{image_ref}'. "
                "Use original_image, tool_image_N, observation_N, a local path, or an image URL."
            )

        caption = self._caption_image_for_search(image)
        if not caption:
            fallback_query = params.get("query")
            if not fallback_query:
                return "Error: Failed to caption image for domestic reverse image search"
            caption = fallback_query

        query_hint = params.get("query")
        search_query = f"{query_hint}; {caption}" if query_hint else caption
        result = self._bocha_image_search(search_query, max_results=max_results)
        return (
            "Reverse image search was performed via local VLM caption + Bocha search.\n\n"
            f"Image search keywords: {caption}\n\n"
            f"{result}"
        )

    def _resolve_image(
        self,
        image_ref: str,
        *,
        image_map: Dict[str, Image.Image],
        save_dir: Optional[str],
    ) -> Optional[Image.Image]:
        if image_ref in image_map:
            return image_map[image_ref]

        candidates = []
        ref_path = Path(image_ref)
        candidates.append(ref_path)

        if save_dir:
            save_root = Path(save_dir)
            candidates.append(save_root / image_ref)
            if ref_path.suffix == "":
                for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    candidates.append(save_root / f"{image_ref}{suffix}")

        for candidate in candidates:
            if candidate.is_file():
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception as exc:
                    print(f"[ImageSearch] Failed to open local image {candidate}: {exc}")

        if image_ref.startswith(("http://", "https://")):
            try:
                response = requests.get(image_ref, timeout=self.timeout, proxies=self.proxies)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as exc:
                print(f"[ImageSearch] Failed to download image URL {image_ref}: {exc}")
                return None

        if image_ref.startswith("data:image/"):
            try:
                _, payload = image_ref.split(",", 1)
                return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")
            except Exception as exc:
                print(f"[ImageSearch] Failed to parse data URI image: {exc}")
                return None

        return None

    def _caption_image_for_search(self, image: Image.Image) -> Optional[str]:
        if not self.caption_endpoint or not self.caption_model:
            print("[ImageSearch] Caption endpoint/model not configured")
            return None

        prompt = (
            "请为这张图片生成适合中文网页和图片搜索的关键词。"
            "包含主体、品牌/地标/人物/文字、颜色、场景、显著细节。"
            "如果不确定，不要编造。只输出一行关键词，中英文都可以。"
        )
        payload = {
            "model": self.caption_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": pil_to_base64_data_uri(image)}},
                    ],
                }
            ],
            "temperature": 0.1,
            "top_p": 1.0,
            "max_tokens": 512,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.caption_api_key}",
        }

        try:
            response = requests.post(
                self.caption_endpoint,
                headers=headers,
                json=payload,
                timeout=self.caption_timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            content = str(content).strip()
            return content or None
        except Exception as exc:
            print(f"[ImageSearch] Local VLM caption failed: {exc}")
            return None

    def _bocha_image_search(self, query: str, *, max_results: int) -> str:
        search_query = self._make_image_query(query)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": search_query,
            "freshness": os.getenv("BOCHA_SEARCH_FRESHNESS", "noLimit"),
            "summary": os.getenv("BOCHA_SEARCH_SUMMARY", "true").lower() not in ("0", "false", "no"),
            "count": min(max_results, 50),
        }

        try:
            print(f"[ImageSearch:bocha] Searching images/pages for: {search_query}")
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                proxies=self.proxies,
            )
            response.raise_for_status()
            result_data = response.json()
        except Exception as exc:
            return f"Error: Bocha image search failed: {exc}"

        image_results = self._extract_image_results(result_data)
        if not image_results:
            image_results = self._extract_webpage_results(result_data)

        image_results = image_results[:max_results]
        if not image_results:
            return f"No image or webpage results found for query: '{search_query}'"

        formatted = []
        for item in image_results:
            image_url = item.get("image_url") or "N/A"
            title = item.get("title") or "No title"
            webpage_url = item.get("webpage_url") or item.get("source_url") or "N/A"
            snippet = item.get("snippet") or ""
            entry = f"Image: {image_url}, Text: {title}, Webpage Url: {webpage_url}"
            if snippet:
                entry += f", Snippet: {snippet}"
            formatted.append(entry)

        print(f"[ImageSearch:bocha] Found {len(image_results)} results")
        return "```\n" + "\n\n".join(formatted) + "\n```"

    def _make_image_query(self, query: str) -> str:
        query = str(query).strip()
        if not query:
            return query
        if self.image_query_suffix and self.image_query_suffix.strip() not in query:
            return f"{query}{self.image_query_suffix}"
        return query

    def _extract_image_results(self, result_data: Dict[str, Any]) -> List[Dict[str, str]]:
        data = result_data.get("data", {}) if isinstance(result_data, dict) else {}
        results: List[Dict[str, str]] = []

        # Common Bocha/Bing-like image result shape.
        images = data.get("images") if isinstance(data, dict) else None
        if isinstance(images, dict):
            values = images.get("value", [])
            if isinstance(values, list):
                for item in values:
                    parsed = self._parse_image_item(item)
                    if parsed:
                        results.append(parsed)
        elif isinstance(images, list):
            for item in images:
                parsed = self._parse_image_item(item)
                if parsed:
                    results.append(parsed)

        web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
        pages = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        if isinstance(pages, list):
            for page in pages:
                for parsed in self._parse_page_images(page):
                    results.append(parsed)

        return self._dedupe(results)

    def _extract_webpage_results(self, result_data: Dict[str, Any]) -> List[Dict[str, str]]:
        data = result_data.get("data", {}) if isinstance(result_data, dict) else {}
        web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
        pages = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        results = []
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                url = page.get("url") or page.get("link")
                title = page.get("name") or page.get("title")
                snippet = page.get("summary") or page.get("snippet")
                if url or title:
                    results.append(
                        {
                            "image_url": "N/A",
                            "title": title or "No title",
                            "webpage_url": url or "N/A",
                            "snippet": snippet or "",
                        }
                    )
        return results

    def _parse_page_images(self, page: Any) -> List[Dict[str, str]]:
        if not isinstance(page, dict):
            return []
        page_url = page.get("url") or page.get("link") or page.get("hostPageUrl")
        page_title = page.get("name") or page.get("title")
        page_snippet = page.get("summary") or page.get("snippet")
        results = []

        for key in ("images", "image", "thumbnail", "thumbnailUrl"):
            value = page.get(key)
            if isinstance(value, list):
                for item in value:
                    parsed = self._parse_image_item(
                        item,
                        default_title=page_title,
                        default_page_url=page_url,
                        default_snippet=page_snippet,
                    )
                    if parsed:
                        results.append(parsed)
            else:
                parsed = self._parse_image_item(
                    value,
                    default_title=page_title,
                    default_page_url=page_url,
                    default_snippet=page_snippet,
                )
                if parsed:
                    results.append(parsed)

        return results

    def _parse_image_item(
        self,
        item: Any,
        *,
        default_title: Optional[str] = None,
        default_page_url: Optional[str] = None,
        default_snippet: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        if isinstance(item, str):
            if item.startswith(("http://", "https://")):
                return {
                    "image_url": item,
                    "title": default_title or "No title",
                    "webpage_url": default_page_url or "N/A",
                    "snippet": default_snippet or "",
                }
            return None
        if not isinstance(item, dict):
            return None

        image_url = (
            item.get("contentUrl")
            or item.get("imageUrl")
            or item.get("thumbnailUrl")
            or item.get("image_url")
            or item.get("url")
        )
        if not image_url:
            return None
        return {
            "image_url": image_url,
            "title": item.get("name") or item.get("title") or default_title or "No title",
            "webpage_url": (
                item.get("hostPageUrl")
                or item.get("webpageUrl")
                or item.get("sourceUrl")
                or item.get("link")
                or default_page_url
                or "N/A"
            ),
            "snippet": item.get("summary") or item.get("snippet") or default_snippet or "",
        }

    def _dedupe(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        deduped = []
        for item in results:
            key = (item.get("image_url"), item.get("webpage_url"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
