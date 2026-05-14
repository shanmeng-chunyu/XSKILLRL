#!/usr/bin/env python3
"""Probe OpenAI-compatible vLLM endpoint replicas before a full accumulation run."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL_ROOT))

from utils.api_router import normalize_chat_endpoint, split_env_list  # noqa: E402


def _data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return round(ordered[idx], 4)


def _build_messages(prompt: str, image_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not image_path:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
            ],
        }
    ]


def _post_streaming(
    *,
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    start = time.time()
    ttft = None
    usage = {}
    completion_chars = 0

    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout, stream=True)
    status_code = response.status_code
    response.raise_for_status()

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content and ttft is None:
                ttft = time.time() - start
            completion_chars += len(content)

    latency = time.time() - start
    return {
        "ok": True,
        "status_code": status_code,
        "latency_sec": latency,
        "ttft_sec": ttft,
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "completion_chars": completion_chars,
    }


def _post_non_streaming(
    *,
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    start = time.time()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    latency = time.time() - start
    status_code = response.status_code
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    content = ""
    choices = data.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content") or ""
    return {
        "ok": True,
        "status_code": status_code,
        "latency_sec": latency,
        "ttft_sec": None,
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "completion_chars": len(content),
    }


def _one_request(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    image_path: Optional[Path],
    max_tokens: int,
    timeout: int,
    stream: bool,
    request_idx: int,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, image_path),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    try:
        if stream:
            result = _post_streaming(endpoint=endpoint, headers=headers, payload=payload, timeout=timeout)
        else:
            result = _post_non_streaming(endpoint=endpoint, headers=headers, payload=payload, timeout=timeout)
        result["endpoint"] = endpoint
        result["request_idx"] = request_idx
        return result
    except Exception as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "request_idx": request_idx,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _summarize(results: Iterable[Dict[str, Any]], endpoints: List[str]) -> Dict[str, Any]:
    by_endpoint: Dict[str, List[Dict[str, Any]]] = {endpoint: [] for endpoint in endpoints}
    for item in results:
        by_endpoint.setdefault(item["endpoint"], []).append(item)

    summary: Dict[str, Any] = {"endpoints": {}, "overall": {}}
    all_items = []
    for endpoint, items in by_endpoint.items():
        all_items.extend(items)
        ok_items = [item for item in items if item.get("ok")]
        failures = [item for item in items if not item.get("ok")]
        latencies = [item["latency_sec"] for item in ok_items if item.get("latency_sec") is not None]
        ttfts = [item["ttft_sec"] for item in ok_items if item.get("ttft_sec") is not None]
        token_items = [item for item in ok_items if isinstance(item.get("completion_tokens"), int)]
        total_tokens = sum(item["completion_tokens"] for item in token_items)
        total_latency = sum(item["latency_sec"] for item in token_items if item.get("latency_sec"))

        summary["endpoints"][endpoint] = {
            "requests": len(items),
            "successes": len(ok_items),
            "failures": len(failures),
            "failure_rate": round(len(failures) / len(items), 4) if items else 0,
            "latency_p50_sec": _percentile(latencies, 0.50),
            "latency_p95_sec": _percentile(latencies, 0.95),
            "ttft_p50_sec": _percentile(ttfts, 0.50),
            "ttft_p95_sec": _percentile(ttfts, 0.95),
            "completion_tokens_per_sec": round(total_tokens / total_latency, 4) if total_latency else None,
            "sample_errors": failures[:3],
        }

    ok_all = [item for item in all_items if item.get("ok")]
    fail_all = [item for item in all_items if not item.get("ok")]
    summary["overall"] = {
        "requests": len(all_items),
        "successes": len(ok_all),
        "failures": len(fail_all),
        "failure_rate": round(len(fail_all) / len(all_items), 4) if all_items else 0,
    }
    return summary


def _default_endpoints() -> str:
    for env_name in ("LOCAL_VLM_ENDPOINTS", "REASONING_END_POINTS", "REASONING_END_POINT"):
        value = os.environ.get(env_name)
        if value:
            return value
    return "http://127.0.0.1:8002/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", default=_default_endpoints(), help="Comma-separated endpoint list.")
    parser.add_argument("--model", default=os.environ.get("REASONING_MODEL_NAME", "qwen3-vl-8b"))
    parser.add_argument("--api-key", default=os.environ.get("REASONING_API_KEY", "EMPTY"))
    parser.add_argument("--prompt", default="Briefly describe what you see. Answer in one sentence.")
    parser.add_argument("--image", type=Path, default=None, help="Optional local image path for VLM probing.")
    parser.add_argument("--requests", type=int, default=16, help="Total requests to send.")
    parser.add_argument("--concurrency", type=int, default=8, help="Client-side concurrent requests.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming and TTFT measurement.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = [normalize_chat_endpoint(item) for item in split_env_list(args.endpoints)]
    endpoints = [endpoint for endpoint in endpoints if endpoint]
    if not endpoints:
        raise SystemExit("No endpoints configured.")
    if args.image and not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    stream = not args.no_stream
    print(f"Probing {len(endpoints)} endpoint(s), requests={args.requests}, concurrency={args.concurrency}, stream={stream}")
    for endpoint in endpoints:
        print(f"  - {endpoint}")

    futures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for request_idx in range(args.requests):
            endpoint = endpoints[request_idx % len(endpoints)]
            futures.append(
                executor.submit(
                    _one_request,
                    endpoint=endpoint,
                    model=args.model,
                    api_key=args.api_key,
                    prompt=args.prompt,
                    image_path=args.image,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    stream=stream,
                    request_idx=request_idx,
                )
            )

        results = [future.result() for future in as_completed(futures)]

    summary = _summarize(results, endpoints)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["overall"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
