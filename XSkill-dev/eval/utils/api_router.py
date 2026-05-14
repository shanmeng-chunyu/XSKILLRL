"""Shared OpenAI-compatible endpoint routing and timing utilities."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import requests


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EndpointSpec:
    endpoint: str
    api_key: str
    name: str = ""


class _EndpointState:
    def __init__(self) -> None:
        self.inflight = 0
        self.unhealthy_until = 0.0
        self.last_error = ""


class EndpointRouter:
    """Thread-safe least-inflight router shared across all local VLM calls."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._states: Dict[str, _EndpointState] = {}
        self._cursor = 0

    def acquire(self, specs: List[EndpointSpec]) -> EndpointSpec:
        if not specs:
            raise ValueError("No endpoint specs configured")

        max_inflight = int(os.environ.get("API_MAX_INFLIGHT_PER_ENDPOINT", "4"))
        wait_timeout = float(os.environ.get("API_ROUTER_ACQUIRE_TIMEOUT", "60"))
        deadline = time.time() + wait_timeout

        with self._condition:
            while True:
                now = time.time()
                states = [(spec, self._states.setdefault(spec.endpoint, _EndpointState())) for spec in specs]
                healthy = [(spec, state) for spec, state in states if state.unhealthy_until <= now]
                candidate_pool = healthy or states
                available = [(spec, state) for spec, state in candidate_pool if state.inflight < max_inflight]

                if available:
                    min_inflight = min(state.inflight for _, state in available)
                    endpoint_to_pair = {spec.endpoint: (spec, state) for spec, state in available}
                    ties = [
                        endpoint_to_pair[spec.endpoint]
                        for spec in specs
                        if spec.endpoint in endpoint_to_pair
                        and endpoint_to_pair[spec.endpoint][1].inflight == min_inflight
                    ]
                    spec, state = ties[self._cursor % len(ties)]
                    self._cursor += 1
                    state.inflight += 1
                    return spec

                if time.time() >= deadline:
                    spec, state = min(candidate_pool, key=lambda item: (item[1].inflight, item[0].endpoint))
                    state.inflight += 1
                    return spec

                self._condition.wait(timeout=0.1)

    def release(self, spec: EndpointSpec) -> None:
        with self._condition:
            state = self._states.setdefault(spec.endpoint, _EndpointState())
            state.inflight = max(0, state.inflight - 1)
            self._condition.notify_all()

    def mark_unhealthy(self, spec: EndpointSpec, error: str) -> None:
        cooldown = float(os.environ.get("API_ENDPOINT_COOLDOWN_SECONDS", "10"))
        with self._condition:
            state = self._states.setdefault(spec.endpoint, _EndpointState())
            state.unhealthy_until = time.time() + cooldown
            state.last_error = error
            self._condition.notify_all()


_ROUTER = EndpointRouter()
_TIMING_LOCK = threading.Lock()


def split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_chat_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return endpoint
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/chat/completions"


def _make_spec(endpoint: str, api_key: Optional[str], *, require_chat_completions: bool) -> EndpointSpec:
    normalized = normalize_chat_endpoint(endpoint) if require_chat_completions else endpoint.strip()
    return EndpointSpec(endpoint=normalized, api_key=api_key or "EMPTY", name=normalized)


def collect_endpoint_specs(
    *,
    plural_env: Optional[str] = None,
    singular_env: Optional[str] = None,
    api_key_env: Optional[str] = None,
    fallback_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    default_api_key: Optional[str] = None,
    require_chat_completions: bool = True,
) -> List[EndpointSpec]:
    """Collect endpoint specs from plural env first, then legacy singular/fallback envs."""

    specs: List[EndpointSpec] = []
    seen = set()

    key = (os.environ.get(api_key_env) if api_key_env else None) or default_api_key or "EMPTY"

    if plural_env:
        endpoints = split_env_list(os.environ.get(plural_env))
        if endpoints:
            for endpoint in endpoints:
                spec = _make_spec(endpoint, key, require_chat_completions=require_chat_completions)
                if spec.endpoint not in seen:
                    specs.append(spec)
                    seen.add(spec.endpoint)
            return specs

    if singular_env and os.environ.get(singular_env):
        spec = _make_spec(os.environ[singular_env], key, require_chat_completions=require_chat_completions)
        if spec.endpoint not in seen:
            specs.append(spec)
            seen.add(spec.endpoint)

    for endpoint_env, key_env in fallback_pairs or []:
        endpoint = os.environ.get(endpoint_env)
        if not endpoint:
            continue
        pair_key = os.environ.get(key_env) or key
        spec = _make_spec(endpoint, pair_key, require_chat_completions=require_chat_completions)
        if spec.endpoint not in seen:
            specs.append(spec)
            seen.add(spec.endpoint)

    return specs


def get_timing_log_path(output_dir: Optional[str] = None) -> Optional[Path]:
    explicit = os.environ.get("XSKILL_API_TIMING_PATH")
    if explicit:
        return Path(explicit)
    root = output_dir or os.environ.get("XSKILL_OUTPUT_DIR")
    if not root:
        return None
    return Path(root) / "api_timings.jsonl"


def record_api_timing(
    *,
    kind: str,
    endpoint: str,
    latency: float,
    status_code: Optional[int] = None,
    error_type: Optional[str] = None,
    retry_count: int = 0,
    usage: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> None:
    path = get_timing_log_path(output_dir)
    if not path:
        return

    usage = usage or {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "endpoint": endpoint,
        "latency_sec": round(float(latency), 4),
        "status_code": status_code,
        "error_type": error_type,
        "retry_count": retry_count,
        "thread_id": threading.get_ident(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _TIMING_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[API Timing] Failed to write timing log: {exc}")


def routed_post_once(
    *,
    specs: List[EndpointSpec],
    payload_factory: Callable[[EndpointSpec], dict],
    timeout: int,
    request_kind: str,
    api_name: str = "API",
    retry_count: int = 0,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[requests.Response], Optional[str], EndpointSpec]:
    """POST once through the shared router and record timing."""

    spec = _ROUTER.acquire(specs)
    start = time.time()
    status_code = None
    error_type = None
    response = None

    try:
        payload = payload_factory(spec)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {spec.api_key}",
        }
        response = requests.post(spec.endpoint, headers=headers, json=payload, timeout=timeout)
        status_code = response.status_code
        if status_code in RETRYABLE_STATUS_CODES:
            error_type = str(status_code)
            _ROUTER.mark_unhealthy(spec, error_type)
        return response, error_type, spec
    except requests.exceptions.Timeout:
        error_type = "timeout"
        print(f"[{api_name}] API timeout at {spec.endpoint}")
        _ROUTER.mark_unhealthy(spec, error_type)
        return None, error_type, spec
    except requests.exceptions.RequestException as exc:
        error_type = "network"
        print(f"[{api_name}] API call failed at {spec.endpoint}: {exc}")
        _ROUTER.mark_unhealthy(spec, error_type)
        return None, error_type, spec
    except Exception as exc:
        error_type = "other"
        print(f"[{api_name}] Unexpected error at {spec.endpoint}: {exc}")
        _ROUTER.mark_unhealthy(spec, error_type)
        return None, error_type, spec
    finally:
        latency = time.time() - start
        usage = None
        if response is not None:
            try:
                body = response.json()
                usage = body.get("usage") if isinstance(body, dict) else None
            except Exception:
                usage = None
        record_api_timing(
            kind=request_kind,
            endpoint=spec.endpoint,
            latency=latency,
            status_code=status_code,
            error_type=error_type,
            retry_count=retry_count,
            usage=usage,
            output_dir=output_dir,
        )
        _ROUTER.release(spec)
