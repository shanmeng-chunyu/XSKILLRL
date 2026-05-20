"""Bridge that lets SkillRL/verl run XSkill visual QA as an online env.

SkillRL's trainer expects an ``agent_system.environments.make_envs`` branch
that returns objects with ``reset``, ``step`` and ``success_evaluator``.
The SkillRL checkout imports this module from ``XSKILL_REPO_ROOT``.
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def make_xskill_envs_from_skillrl(config):
    """Factory called by the patched SkillRL ``env_manager.make_envs``."""

    train_env = XSkillVisualQAEnvironment(config=config, is_train=True)
    val_env = XSkillVisualQAEnvironment(config=config, is_train=False)
    return train_env, val_env


class XSkillVisualQAEnvironment:
    """Single-step visual QA environment for GRPO rollouts.

    Each rollout sample is one episode.  The model receives the current sample
    prompt, returns one response, and gets a binary reward from the configured
    answer matcher.
    """

    def __init__(self, config, *, is_train: bool) -> None:
        self.config = config
        self.is_train = is_train
        self.cursor = 0
        self.current_items: List[Dict[str, Any]] = []
        self.samples = self._load_samples_from_config()

    def reset(self, kwargs=None):
        items = self._items_from_kwargs(kwargs)
        if not items:
            items = self._next_items_from_dataset()
        self.current_items = items
        observations = {
            "text": [self._prompt_text(item) for item in items],
            "image": self._load_images(items),
            "anchor": np.array([item.get("doc_id", str(i)) for i, item in enumerate(items)], dtype=object),
        }
        infos = [self._base_info(item) for item in items]
        return observations, infos

    def step(self, text_actions: Sequence[str]):
        rewards = []
        dones = []
        infos = []
        for item, action in zip(self.current_items, text_actions):
            reward = self._score(action, item.get("solution", ""))
            rewards.append(reward)
            dones.append(True)
            info = self._base_info(item)
            info.update(
                {
                    "won": float(reward),
                    "task_score": float(reward),
                    "response": action,
                    "ground_truth": item.get("solution", ""),
                    "is_action_valid": True,
                    "tool_calling": 0.0,
                }
            )
            infos.append(info)

        observations = {
            "text": ["Episode finished." for _ in self.current_items],
            "image": None,
            "anchor": np.array([item.get("doc_id", str(i)) for i, item in enumerate(self.current_items)], dtype=object),
        }
        return (
            observations,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        success: Dict[str, List[float]] = {
            "success_rate": [],
        }

        for batch_idx, trajectory in enumerate(total_batch_list):
            chosen_info = None
            for step_idx in reversed(range(len(trajectory))):
                if trajectory[step_idx].get("active_masks"):
                    chosen_info = total_infos[batch_idx][step_idx]
                    break
            if chosen_info is None and total_infos[batch_idx]:
                chosen_info = total_infos[batch_idx][-1]
            if chosen_info is None:
                value = 0.0
                benchmark = "unknown"
            else:
                value = float(chosen_info.get("won", 0.0))
                benchmark = str(chosen_info.get("data_source") or chosen_info.get("benchmark_name") or "unknown")
            success["success_rate"].append(value)
            success.setdefault(f"{benchmark}_success_rate", []).append(value)

        batch_len = len(total_batch_list)
        return {
            key: _pad_metric(values, batch_len)
            for key, values in success.items()
        }

    def close(self) -> None:
        return None

    def _items_from_kwargs(self, kwargs) -> List[Dict[str, Any]]:
        if kwargs is None:
            return []
        if isinstance(kwargs, np.ndarray):
            raw_items = kwargs.tolist()
        elif isinstance(kwargs, list):
            raw_items = kwargs
        else:
            raw_items = [kwargs]
        return [self._normalize_item(item) for item in raw_items if item is not None]

    def _next_items_from_dataset(self) -> List[Dict[str, Any]]:
        batch_size = _cfg_get(self.config.data, "train_batch_size", 1)
        if not self.is_train:
            batch_size = _cfg_get(self.config.data, "val_batch_size", batch_size) or batch_size
        if self.is_train:
            group_n = _cfg_get(self.config.env.rollout, "n", 1)
            batch_size = int(batch_size) * max(1, int(group_n))
        batch_size = int(batch_size)
        if not self.samples:
            raise ValueError("No XSkill samples are available for SkillRL environment reset.")
        items = []
        for _ in range(batch_size):
            items.append(self.samples[self.cursor % len(self.samples)])
            self.cursor += 1
        return [dict(item) for item in items]

    def _load_samples_from_config(self) -> List[Dict[str, Any]]:
        data_files = _cfg_get(self.config.data, "train_files", None)
        if not self.is_train:
            data_files = _cfg_get(self.config.data, "val_files", data_files)
        samples = []
        for record in _iter_records_from_files(data_files):
            item = _first_present(record, "env_kwargs")
            if not isinstance(item, dict):
                item = {}
            if not item:
                extra_info = _first_present(record, "extra_info")
                if isinstance(extra_info, dict):
                    sample = _first_present(extra_info, "sample")
                    if isinstance(sample, dict):
                        item = sample
            if item:
                samples.append(self._normalize_item(item))
        return samples

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(item)
        sample = payload.get("sample")
        if isinstance(sample, dict):
            merged = dict(sample)
            merged.update(payload)
            payload = merged
        payload.setdefault("doc_id", payload.get("question_id", "unknown"))
        payload.setdefault("benchmark_name", payload.get("data_source", "xskill"))
        payload.setdefault("problem", payload.get("question", ""))
        payload.setdefault("solution", payload.get("answer", ""))
        payload.setdefault("images", _as_list(_first_present(payload, "images", "image", "img")))
        return payload

    def _prompt_text(self, item: Dict[str, Any]) -> str:
        prompt = item.get("prompt")
        if isinstance(prompt, list):
            lines = []
            for message in prompt:
                role = message.get("role", "user")
                content = message.get("content", "")
                lines.append(f"{role.upper()}:\n{content}")
            return "\n\n".join(lines)
        if isinstance(prompt, str) and prompt:
            return prompt
        return str(item.get("problem", ""))

    def _load_images(self, items: Sequence[Dict[str, Any]]):
        image_arrays = np.empty(len(items), dtype=object)
        has_image = False
        for idx, item in enumerate(items):
            image_paths = self._resolve_image_paths(item)
            if not image_paths:
                image_arrays[idx] = None
                continue
            loaded_images = []
            try:
                from PIL import Image
            except Exception as exc:
                print(f"[XSkillVisualQAEnvironment] PIL import failed: {exc}")
                image_arrays[idx] = None
                continue
            for image_path in image_paths:
                try:
                    if _is_http_url(image_path):
                        image = self._load_remote_image(str(image_path), Image)
                    else:
                        image = Image.open(image_path).convert("RGB")
                    loaded_images.append(np.asarray(image))
                except Exception as exc:
                    print(f"[XSkillVisualQAEnvironment] Failed to load image {image_path}: {exc}")
            if loaded_images:
                image_arrays[idx] = loaded_images
                has_image = True
            else:
                image_arrays[idx] = None
        if not has_image:
            return None
        return image_arrays

    def _load_remote_image(self, url: str, image_cls):
        from urllib.request import Request, urlopen

        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return image_cls.open(BytesIO(response.read())).convert("RGB")

    def _resolve_first_image(self, item: Dict[str, Any]) -> Optional[Any]:
        paths = self._resolve_image_paths(item, max_images=1)
        return paths[0] if paths else None

    def _resolve_image_paths(self, item: Dict[str, Any], max_images: Optional[int] = None) -> List[Any]:
        images = _as_list(item.get("images"))
        if not images:
            return []
        resolved = []
        for image in images:
            path = self._resolve_one_image(image)
            if path is not None:
                resolved.append(path)
                if max_images is not None and len(resolved) >= max_images:
                    break
        return resolved

    def _resolve_one_image(self, image: Any) -> Optional[Any]:
        if isinstance(image, dict):
            image = image.get("image") or image.get("path") or image.get("url")
        if image is None:
            return None
        text = _normalize_url(str(image))
        if _is_http_url(text):
            return text
        if text.startswith("file://"):
            parsed = urlparse(text)
            candidate = Path(unquote(parsed.path))
            if candidate.is_file():
                return candidate
        first = Path(text)
        if first.is_file():
            return first
        relocated = _resolve_relocated_image(first, self.config)
        if relocated is not None:
            return relocated
        env_image_root = os.environ.get("XSKILL_IMAGE_ROOT")
        if env_image_root:
            candidate = Path(env_image_root) / first
            if candidate.is_file():
                return candidate
        image_root = _cfg_get(_cfg_get(self.config.env, "xskill", {}), "image_root", None)
        if image_root:
            candidate = Path(str(image_root)) / first
            if candidate.is_file():
                return candidate
        env_repo_root = os.environ.get("XSKILL_REPO_ROOT")
        if env_repo_root:
            candidate = Path(env_repo_root) / first
            if candidate.is_file():
                return candidate
        repo_root = _cfg_get(_cfg_get(self.config.env, "xskill", {}), "repo_root", None)
        if repo_root:
            candidate = Path(str(repo_root)) / first
            if candidate.is_file():
                return candidate
        return first

    def _score(self, response: str, solution: str) -> float:
        if not solution:
            return 0.0
        mode = _cfg_get(_cfg_get(self.config.env, "xskill", {}), "reward_mode", "contains")
        prediction = _normalize_answer(_extract_answer(response))
        target = _normalize_answer(solution)
        if not prediction or not target:
            return 0.0
        if mode == "exact":
            return 1.0 if prediction == target else 0.0
        if prediction == target or target in prediction or prediction in target:
            return 1.0
        return 0.0

    def _base_info(self, item: Dict[str, Any]) -> Dict[str, Any]:
        benchmark = item.get("benchmark_name", "xskill")
        return {
            "doc_id": item.get("doc_id"),
            "benchmark_name": benchmark,
            "data_source": benchmark,
            "solution": item.get("solution", ""),
            "won": 0.0,
            "task_score": 0.0,
            "is_action_valid": True,
            "tool_calling": 0.0,
        }


def _image_suffixes(path: Path) -> List[Path]:
    parts = path.parts
    suffixes: List[Path] = []
    for marker in ("images", "benchmark"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                suffixes.append(Path(*parts[index + 1 :]))
            suffixes.append(Path(*parts[index:]))
    if path.name:
        suffixes.append(Path(path.name))
    return suffixes


def _candidate_image_roots(config) -> List[Path]:
    roots: List[Path] = []
    env_image_root = os.environ.get("XSKILL_IMAGE_ROOT")
    if env_image_root:
        roots.append(Path(env_image_root))

    image_root = _cfg_get(_cfg_get(config.env, "xskill", {}), "image_root", None)
    if image_root:
        roots.append(Path(str(image_root)))

    env_repo_root = os.environ.get("XSKILL_REPO_ROOT")
    if env_repo_root:
        repo_path = Path(env_repo_root)
        roots.extend([repo_path, repo_path / "benchmark", repo_path.parent / "images"])

    repo_root = _cfg_get(_cfg_get(config.env, "xskill", {}), "repo_root", None)
    if repo_root:
        repo_path = Path(str(repo_root))
        roots.extend([repo_path, repo_path / "benchmark", repo_path.parent / "images"])

    cwd = Path.cwd()
    roots.extend([cwd, cwd / "images", cwd.parent / "images", cwd.parent / "XSkill-dev"])
    return roots


def _resolve_relocated_image(path: Path, config) -> Optional[Path]:
    candidates = [path]
    if path.is_absolute():
        candidates = _image_suffixes(path)
    else:
        candidates.extend(_image_suffixes(path))
    for root in _candidate_image_roots(config):
        for candidate in candidates:
            resolved = root / candidate
            if resolved.is_file():
                return resolved
    return None


def _iter_records_from_files(data_files) -> Iterable[Dict[str, Any]]:
    if data_files is None:
        return []
    paths = data_files if isinstance(data_files, list) else [data_files]
    records = []
    for path_like in paths:
        path = Path(str(path_like)).expanduser()
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            records.extend(payload if isinstance(payload, list) else [payload])
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        elif suffix == ".parquet":
            import pandas as pd

            records.extend(pd.read_parquet(path).to_dict(orient="records"))
    return records


def _cfg_get(container, key: str, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _normalize_url(value: str) -> str:
    text = value.strip()
    if text.startswith("https:/") and not text.startswith("https://"):
        return "https://" + text[len("https:/") :]
    if text.startswith("http:/") and not text.startswith("http://"):
        return "http://" + text[len("http:/") :]
    return text


def _is_http_url(value: Any) -> bool:
    text = str(value)
    return text.startswith(("http://", "https://"))


def _as_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _first_present(container: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in container:
            continue
        value = container[key]
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return None


def _extract_answer(response: str) -> str:
    text = str(response or "").strip()
    for pattern in [
        r"<answer>(.*?)</answer>",
        r"final answer\s*(?::|\uff1a)\s*(.*)",
        r"answer\s*(?::|\uff1a)\s*(.*)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _normalize_answer(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pad_metric(values: List[float], length: int) -> np.ndarray:
    if len(values) == length:
        return np.asarray(values, dtype=np.float32)
    padded = list(values) + [0.0] * max(0, length - len(values))
    return np.asarray(padded[:length], dtype=np.float32)
