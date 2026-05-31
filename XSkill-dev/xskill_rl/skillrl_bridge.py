"""Bridge that lets SkillRL/verl run XSkill visual QA as an online env.

SkillRL's trainer expects an ``agent_system.environments.make_envs`` branch
that returns objects with ``reset``, ``step`` and ``success_evaluator``.
The SkillRL checkout imports this module from ``XSKILL_REPO_ROOT``.
"""

from __future__ import annotations

import json
import os
import re
import ast
import sys
import uuid
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal reasoning agent. Use the provided skills when they "
    "are relevant, reason carefully over the question and images, and produce "
    "the final answer."
)
SKILL_SECTION_TITLE = "## Retrieved Relevant Experience"
DEFAULT_TOOLS = ["web_search", "visit", "image_search", "code_interpreter", "zoom"]
TOOL_PROTOCOL = """
You may use tools when external information, webpage reading, OCR, image inspection, or calculation is needed.
Call exactly one tool at a time using this format:
<tool_call>{"name":"web_search","arguments":{"query":"search terms","max_results":5}}</tool_call>
Qwen XML tool calls are also accepted:
<tool_call>
<function=web_search>
<parameter=query>search terms</parameter>
<parameter=max_results>5</parameter>
</function>
</tool_call>

Available tools:
- web_search: {"query": "...", "max_results": 5}
- visit: {"url": "https://...", "goal": "what to find"}
- image_search: {"search_type": "text|reverse", "query": "...", "image_url": "original_image or tool_image_N", "max_results": 5}
- code_interpreter: {"code": "python code"}
- zoom: {"code": "python code to crop or inspect images"}

After each tool observation, continue reasoning or call another tool. When ready, give the final answer as <answer>...</answer>.
""".strip()


def make_xskill_envs_from_skillrl(config):
    """Factory called by the patched SkillRL ``env_manager.make_envs``."""

    train_env = XSkillVisualQAEnvironment(config=config, is_train=True)
    val_env = XSkillVisualQAEnvironment(config=config, is_train=False)
    return train_env, val_env


class XSkillVisualQAEnvironment:
    """Multi-turn XSkill visual QA environment for SkillRL rollouts."""

    def __init__(self, config, *, is_train: bool) -> None:
        self.config = config
        self.is_train = is_train
        self.cursor = 0
        self.current_items: List[Dict[str, Any]] = []
        self.episode_states: List[Dict[str, Any]] = []
        self.retrieval_memory = self._init_retrieval_memory()
        self.samples = self._load_samples_from_config()

    def reset(self, kwargs=None):
        items = self._items_from_kwargs(kwargs)
        if not items:
            items = self._next_items_from_dataset()
        self.current_items = items
        self.episode_states = [self._new_episode_state(item, idx) for idx, item in enumerate(items)]
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
        next_texts = []
        next_images = np.empty(len(self.current_items), dtype=object)
        has_next_images = False

        for idx, (item, action) in enumerate(zip(self.current_items, text_actions)):
            state = self.episode_states[idx]
            if state.get("done"):
                info = self._base_info(item)
                info.update(
                    {
                        "won": 0.0,
                        "task_score": 0.0,
                        "response": "",
                        "parsed_action_type": "finished",
                        "ground_truth": item.get("solution", ""),
                        "is_action_valid": True,
                        "tool_calling": 0.0,
                    }
                )
                rewards.append(0.0)
                dones.append(True)
                next_texts.append("Episode finished.")
                next_images[idx] = None
                infos.append(info)
                continue

            state["step"] += 1
            info = self._base_info(item)
            parsed = self._parse_model_action(action)
            reward = 0.0
            done = False
            tool_calling = 0.0
            next_text = "Episode finished."
            returned_images = []
            tool_observation = ""
            final_answer = ""

            if parsed["type"] == "tool_call" and self._tools_enabled():
                tool_calling = 1.0
                tool_result = self._execute_tool_call(parsed["name"], parsed["arguments"], state)
                tool_observation = str(tool_result.get("observation", ""))
                returned_images = tool_result.get("images", [])
                next_text = self._build_followup_observation(
                    item,
                    action=action,
                    observation=tool_observation,
                    step=state["step"],
                )
            else:
                response = parsed.get("answer", action)
                final_answer = str(response)
                reward = self._score(response, item.get("solution", ""))
                done = True

            if state["step"] >= int(_cfg_get(self.config.env, "max_steps", 1)):
                if not done:
                    reward = self._score(action, item.get("solution", ""))
                    done = True
                    final_answer = _extract_answer(action)
                    next_text = "Episode finished: maximum interaction steps reached."
            state["done"] = bool(done)

            info.update(
                {
                    "won": float(reward),
                    "task_score": float(reward),
                    "response": action,
                    "parsed_action_type": parsed["type"],
                    "ground_truth": item.get("solution", ""),
                    "is_action_valid": parsed["type"] != "invalid",
                    "tool_calling": float(tool_calling),
                    "step": int(state["step"]),
                    "reward": float(reward),
                    "done": bool(done),
                    "final_answer": final_answer,
                }
            )
            if parsed["type"] == "tool_call":
                info["tool_name"] = parsed.get("name", "")
                info["tool_arguments"] = parsed.get("arguments", {})
                info["tool_observation"] = _truncate_trace_text(tool_observation)
                info["tool_returned_image_count"] = len(returned_images)
            rewards.append(reward)
            dones.append(done)
            next_texts.append(next_text)
            if returned_images:
                next_images[idx] = returned_images
                has_next_images = True
            else:
                next_images[idx] = None
            infos.append(info)

        observations = {
            "text": next_texts,
            "image": next_images if has_next_images else None,
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
        payload["problem"] = _append_source_urls(str(payload.get("problem", "")), payload)
        return payload

    def _prompt_text(self, item: Dict[str, Any]) -> str:
        if self.retrieval_memory is not None:
            return self._runtime_skill_prompt(item)

        prompt = item.get("prompt")
        tool_section = f"\n\n## Tool Protocol\n{TOOL_PROTOCOL}" if self._tools_enabled() else ""
        if isinstance(prompt, list):
            lines = []
            for message in prompt:
                role = message.get("role", "user")
                content = message.get("content", "")
                lines.append(f"{role.upper()}:\n{content}")
            text = "\n\n".join(lines)
            if self._tools_enabled() and "Tool Protocol" not in text:
                text = f"SYSTEM:\n{DEFAULT_SYSTEM_PROMPT}{tool_section}\n\n{text}"
            return text
        if isinstance(prompt, str) and prompt:
            if self._tools_enabled() and "Tool Protocol" not in prompt:
                return f"SYSTEM:\n{DEFAULT_SYSTEM_PROMPT}{tool_section}\n\nUSER:\n{prompt}"
            return prompt
        problem = str(item.get("problem", ""))
        if self._tools_enabled():
            return f"SYSTEM:\n{DEFAULT_SYSTEM_PROMPT}{tool_section}\n\nUSER:\n{problem}"
        return problem

    def _runtime_skill_prompt(self, item: Dict[str, Any]) -> str:
        """Build the prompt from the current SkillBank at rollout time.

        This deliberately ignores any skill text already serialized into the
        parquet record.  Passing a skill bank to the runtime env should mean
        the bank is the source of truth, including skills added later by
        validation-time updates.
        """

        problem = str(item.get("problem", ""))
        skill_cfg = _cfg_get(self.config.env, "skills_only_memory", {})
        top_k = int(_cfg_get(skill_cfg, "top_k", 6) or 6)
        retrieval_payload: Dict[str, Any] = {}
        skill_text = ""
        try:
            retrieval_payload = self.retrieval_memory.retrieve(
                task_description=problem,
                top_k=top_k,
            )
            skill_text = self.retrieval_memory.format_for_prompt(retrieval_payload)
        except Exception as exc:
            print(f"[XSkillVisualQAEnvironment] Skill retrieval failed for {item.get('doc_id')}: {exc}")

        item["skill_retrieval"] = {
            "enabled": True,
            "runtime": True,
            "top_k": top_k,
            "task_type": str(retrieval_payload.get("task_type", "")),
            "retrieval_mode": str(retrieval_payload.get("retrieval_mode", "")),
        }

        system_prompt = _cfg_get(_cfg_get(self.config.env, "xskill", {}), "system_prompt", DEFAULT_SYSTEM_PROMPT)
        if skill_text and skill_text != "No relevant skills found for this task.":
            system_prompt = f"{system_prompt}\n\n{SKILL_SECTION_TITLE}\n{skill_text}"
        if self._tools_enabled():
            system_prompt = f"{system_prompt}\n\n## Tool Protocol\n{TOOL_PROTOCOL}"
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{problem}"

    def _tools_enabled(self) -> bool:
        xskill_cfg = _cfg_get(self.config.env, "xskill", {})
        return bool(_cfg_get(xskill_cfg, "enable_tools", True))

    def _enabled_tool_names(self) -> List[str]:
        xskill_cfg = _cfg_get(self.config.env, "xskill", {})
        configured = _cfg_get(xskill_cfg, "enabled_tools", None) or os.environ.get("ENABLED_TOOLS")
        if configured:
            if isinstance(configured, str):
                return [name.strip() for name in configured.split(",") if name.strip()]
            if isinstance(configured, (list, tuple)):
                return [str(name).strip() for name in configured if str(name).strip()]
        return list(DEFAULT_TOOLS)

    def _parse_model_action(self, action: str) -> Dict[str, Any]:
        text = str(action or "").strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if answer_match:
            return {"type": "answer", "answer": answer_match.group(1).strip()}

        for pattern in [
            r"<tool_call\b[^>]*>(.*?)</tool_call>",
            r"<tool\b[^>]*>(.*?)</tool>",
            r"<function_calls\b[^>]*>(.*?)</function_calls>",
            r"```(?:json)?\s*(\{.*?\"(?:name|tool_name)\".*?\})\s*```",
        ]:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                parsed = self._parse_tool_payload(match.group(1))
                if parsed:
                    return parsed

        parsed = self._parse_qwen_xml_tool_payload(text)
        if parsed:
            return parsed

        action_match = re.search(r"Action\s*:\s*([a-zA-Z_][\w]*)\s*(?:\((.*?)\)|\{(.*?)\})", text, flags=re.DOTALL)
        if action_match:
            name = action_match.group(1)
            raw_args = action_match.group(2) if action_match.group(2) is not None else "{" + action_match.group(3) + "}"
            return {"type": "tool_call", "name": name, "arguments": self._parse_arguments(raw_args)}

        if re.search(r"\b(final answer|answer)\s*(?::|\uff1a)", text, flags=re.IGNORECASE):
            return {"type": "answer", "answer": _extract_answer(text)}
        return {"type": "answer", "answer": text}

    def _parse_tool_payload(self, payload: str) -> Optional[Dict[str, Any]]:
        payload = payload.strip()
        qwen_xml = self._parse_qwen_xml_tool_payload(payload)
        if qwen_xml:
            return qwen_xml
        try:
            data = json.loads(payload)
        except Exception:
            try:
                data = ast.literal_eval(payload)
            except Exception:
                return None
        if not isinstance(data, dict):
            return None
        name = data.get("name") or data.get("tool_name") or data.get("tool")
        arguments = data.get("arguments") or data.get("parameters") or data.get("args") or {}
        if isinstance(arguments, str):
            arguments = self._parse_arguments(arguments)
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        if not name:
            return None
        return {"type": "tool_call", "name": str(name), "arguments": arguments}

    def _parse_qwen_xml_tool_payload(self, payload: str) -> Optional[Dict[str, Any]]:
        """Parse Qwen-style XML function calls emitted by local vLLM rollouts."""

        text = str(payload or "").strip()
        if not text:
            return None

        tool_match = re.search(r"<tool_call\b[^>]*>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
        if tool_match:
            text = tool_match.group(1).strip()

        function_match = re.search(
            r"<function=([A-Za-z_][\w.-]*)\s*>(.*?)</function>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not function_match:
            function_match = re.search(
                r"<function\s+name=[\"']?([A-Za-z_][\w.-]*)[\"']?\s*>(.*?)</function>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not function_match:
            return None

        name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments: Dict[str, Any] = {}
        for param_match in re.finditer(
            r"<parameter=([A-Za-z_][\w.-]*)\s*>(.*?)</parameter>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            key = param_match.group(1).strip()
            arguments[key] = _parse_scalar(param_match.group(2).strip())
        for param_match in re.finditer(
            r"<parameter\s+name=[\"']?([A-Za-z_][\w.-]*)[\"']?\s*>(.*?)</parameter>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            key = param_match.group(1).strip()
            arguments[key] = _parse_scalar(param_match.group(2).strip())

        return {"type": "tool_call", "name": name, "arguments": arguments}

    def _parse_arguments(self, raw_args: str) -> Dict[str, Any]:
        raw_args = str(raw_args or "").strip()
        if not raw_args:
            return {}
        try:
            value = json.loads(raw_args)
        except Exception:
            try:
                value = ast.literal_eval(raw_args)
            except Exception:
                return {"query": raw_args}
        return value if isinstance(value, dict) else {"input": value}

    def _new_episode_state(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        node = SimpleNamespace(
            node_id=str(item.get("doc_id") or index),
            image_map={},
            conversation_history=[],
            api_conversation_history=[],
            current_token_count=0,
        )
        for image_idx, image in enumerate(self._load_pil_images(item)):
            key = "original_image" if image_idx == 0 else f"original_image_{image_idx}"
            node.image_map[key] = image
        save_dir = self._tool_save_dir(item, index)
        return {
            "step": 0,
            "done": False,
            "node": node,
            "tool_handler": self._build_tool_handler(save_dir),
            "save_dir": save_dir,
        }

    def _tool_save_dir(self, item: Dict[str, Any], index: int) -> str:
        xskill_cfg = _cfg_get(self.config.env, "xskill", {})
        configured = _cfg_get(xskill_cfg, "tool_save_dir", None)
        if configured:
            root = Path(str(configured))
        else:
            repo_root = os.environ.get("XSKILL_REPO_ROOT") or _cfg_get(xskill_cfg, "repo_root", ".")
            root = Path(str(repo_root)) / "output" / "skillrl_tool_runs"
        split = "train" if self.is_train else "val"
        doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("doc_id") or index))[:80]
        path = root / split / f"{doc_id}_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _build_tool_handler(self, save_dir: str):
        self._ensure_eval_imports()
        try:
            from engine.api_tool_handler import APIToolHandler
        except Exception as exc:
            print(f"[XSkillVisualQAEnvironment] Could not import APIToolHandler: {exc}")
            return None
        xskill_cfg = _cfg_get(self.config.env, "xskill", {})
        args = SimpleNamespace(
            max_pixels=int(_cfg_get(xskill_cfg, "tool_max_pixels", 1024 * 1024)),
            min_pixels=int(_cfg_get(xskill_cfg, "tool_min_pixels", 128 * 128)),
            image_search_max_calls=int(_cfg_get(xskill_cfg, "image_search_max_calls", 3)),
            web_search_max_calls=int(_cfg_get(xskill_cfg, "web_search_max_calls", 5)),
            tool_configs={},
        )
        return APIToolHandler(args=args, save_dir=save_dir)

    def _ensure_eval_imports(self) -> None:
        repo_root = _cfg_get(_cfg_get(self.config.env, "xskill", {}), "repo_root", None) or os.environ.get("XSKILL_REPO_ROOT")
        if not repo_root:
            return
        eval_root = str(Path(str(repo_root)) / "eval")
        for path in (str(repo_root), eval_root):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        handler = state.get("tool_handler")
        if handler is None:
            return {"observation": "Tool execution is unavailable in this environment.", "images": []}
        enabled_tools = self._enabled_tool_names()
        if tool_name not in enabled_tools:
            return {"observation": f"Tool '{tool_name}' is not enabled. Enabled tools: {', '.join(enabled_tools)}", "images": []}
        try:
            result = handler.execute_tool_call(
                tool_name=tool_name,
                parameters=dict(arguments or {}),
                node=state["node"],
                turn_idx=int(state.get("step", 0)),
                tool_call_id=f"call_{state.get('step', 0)}",
            )
            observation = result.get("processed_result") or result.get("tool_result") or ""
            images = [np.asarray(image.convert("RGB")) for _, image in result.get("new_images", [])]
            return {"observation": str(observation), "images": images}
        except Exception as exc:
            return {"observation": f"Tool execution failed for {tool_name}: {exc}", "images": []}

    def _build_followup_observation(self, item: Dict[str, Any], *, action: str, observation: str, step: int) -> str:
        max_steps = int(_cfg_get(self.config.env, "max_steps", 1))
        return (
            f"Previous assistant message:\n{action}\n\n"
            f"Tool observation:\n{observation}\n\n"
            f"Step {step}/{max_steps}. Continue reasoning, call another tool if needed, "
            "or provide the final answer as <answer>...</answer>."
        )

    def _load_pil_images(self, item: Dict[str, Any]) -> List[Any]:
        try:
            from PIL import Image
        except Exception:
            return []
        images = []
        for image_path in self._resolve_image_paths(item):
            try:
                if _is_http_url(image_path):
                    image = self._load_remote_image(str(image_path), Image)
                else:
                    image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception:
                continue
        return images

    def _init_retrieval_memory(self):
        if not bool(_cfg_get(self.config.env, "use_skills_only_memory", False)):
            return None

        skill_cfg = _cfg_get(self.config.env, "skills_only_memory", {})
        skills_json_path = _cfg_get(skill_cfg, "skills_json_path", None)
        if not skills_json_path:
            print("[XSkillVisualQAEnvironment] use_skills_only_memory is enabled but no skills_json_path was provided")
            return None

        try:
            from agent_system.memory import SkillsOnlyMemory
        except Exception as exc:
            print(f"[XSkillVisualQAEnvironment] Could not import SkillRL SkillsOnlyMemory: {exc}")
            return None

        memory_kwargs = {
            "skills_json_path": str(skills_json_path),
            "retrieval_mode": _cfg_get(skill_cfg, "retrieval_mode", "template"),
            "embedding_model_path": _cfg_get(skill_cfg, "embedding_model_path", None),
            "task_specific_top_k": _cfg_get(skill_cfg, "task_specific_top_k", None),
        }
        try:
            memory = SkillsOnlyMemory(**memory_kwargs)
            mode = memory_kwargs["retrieval_mode"]
            split = "train" if self.is_train else "val"
            print(f"[XSkillVisualQAEnvironment] Runtime skill retrieval enabled for {split}: {skills_json_path} ({mode})")
            return memory
        except Exception as exc:
            print(f"[XSkillVisualQAEnvironment] Failed to load skills-only memory from {skills_json_path}: {exc}")
            return None

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
            localized = _resolve_remote_image_url(text, self.config)
            return localized if localized is not None else text
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
            "skill_retrieval": item.get("skill_retrieval", {}),
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


def _resolve_remote_image_url(url: str, config) -> Optional[Path]:
    parsed = urlparse(url)
    if not parsed.netloc:
        return None
    host = parsed.netloc.replace(":", "_")
    url_path = Path(unquote(parsed.path.lstrip("/")))
    if not url_path.name:
        return None

    relative_dir = Path(host) / url_path.parent
    exact_relative = relative_dir / url_path.name
    glob_pattern = f"{url_path.stem}_*{url_path.suffix or '.*'}"

    roots: List[Path] = []
    for root in _candidate_image_roots(config):
        roots.extend([root, root / "_remote_images", root / "benchmark" / "_remote_images"])

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        exact = root / exact_relative
        if exact.is_file():
            return exact
        search_dir = root / relative_dir
        try:
            matches = sorted(search_dir.glob(glob_pattern))
        except OSError:
            matches = []
        for match in matches:
            if match.is_file():
                return match
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


def _append_source_urls(problem: str, item: Dict[str, Any]) -> str:
    source = item.get("source")
    if source is None:
        return problem
    values = source if isinstance(source, list) else [source]
    urls: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in urls:
            urls.append(text)
    if not urls:
        return problem
    existing = str(problem or "").lower()
    missing_urls = [url for url in urls if url.lower() not in existing]
    if not missing_urls:
        return str(problem or "").strip()
    source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(missing_urls, start=1))
    source_section = (
        "Source URLs provided by the benchmark. Use the visit tool on these URLs "
        "when they are relevant:\n"
        f"{source_lines}"
    )
    problem_text = str(problem or "").strip()
    return f"{problem_text}\n\n{source_section}" if problem_text else source_section


def _as_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                return _as_list(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                pass
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return _as_list(value.tolist())
        except Exception:
            pass
    return [value]


def _truncate_trace_text(value: Any) -> str:
    text = str(value or "")
    try:
        limit = int(os.environ.get("XSKILL_VAL_TRAJECTORY_OBS_MAX_CHARS", "12000"))
    except ValueError:
        limit = 12000
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text


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


def _parse_scalar(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        pass
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
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
