"""Lightweight experiment recorder for SkillRL/verl runs.

The recorder writes one directory per run. It intentionally appends JSONL files
before rendering markdown so interrupted runs still keep all completed steps.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in training env
    torch = None

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in training env
    np = None

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - omegaconf is expected in training env
    OmegaConf = None


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _safe_name(value: Any) -> str:
    text = str(value if value is not None else "run").strip()
    safe = []
    for char in text:
        safe.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(safe).strip("._-") or "run"


def _to_builtin(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _to_builtin(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def _select(config: Any, path: str, default: Any = "") -> Any:
    if OmegaConf is not None:
        try:
            value = OmegaConf.select(config, path)
            return default if value is None else value
        except Exception:
            pass
    current = config
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _fmt(value: Any, digits: int = 4) -> str:
    value = _to_builtin(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _make_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


class SkillRLExperimentRecorder:
    """Append-only metrics recorder with a markdown summary."""

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        train_size: int | None = None,
        val_size: int | None = None,
        total_training_steps: int | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.train_size = train_size
        self.val_size = val_size
        self.total_training_steps = total_training_steps
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.validation_path = self.run_dir / "validation.jsonl"
        self.checkpoints_path = self.run_dir / "checkpoints.jsonl"
        self.record_path = self.run_dir / "experiment_record.md"
        self.config_path = self.run_dir / "resolved_config.json"
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.status = "running"
        self.latest_step: dict[str, Any] | None = None
        self.latest_validation: dict[str, Any] | None = None
        self.checkpoints: list[dict[str, Any]] = []
        self.recent_steps: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self._write_config()
        self._render()

    @classmethod
    def from_config(
        cls,
        config: Any,
        train_size: int | None = None,
        val_size: int | None = None,
        total_training_steps: int | None = None,
    ) -> "SkillRLExperimentRecorder | None":
        if not _truthy(os.environ.get("SKILLRL_EXPERIMENT_RECORD"), default=True):
            return None

        explicit_dir = os.environ.get("SKILLRL_EXPERIMENT_RECORD_DIR")
        if explicit_dir:
            run_dir = Path(explicit_dir)
        else:
            base = Path(str(_select(config, "trainer.default_local_dir", "checkpoints/skillrl")))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment = _safe_name(_select(config, "trainer.experiment_name", "skillrl"))
            run_dir = base / "experiment_records" / f"{timestamp}_{experiment}_pid{os.getpid()}"
        return cls(
            run_dir=run_dir,
            config=config,
            train_size=train_size,
            val_size=val_size,
            total_training_steps=total_training_steps,
        )

    def _write_config(self) -> None:
        if self.config_path.exists():
            return
        try:
            if OmegaConf is not None:
                data = OmegaConf.to_container(self.config, resolve=True)
            else:
                data = _to_builtin(self.config)
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.config_path.write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, row: Mapping[str, Any]) -> None:
        serializable = _to_builtin(dict(row))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def log_validation(self, step: int, metrics: Mapping[str, Any], stage: str = "validation") -> None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "step": step,
            "metrics": _to_builtin(metrics),
        }
        self.latest_validation = row
        self.validation_rows.append(row)
        self.validation_rows = self.validation_rows[-20:]
        self._append_jsonl(self.validation_path, row)
        self._render()

    def log_step(self, step: int, epoch: int, metrics: Mapping[str, Any]) -> None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "epoch": epoch,
            "metrics": _to_builtin(metrics),
        }
        self.latest_step = row
        self.recent_steps.append(row)
        self.recent_steps = self.recent_steps[-30:]
        self._append_jsonl(self.metrics_path, row)

        interval = int(os.environ.get("SKILLRL_EXPERIMENT_RECORD_INTERVAL", "1"))
        if interval <= 1 or step % interval == 0:
            self._render()

    def log_checkpoint(self, step: int, path: str) -> None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "path": path,
        }
        self.checkpoints.append(row)
        self.checkpoints = self.checkpoints[-20:]
        self._append_jsonl(self.checkpoints_path, row)
        self._render()

    def mark_status(self, status: str, message: str = "") -> None:
        self.status = status
        self._render(message=message)

    def mark_interrupted_if_running(self) -> None:
        if self.status == "running":
            self.mark_status("interrupted", "process exited before training completed")

    def _config_rows(self) -> list[list[Any]]:
        keys = [
            ("project_name", "trainer.project_name"),
            ("experiment_name", "trainer.experiment_name"),
            ("model", "actor_rollout_ref.model.path"),
            ("train_files", "data.train_files"),
            ("val_files", "data.val_files"),
            ("train_batch_size", "data.train_batch_size"),
            ("val_batch_size", "data.val_batch_size"),
            ("max_prompt_length", "data.max_prompt_length"),
            ("max_response_length", "data.max_response_length"),
            ("adv_estimator", "algorithm.adv_estimator"),
            ("env_name", "env.env_name"),
            ("env.rollout.n", "env.rollout.n"),
            ("max_steps", "env.max_steps"),
            ("reward_mode", "env.xskill.reward_mode"),
            ("skills_json_path", "env.skills_only_memory.skills_json_path"),
            ("n_gpus_per_node", "trainer.n_gpus_per_node"),
            ("total_epochs", "trainer.total_epochs"),
            ("save_freq", "trainer.save_freq"),
            ("test_freq", "trainer.test_freq"),
            ("default_local_dir", "trainer.default_local_dir"),
        ]
        return [[name, _select(self.config, path, "")] for name, path in keys]

    def _metric(self, row: dict[str, Any] | None, *names: str) -> Any:
        if not row:
            return ""
        metrics = row.get("metrics", {})
        for name in names:
            if name in metrics:
                return metrics[name]
        return ""

    def _latest_rows(self) -> list[list[Any]]:
        row = self.latest_step
        return [
            ["step", row.get("step", "") if row else ""],
            ["epoch", row.get("epoch", "") if row else ""],
            ["reward mean", self._metric(row, "episode/reward/mean", "critic/rewards/mean", "critic/score/mean")],
            ["reward max", self._metric(row, "episode/reward/max", "critic/rewards/max", "critic/score/max")],
            ["reward min", self._metric(row, "episode/reward/min", "critic/rewards/min", "critic/score/min")],
            ["response length mean", self._metric(row, "response_length/mean")],
            ["response clip ratio", self._metric(row, "response_length/clip_ratio")],
            ["entropy", self._metric(row, "actor/entropy_loss")],
            ["actor loss", self._metric(row, "actor/pg_loss", "actor/loss")],
            ["ppo kl", self._metric(row, "actor/ppo_kl", "actor/kl_loss", "actor/reward_kl_penalty")],
            ["step time", self._metric(row, "timing_s/step")],
            ["generation time", self._metric(row, "timing_s/gen")],
        ]

    def _recent_step_rows(self) -> list[list[Any]]:
        rows = []
        for row in self.recent_steps[-20:]:
            rows.append(
                [
                    row.get("step", ""),
                    row.get("epoch", ""),
                    self._metric(row, "episode/reward/mean", "critic/rewards/mean", "critic/score/mean"),
                    self._metric(row, "response_length/mean"),
                    self._metric(row, "response_length/clip_ratio"),
                    self._metric(row, "actor/entropy_loss"),
                    self._metric(row, "actor/pg_loss", "actor/loss"),
                    self._metric(row, "actor/ppo_kl", "actor/kl_loss", "actor/reward_kl_penalty"),
                    self._metric(row, "timing_s/step"),
                ]
            )
        return rows or [["", "", "", "", "", "", "", "", ""]]

    def _validation_summary_rows(self) -> list[list[Any]]:
        rows = []
        for row in self.validation_rows[-20:]:
            metrics = row.get("metrics", {})
            score_keys = [key for key in metrics if key.endswith("/test_score")]
            success_keys = [key for key in metrics if "success_rate" in key]
            mean_score = ""
            if score_keys:
                vals = []
                for key in score_keys:
                    try:
                        vals.append(float(metrics[key]))
                    except Exception:
                        pass
                if vals:
                    mean_score = sum(vals) / len(vals)
            rows.append(
                [
                    row.get("step", ""),
                    row.get("stage", ""),
                    mean_score,
                    "; ".join(f"{k}={_fmt(metrics[k])}" for k in score_keys[:5]),
                    "; ".join(f"{k}={_fmt(metrics[k])}" for k in success_keys[:5]),
                ]
            )
        return rows or [["", "", "", "", ""]]

    def _checkpoint_rows(self) -> list[list[Any]]:
        return [[x.get("step", ""), x.get("path", ""), x.get("time", "")] for x in self.checkpoints] or [["", "", ""]]

    def _render(self, message: str = "") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        status_rows = [
            ["status", self.status],
            ["message", message],
            ["started_at", self.started_at],
            ["updated_at", now],
            ["hostname", socket.gethostname()],
            ["pid", os.getpid()],
            ["train_size", self.train_size if self.train_size is not None else ""],
            ["val_size", self.val_size if self.val_size is not None else ""],
            ["total_training_steps", self.total_training_steps if self.total_training_steps is not None else ""],
            ["run_dir", self.run_dir],
        ]
        content = [
            "# SkillRL Experiment Record",
            "",
            "This document is generated automatically during training. The JSONL files in the same directory are the source of truth for interrupted runs.",
            "",
            "## Status",
            _make_table(["Field", "Value"], status_rows),
            "",
            "## Configuration",
            _make_table(["Parameter", "Value"], self._config_rows()),
            "",
            "## Latest Training Metrics",
            _make_table(["Metric", "Value"], self._latest_rows()),
            "",
            "## Recent Training Curve",
            _make_table(["Step", "Epoch", "Reward mean", "Response len", "Clip ratio", "Entropy", "Actor loss", "KL", "Step time"], self._recent_step_rows()),
            "",
            "## Validation History",
            _make_table(["Step", "Stage", "Mean test score", "Per-source scores", "Success rates"], self._validation_summary_rows()),
            "",
            "## Checkpoints",
            _make_table(["Step", "Path", "Time"], self._checkpoint_rows()),
            "",
            "## Files",
            _make_table(
                ["File", "Description"],
                [
                    [self.metrics_path.name, "Append-only training metrics"],
                    [self.validation_path.name, "Append-only validation metrics"],
                    [self.checkpoints_path.name, "Checkpoint events"],
                    [self.config_path.name, "Resolved Hydra config"],
                ],
            ),
            "",
        ]
        tmp = self.record_path.with_suffix(".md.tmp")
        tmp.write_text("\n".join(content), encoding="utf-8")
        tmp.replace(self.record_path)
