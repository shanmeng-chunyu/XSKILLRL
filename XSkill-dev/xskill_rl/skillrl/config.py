"""SkillRL/verl GRPO command configuration helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import List, Optional


def _quote(value: str) -> str:
    return shlex.quote(str(value))


@dataclass
class SkillRLGRPOConfig:
    """Hydra override builder following SkillRL's GRPO scripts."""

    model_path: str
    train_file: str
    val_file: str
    skill_bank_json: Optional[str] = None
    xskill_repo_root: Optional[str] = None
    image_root: Optional[str] = None
    reward_mode: str = "contains"
    env_name: str = "xskill_visual"
    retrieval_mode: str = "template"
    embedding_model_path: Optional[str] = None
    top_k: int = 6
    task_specific_top_k: Optional[int] = None
    enable_dynamic_update: bool = False
    update_threshold: float = 0.4
    max_new_skills: int = 3
    train_batch_size: int = 64
    val_batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 1e-6
    ppo_mini_batch_size: int = 64
    ppo_micro_batch_size_per_gpu: int = 4
    log_prob_micro_batch_size_per_gpu: int = 8
    tensor_model_parallel_size: int = 1
    rollout_engine: str = "vllm"
    gpu_memory_utilization: float = 0.7
    kl_loss_coef: float = 0.01
    invalid_action_penalty_coef: float = 0.1
    max_prompt_length: int = 6000
    max_response_length: int = 1024
    total_epochs: int = 150
    save_freq: int = 10
    test_freq: int = 5
    n_gpus_per_node: int = 8
    nnodes: int = 1
    project_name: str = "xskill_skillrl"
    experiment_name: str = "xskill_grpo_skills"
    logger: str = "['console']"
    default_local_dir: Optional[str] = None

    def to_hydra_overrides(self) -> List[str]:
        overrides = [
            "algorithm.adv_estimator=grpo",
            f"data.train_files={self.train_file}",
            f"data.val_files={self.val_file}",
            f"data.train_batch_size={self.train_batch_size}",
            f"data.val_batch_size={self.val_batch_size}",
            f"data.max_prompt_length={self.max_prompt_length}",
            f"data.max_response_length={self.max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=left",
            "data.return_raw_chat=True",
            f"actor_rollout_ref.model.path={self.model_path}",
            f"actor_rollout_ref.actor.optim.lr={self.learning_rate}",
            "actor_rollout_ref.model.use_remove_padding=True",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.ppo_mini_batch_size}",
            (
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
                f"{self.ppo_micro_batch_size_per_gpu}"
            ),
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            (
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
                f"{self.log_prob_micro_batch_size_per_gpu}"
            ),
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.tensor_model_parallel_size}",
            f"actor_rollout_ref.rollout.name={self.rollout_engine}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            (
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
                f"{self.log_prob_micro_batch_size_per_gpu}"
            ),
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.use_invalid_action_penalty=True",
            (
                "actor_rollout_ref.actor.invalid_action_penalty_coef="
                f"{self.invalid_action_penalty_coef}"
            ),
            "algorithm.use_kl_in_reward=False",
            f"env.env_name={self.env_name}",
            "env.max_steps=1",
            f"env.rollout.n={self.group_size}",
            f"+env.xskill.reward_mode={self.reward_mode}",
            "trainer.critic_warmup=0",
            f"trainer.logger={self.logger}",
            f"trainer.project_name={self.project_name}",
            f"trainer.experiment_name={self.experiment_name}",
            f"trainer.n_gpus_per_node={self.n_gpus_per_node}",
            f"trainer.nnodes={self.nnodes}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.test_freq={self.test_freq}",
            f"trainer.total_epochs={self.total_epochs}",
        ]

        if self.skill_bank_json:
            overrides.extend(
                [
                    "+env.use_skills_only_memory=True",
                    f"+env.skills_only_memory.skills_json_path={self.skill_bank_json}",
                    f"+env.skills_only_memory.retrieval_mode={self.retrieval_mode}",
                    f"+env.skills_only_memory.top_k={self.top_k}",
                    (
                        "+env.skills_only_memory.enable_dynamic_update="
                        f"{str(self.enable_dynamic_update)}"
                    ),
                    f"+env.skills_only_memory.update_threshold={self.update_threshold}",
                    f"+env.skills_only_memory.max_new_skills={self.max_new_skills}",
                ]
            )
            if self.embedding_model_path:
                overrides.append(
                    f"+env.skills_only_memory.embedding_model_path={self.embedding_model_path}"
                )
            if self.task_specific_top_k is not None:
                overrides.append(
                    f"+env.skills_only_memory.task_specific_top_k={self.task_specific_top_k}"
                )

        if self.xskill_repo_root:
            overrides.append(f"+env.xskill.repo_root={self.xskill_repo_root}")
        if self.image_root:
            overrides.append(f"+env.xskill.image_root={self.image_root}")
        if self.default_local_dir:
            overrides.append(f"trainer.default_local_dir={self.default_local_dir}")
        return overrides

    def to_command(self, python_executable: str = "python3") -> str:
        parts = [python_executable, "-m", "verl.trainer.main_ppo"]
        parts.extend(_quote(item) for item in self.to_hydra_overrides())
        return " ".join(parts)
