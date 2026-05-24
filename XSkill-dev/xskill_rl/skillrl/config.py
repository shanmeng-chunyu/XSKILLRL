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
    max_steps: int = 20
    enable_tools: bool = True
    enabled_tools: Optional[str] = None
    retrieval_mode: str = "template"
    embedding_model_path: Optional[str] = None
    top_k: int = 6
    task_specific_top_k: Optional[int] = None
    enable_dynamic_update: bool = False
    update_threshold: float = 0.4
    max_new_skills: int = 3
    train_batch_size: int = 16
    val_batch_size: int = 16
    group_size: int = 8
    learning_rate: float = 1e-6
    ppo_mini_batch_size: int = 16
    ppo_micro_batch_size_per_gpu: int = 4
    log_prob_micro_batch_size_per_gpu: int = 8
    tensor_model_parallel_size: int = 1
    rollout_engine: str = "vllm"
    model_dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.7
    rollout_max_model_len: Optional[int] = None
    rollout_max_num_batched_tokens: Optional[int] = None
    rollout_limit_images: Optional[int] = None
    kl_loss_coef: float = 0.01
    invalid_action_penalty_coef: float = 0.1
    max_prompt_length: int = 6000
    max_response_length: int = 1024
    actor_ppo_max_token_len_per_gpu: Optional[int] = None
    rollout_log_prob_max_token_len_per_gpu: Optional[int] = None
    ref_log_prob_max_token_len_per_gpu: Optional[int] = None
    total_epochs: int = 150
    save_freq: int = 10
    test_freq: int = 5
    val_before_train: bool = True
    val_only: bool = False
    resume_mode: str = "auto"
    validation_data_dir: Optional[str] = None
    rollout_data_dir: Optional[str] = None
    log_val_generations: int = 0
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
            f"+actor_rollout_ref.actor.fsdp_config.model_dtype={self.model_dtype}",
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
            f"actor_rollout_ref.rollout.dtype={self.model_dtype}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            (
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
                f"{self.log_prob_micro_batch_size_per_gpu}"
            ),
            f"+actor_rollout_ref.ref.fsdp_config.model_dtype={self.model_dtype}",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.use_invalid_action_penalty=True",
            (
                "actor_rollout_ref.actor.invalid_action_penalty_coef="
                f"{self.invalid_action_penalty_coef}"
            ),
            "algorithm.use_kl_in_reward=False",
            f"env.env_name={self.env_name}",
            f"env.max_steps={self.max_steps}",
            f"env.rollout.n={self.group_size}",
            f"+env.xskill.reward_mode={self.reward_mode}",
            f"+env.xskill.enable_tools={str(self.enable_tools)}",
            "trainer.critic_warmup=0",
            f"trainer.logger={self.logger}",
            f"trainer.project_name={self.project_name}",
            f"trainer.experiment_name={self.experiment_name}",
            f"trainer.n_gpus_per_node={self.n_gpus_per_node}",
            f"trainer.nnodes={self.nnodes}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.test_freq={self.test_freq}",
            f"trainer.total_epochs={self.total_epochs}",
            f"trainer.val_before_train={str(self.val_before_train)}",
            f"trainer.val_only={str(self.val_only)}",
            f"trainer.resume_mode={self.resume_mode}",
            f"trainer.log_val_generations={self.log_val_generations}",
        ]
        if self.validation_data_dir:
            overrides.append(f"trainer.validation_data_dir={self.validation_data_dir}")
        if self.rollout_data_dir:
            overrides.append(f"trainer.rollout_data_dir={self.rollout_data_dir}")

        if self.actor_ppo_max_token_len_per_gpu is not None:
            overrides.append(
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu="
                f"{self.actor_ppo_max_token_len_per_gpu}"
            )
        if self.rollout_max_model_len is not None:
            overrides.append(f"actor_rollout_ref.rollout.max_model_len={self.rollout_max_model_len}")
        if self.rollout_max_num_batched_tokens is not None:
            overrides.append(
                "actor_rollout_ref.rollout.max_num_batched_tokens="
                f"{self.rollout_max_num_batched_tokens}"
            )
        if self.rollout_limit_images is not None:
            overrides.append(f"+actor_rollout_ref.rollout.limit_images={self.rollout_limit_images}")
        if self.rollout_log_prob_max_token_len_per_gpu is not None:
            overrides.append(
                "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="
                f"{self.rollout_log_prob_max_token_len_per_gpu}"
            )
        if self.ref_log_prob_max_token_len_per_gpu is not None:
            overrides.append(
                "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="
                f"{self.ref_log_prob_max_token_len_per_gpu}"
            )

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
        if self.enabled_tools:
            tools = self.enabled_tools.replace("'", "\\'")
            overrides.append(f"+env.xskill.enabled_tools='{tools}'")
        if self.default_local_dir:
            overrides.append(f"trainer.default_local_dir={self.default_local_dir}")
        return overrides

    def to_command(self, python_executable: str = "python3") -> str:
        parts = [python_executable, "-m", "verl.trainer.main_ppo"]
        parts.extend(_quote(item) for item in self.to_hydra_overrides())
        return " ".join(parts)
