"""Build a SkillRL/verl GRPO command for an exported XSkill dataset."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
import sys
from dataclasses import fields, replace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.skillrl.config import SkillRLGRPOConfig


def _portable_path(path: str | None, *, root_var: str = "XSKILL_REPO_ROOT") -> str | None:
    if not path:
        return path
    text = str(path)
    if text.startswith("$") or text.startswith("${") or Path(text).is_absolute():
        return text
    return "${" + root_var + "}/" + text.replace("\\", "/")


def _bash_quote(value: str) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{text}"'


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _write_portable_script(
    target: Path,
    *,
    config: SkillRLGRPOConfig,
    python_executable: str,
    cuda_visible_devices: str | None = None,
    extra_overrides: list[str] | None = None,
    preserve_proxy_env: bool = False,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    overrides = [*config.to_hydra_overrides(), *(extra_overrides or [])]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export HYDRA_FULL_ERROR=1",
        "export TOKENIZERS_PARALLELISM=false",
        "unset ROCR_VISIBLE_DEVICES || true",
        "unset HIP_VISIBLE_DEVICES || true",
        (
            "# Preserve proxy environment for search/visit APIs."
            if preserve_proxy_env
            else "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true"
        ),
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'XSKILL_REPO_ROOT="${XSKILL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"',
        'MONOREPO_ROOT="${MONOREPO_ROOT:-$(cd "${XSKILL_REPO_ROOT}/.." && pwd)}"',
        'SKILLRL_REPO_ROOT="${SKILLRL_REPO_ROOT:-${MONOREPO_ROOT}/SkillRL}"',
        'if [[ -z "${XSKILL_IMAGE_ROOT:-}" ]]; then',
        '  if [[ -d "${MONOREPO_ROOT}/images" ]]; then',
        '    XSKILL_IMAGE_ROOT="${MONOREPO_ROOT}/images"',
        "  else",
        '    XSKILL_IMAGE_ROOT="${XSKILL_REPO_ROOT}"',
        "  fi",
        "fi",
        'export XSKILL_REPO_ROOT XSKILL_IMAGE_ROOT',
        'export PYTHONPATH="${XSKILL_REPO_ROOT}:${XSKILL_REPO_ROOT}/eval:${PYTHONPATH:-}"',
        ': "${XSKILL_RL_IMAGE_MAX_PIXELS:=1048576}"',
        ': "${XSKILL_RL_IMAGE_MIN_MAX_PIXELS:=262144}"',
        ': "${XSKILL_RL_IMAGE_MIN_PIXELS:=16384}"',
        'export XSKILL_RL_IMAGE_MAX_PIXELS XSKILL_RL_IMAGE_MIN_MAX_PIXELS XSKILL_RL_IMAGE_MIN_PIXELS',
    ]
    if cuda_visible_devices:
        lines.append(f'export CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"')
    lines.extend(
        [
            "",
            'if [[ ! -d "${SKILLRL_REPO_ROOT}" ]]; then',
            '  echo "SkillRL repo not found: ${SKILLRL_REPO_ROOT}" >&2',
            "  exit 1",
            "fi",
            'cd "${SKILLRL_REPO_ROOT}"',
            "",
            "ARGS=(",
        ]
    )
    for override in overrides:
        lines.append(f"  {_bash_quote(override)}")
    lines.extend(
        [
            ")",
            "",
            f"exec {python_executable} -m verl.trainer.main_ppo \"${{ARGS[@]}}\"",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_compute_node_script(
    target: Path,
    *,
    config: SkillRLGRPOConfig,
    cuda_visible_devices: str | None = None,
    conda_init: str,
    conda_env: str,
    monorepo_root: str,
    model_path: str,
    extra_overrides: list[str] | None = None,
    preserve_proxy_env: bool = False,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    script_config = replace(config, model_path="${MODEL_PATH}")
    overrides = [*script_config.to_hydra_overrides(), *(extra_overrides or [])]
    xskill_root = "${XSKILL_REPO_ROOT}"
    skillrl_root = "${SKILLRL_REPO_ROOT}"
    image_root = "${XSKILL_IMAGE_ROOT}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export HYDRA_FULL_ERROR=1",
        "export TOKENIZERS_PARALLELISM=false",
        "export PYTHONNOUSERSITE=1",
        "",
        'CONDA_INIT_SCRIPT="${CONDA_INIT_SCRIPT:-' + conda_init + '}"',
        'if [[ -n "${CONDA_INIT_SCRIPT}" && -f "${CONDA_INIT_SCRIPT}" ]]; then',
        '  source "${CONDA_INIT_SCRIPT}"',
        "elif command -v conda >/dev/null 2>&1; then",
        '  eval "$(conda shell.bash hook)"',
        "else",
        '  echo "conda is not available; set CONDA_INIT_SCRIPT or load conda before running this script" >&2',
        "  exit 1",
        "fi",
        f"conda activate {conda_env}",
        'PYTHON_BIN="$(command -v python)"',
        "",
        'echo "CONDA_PREFIX=${CONDA_PREFIX:-}"',
        'echo "PYTHON_BIN=${PYTHON_BIN}"',
        '"${PYTHON_BIN}" - <<\'PY\'',
        "import sys, torch",
        'print("sys.executable:", sys.executable)',
        'print("torch:", torch.__version__)',
        'print("torch file:", torch.__file__)',
        "PY",
        '"${PYTHON_BIN}" -m pip --version',
        "",
        "unset ROCR_VISIBLE_DEVICES || true",
        "unset HIP_VISIBLE_DEVICES || true",
        (
            "# Preserve proxy environment for search/visit APIs."
            if preserve_proxy_env
            else "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true"
        ),
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'DEFAULT_XSKILL_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
        'DEFAULT_MONOREPO_ROOT="$(cd "${DEFAULT_XSKILL_REPO_ROOT}/.." && pwd)"',
        (
            f'export MONOREPO_ROOT="${{MONOREPO_ROOT:-{monorepo_root}}}"'
            if monorepo_root
            else 'export MONOREPO_ROOT="${MONOREPO_ROOT:-${DEFAULT_MONOREPO_ROOT}}"'
        ),
        'export XSKILL_REPO_ROOT="${XSKILL_REPO_ROOT:-${MONOREPO_ROOT}/XSkill-dev}"',
        'export SKILLRL_REPO_ROOT="${SKILLRL_REPO_ROOT:-${MONOREPO_ROOT}/SkillRL}"',
        'if [[ -z "${XSKILL_IMAGE_ROOT:-}" ]]; then',
        '  if [[ -d "${MONOREPO_ROOT}/images" ]]; then',
        '    XSKILL_IMAGE_ROOT="${MONOREPO_ROOT}/images"',
        "  else",
        '    XSKILL_IMAGE_ROOT="${XSKILL_REPO_ROOT}"',
        "  fi",
        "fi",
        f'export MODEL_PATH="${{MODEL_PATH:-{model_path}}}"',
        f'export PYTHONPATH="{skillrl_root}:{xskill_root}:{xskill_root}/eval:${{PYTHONPATH:-}}"',
        ': "${XSKILL_RL_IMAGE_MAX_PIXELS:=1048576}"',
        ': "${XSKILL_RL_IMAGE_MIN_MAX_PIXELS:=262144}"',
        ': "${XSKILL_RL_IMAGE_MIN_PIXELS:=16384}"',
        'export XSKILL_RL_IMAGE_MAX_PIXELS XSKILL_RL_IMAGE_MIN_MAX_PIXELS XSKILL_RL_IMAGE_MIN_PIXELS',
        "",
    ]
    if cuda_visible_devices:
        lines.append(f'export CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"')
    else:
        lines.extend(
            [
                'if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then',
                '  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"',
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            'if [[ ! -d "${SKILLRL_REPO_ROOT}" ]]; then',
            '  echo "SkillRL repo not found: ${SKILLRL_REPO_ROOT}" >&2',
            "  exit 1",
            "fi",
            "",
            "REQUIRED_PATHS=(",
            f'  "{xskill_root}/{_strip_xskill_root(config.train_file)}"',
            f'  "{xskill_root}/{_strip_xskill_root(config.val_file)}"',
        ]
    )
    if config.skill_bank_json:
        lines.append(f'  "{xskill_root}/{_strip_xskill_root(config.skill_bank_json)}"')
    lines.append('  "${MODEL_PATH}"')
    lines.append(")")
    lines.extend(
        [
            'for p in "${REQUIRED_PATHS[@]}"; do',
            '  if [[ ! -e "$p" ]]; then',
            '    echo "Required path not found: $p" >&2',
            "    exit 1",
            "  fi",
            "done",
            "",
            'cd "${SKILLRL_REPO_ROOT}"',
            "",
            "ARGS=(",
        ]
    )
    for override in overrides:
        lines.append(f"  {_bash_quote(override)}")
    lines.extend(
        [
            ")",
            "",
            'exec "${PYTHON_BIN}" -u -m verl.trainer.main_ppo "${ARGS[@]}"',
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _strip_xskill_root(path: str) -> str:
    text = str(path).replace("\\", "/")
    for prefix in ("${XSKILL_REPO_ROOT}/", "$XSKILL_REPO_ROOT/"):
        if text.startswith(prefix):
            return text[len(prefix):]
    marker = "/XSkill-dev/"
    if marker in text:
        return text.split(marker, 1)[1]
    return text.lstrip("/")


def _make_grpo_config(**kwargs) -> SkillRLGRPOConfig:
    valid_names = {field.name for field in fields(SkillRLGRPOConfig)}
    filtered = {key: value for key, value in kwargs.items() if key in valid_names}
    return SkillRLGRPOConfig(**filtered)


def _has_config_field(name: str) -> bool:
    return name in {field.name for field in fields(SkillRLGRPOConfig)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--skill-bank-json", default=None)
    parser.add_argument("--xskill-repo-root", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--reward-mode", choices=["contains", "exact"], default="contains")
    parser.add_argument("--env-name", default="xskill_visual")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum multi-turn environment steps per rollout. Tool-use rollouts need this > 1.",
    )
    parser.add_argument(
        "--enable-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable XSkill tool execution during SkillRL environment rollouts.",
    )
    parser.add_argument(
        "--enabled-tools",
        default=None,
        help="Comma-separated tool allowlist, e.g. web_search,visit,image_search,code_interpreter,zoom.",
    )
    parser.add_argument("--retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--enable-dynamic-update", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help=(
            "Target PPO gradient accumulation steps. In this SkillRL/verl actor, "
            "gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu."
        ),
    )
    parser.add_argument(
        "--ppo-micro-batch-size-per-gpu",
        type=int,
        default=None,
        help=(
            "Override PPO micro batch size per GPU. If omitted, it is computed as "
            "ppo_mini_batch_size // gradient_accumulation_steps."
        ),
    )
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-gpus-per-node", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--model-dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16")
    parser.add_argument("--max-prompt-length", type=int, default=6000)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--rollout-max-model-len", type=int, default=None)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--rollout-limit-images", type=int, default=None)
    parser.add_argument("--actor-ppo-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--rollout-log-prob-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--ref-log-prob-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--test-freq", type=int, default=5)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--val-before-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-mode", default="auto")
    parser.add_argument(
        "--val-rollout-n",
        type=int,
        default=None,
        help=(
            "Number of validation trajectories sampled per input. This is not the "
            "multi-turn dialogue limit; use --max-steps for that."
        ),
    )
    parser.add_argument(
        "--val-do-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override actor_rollout_ref.rollout.val_kwargs.do_sample for validation.",
    )
    parser.add_argument("--val-temperature", type=float, default=None)
    parser.add_argument("--validation-data-dir", default=None)
    parser.add_argument("--validation-trajectory-dir", default=None)
    parser.add_argument("--rollout-data-dir", default=None)
    parser.add_argument("--log-val-generations", type=int, default=0)
    parser.add_argument("--project-name", default="xskill_skillrl")
    parser.add_argument("--experiment-name", default="xskill_grpo_skills")
    parser.add_argument("--default-local-dir", default=None)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--output-script", default=None)
    parser.add_argument(
        "--portable-output-script",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --output-script is used, write a bash script that resolves repo paths at runtime.",
    )
    parser.add_argument(
        "--compute-node-output-script",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --output-script is used, write a compute-node bash script with "
            "conda activation. Portable scripts are the default."
        ),
    )
    parser.add_argument("--compute-conda-init", default="")
    parser.add_argument("--compute-conda-env", default="skillrl")
    parser.add_argument("--compute-monorepo-root", default="")
    parser.add_argument(
        "--preserve-proxy-env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not unset HTTP_PROXY/HTTPS_PROXY/ALL_PROXY in generated bash scripts.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES line to embed in the generated bash script.",
    )
    args = parser.parse_args()

    if args.val_only:
        if args.n_gpus_per_node is None:
            args.n_gpus_per_node = 1
        if args.cuda_visible_devices is None:
            args.cuda_visible_devices = "0"
        if args.enable_tools and args.max_steps <= 1:
            raise SystemExit("--max-steps must be greater than 1 for tool-enabled multi-turn validation")
    elif args.n_gpus_per_node is None:
        args.n_gpus_per_node = 8
    if args.val_rollout_n is not None and args.val_rollout_n <= 0:
        raise SystemExit("--val-rollout-n must be positive")

    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.ppo_micro_batch_size_per_gpu is None:
        if args.ppo_mini_batch_size % args.gradient_accumulation_steps != 0:
            raise SystemExit(
                "--ppo-mini-batch-size must be divisible by --gradient-accumulation-steps "
                "when --ppo-micro-batch-size-per-gpu is omitted"
            )
        args.ppo_micro_batch_size_per_gpu = (
            args.ppo_mini_batch_size // args.gradient_accumulation_steps
        )
    actual_grad_accum = args.ppo_mini_batch_size // args.ppo_micro_batch_size_per_gpu
    if args.ppo_mini_batch_size % args.ppo_micro_batch_size_per_gpu != 0:
        raise SystemExit("--ppo-mini-batch-size must be divisible by --ppo-micro-batch-size-per-gpu")
    if actual_grad_accum != args.gradient_accumulation_steps:
        raise SystemExit(
            "Requested gradient accumulation steps does not match PPO sizes: "
            f"{args.ppo_mini_batch_size} // {args.ppo_micro_batch_size_per_gpu} = {actual_grad_accum}, "
            f"expected {args.gradient_accumulation_steps}"
        )

    portable_paths = args.portable_output_script or args.compute_node_output_script
    if portable_paths:
        train_file = _portable_path(args.train_file)
        val_file = _portable_path(args.val_file)
        skill_bank_json = _portable_path(args.skill_bank_json)
        xskill_repo_root = args.xskill_repo_root or "${XSKILL_REPO_ROOT}"
        image_root = args.image_root or "${XSKILL_IMAGE_ROOT}"
        default_local_dir = args.default_local_dir or (
            f"${{SKILLRL_REPO_ROOT}}/checkpoints/{args.project_name}/{args.experiment_name}"
        )
        if args.default_local_dir is not None:
            default_local_dir = _portable_path(args.default_local_dir, root_var="SKILLRL_REPO_ROOT")
    else:
        train_file = args.train_file
        val_file = args.val_file
        skill_bank_json = args.skill_bank_json
        xskill_repo_root = args.xskill_repo_root
        image_root = args.image_root
        default_local_dir = args.default_local_dir

    validation_data_dir = _portable_path(args.validation_data_dir) if portable_paths else args.validation_data_dir
    validation_trajectory_dir = (
        _portable_path(args.validation_trajectory_dir) if portable_paths else args.validation_trajectory_dir
    )
    rollout_data_dir = _portable_path(args.rollout_data_dir) if portable_paths else args.rollout_data_dir
    extra_overrides = []
    if not _has_config_field("val_before_train"):
        extra_overrides.append(f"trainer.val_before_train={str(args.val_before_train)}")
    if not _has_config_field("val_only"):
        extra_overrides.append(f"trainer.val_only={str(args.val_only)}")
    if not _has_config_field("resume_mode"):
        extra_overrides.append(f"trainer.resume_mode={args.resume_mode}")
    if not _has_config_field("log_val_generations"):
        extra_overrides.append(f"trainer.log_val_generations={args.log_val_generations}")
    if validation_data_dir and not _has_config_field("validation_data_dir"):
        extra_overrides.append(f"trainer.validation_data_dir={validation_data_dir}")
    if validation_trajectory_dir and not _has_config_field("validation_trajectory_dir"):
        extra_overrides.append(f"trainer.validation_trajectory_dir={validation_trajectory_dir}")
    if rollout_data_dir and not _has_config_field("rollout_data_dir"):
        extra_overrides.append(f"trainer.rollout_data_dir={rollout_data_dir}")

    config = _make_grpo_config(
        model_path=args.model_path,
        train_file=train_file,
        val_file=val_file,
        skill_bank_json=skill_bank_json,
        xskill_repo_root=xskill_repo_root,
        image_root=image_root,
        reward_mode=args.reward_mode,
        env_name=args.env_name,
        max_steps=args.max_steps,
        enable_tools=args.enable_tools,
        enabled_tools=args.enabled_tools,
        retrieval_mode=args.retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        top_k=args.top_k,
        task_specific_top_k=args.task_specific_top_k,
        enable_dynamic_update=args.enable_dynamic_update,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        group_size=args.group_size,
        model_dtype=args.model_dtype,
        gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_max_model_len=args.rollout_max_model_len,
        rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        rollout_limit_images=args.rollout_limit_images,
        actor_ppo_max_token_len_per_gpu=args.actor_ppo_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=args.rollout_log_prob_max_token_len_per_gpu,
        ref_log_prob_max_token_len_per_gpu=args.ref_log_prob_max_token_len_per_gpu,
        n_gpus_per_node=args.n_gpus_per_node,
        nnodes=args.nnodes,
        total_epochs=args.total_epochs,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        val_before_train=args.val_before_train,
        val_only=args.val_only,
        resume_mode=args.resume_mode,
        val_rollout_n=args.val_rollout_n,
        val_do_sample=args.val_do_sample,
        val_temperature=args.val_temperature,
        validation_data_dir=validation_data_dir,
        validation_trajectory_dir=validation_trajectory_dir,
        rollout_data_dir=rollout_data_dir,
        log_val_generations=args.log_val_generations,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        default_local_dir=default_local_dir,
    )
    command = " ".join(
        [
            config.to_command(python_executable=args.python_executable),
            *(_quote(item) for item in extra_overrides),
        ]
    )
    if args.output_script:
        target = Path(args.output_script)
        if args.compute_node_output_script:
            _write_compute_node_script(
                target,
                config=config,
                cuda_visible_devices=args.cuda_visible_devices,
                conda_init=args.compute_conda_init,
                conda_env=args.compute_conda_env,
                monorepo_root=args.compute_monorepo_root,
                model_path=args.model_path,
                extra_overrides=extra_overrides,
                preserve_proxy_env=args.preserve_proxy_env,
            )
        elif args.portable_output_script:
            _write_portable_script(
                target,
                config=config,
                python_executable=args.python_executable,
                cuda_visible_devices=args.cuda_visible_devices,
                extra_overrides=extra_overrides,
                preserve_proxy_env=args.preserve_proxy_env,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(command + "\n", encoding="utf-8", newline="\n")
    print(command)


if __name__ == "__main__":
    main()
