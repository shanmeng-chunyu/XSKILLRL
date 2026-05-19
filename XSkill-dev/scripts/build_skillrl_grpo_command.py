"""Build a SkillRL/verl GRPO command for an exported XSkill dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def _write_portable_script(
    target: Path,
    *,
    config: SkillRLGRPOConfig,
    python_executable: str,
    cuda_visible_devices: str | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    overrides = config.to_hydra_overrides()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'XSKILL_REPO_ROOT="${XSKILL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"',
        'MONOREPO_ROOT="${MONOREPO_ROOT:-$(cd "${XSKILL_REPO_ROOT}/.." && pwd)}"',
        'SKILLRL_REPO_ROOT="${SKILLRL_REPO_ROOT:-${MONOREPO_ROOT}/SkillRL}"',
        'XSKILL_IMAGE_ROOT="${XSKILL_IMAGE_ROOT:-${XSKILL_REPO_ROOT}}"',
        'export XSKILL_REPO_ROOT XSKILL_IMAGE_ROOT',
        'export PYTHONPATH="${XSKILL_REPO_ROOT}:${PYTHONPATH:-}"',
        ': "${XSKILL_RL_IMAGE_MAX_PIXELS:=1048576}"',
        ': "${XSKILL_RL_IMAGE_MIN_MAX_PIXELS:=262144}"',
        ': "${XSKILL_RL_IMAGE_MIN_PIXELS:=16384}"',
        'export XSKILL_RL_IMAGE_MAX_PIXELS XSKILL_RL_IMAGE_MIN_MAX_PIXELS XSKILL_RL_IMAGE_MIN_PIXELS',
    ]
    if cuda_visible_devices:
        lines.append(f'export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"')
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
    target.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--enable-dynamic-update", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--model-dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16")
    parser.add_argument("--rollout-max-model-len", type=int, default=None)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--rollout-limit-images", type=int, default=None)
    parser.add_argument("--actor-ppo-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--rollout-log-prob-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--ref-log-prob-max-token-len-per-gpu", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=150)
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
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES line to embed in the generated bash script.",
    )
    args = parser.parse_args()

    if args.portable_output_script:
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

    config = SkillRLGRPOConfig(
        model_path=args.model_path,
        train_file=train_file,
        val_file=val_file,
        skill_bank_json=skill_bank_json,
        xskill_repo_root=xskill_repo_root,
        image_root=image_root,
        reward_mode=args.reward_mode,
        env_name=args.env_name,
        retrieval_mode=args.retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        top_k=args.top_k,
        task_specific_top_k=args.task_specific_top_k,
        enable_dynamic_update=args.enable_dynamic_update,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        group_size=args.group_size,
        model_dtype=args.model_dtype,
        rollout_max_model_len=args.rollout_max_model_len,
        rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        rollout_limit_images=args.rollout_limit_images,
        actor_ppo_max_token_len_per_gpu=args.actor_ppo_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=args.rollout_log_prob_max_token_len_per_gpu,
        ref_log_prob_max_token_len_per_gpu=args.ref_log_prob_max_token_len_per_gpu,
        n_gpus_per_node=args.n_gpus_per_node,
        nnodes=args.nnodes,
        total_epochs=args.total_epochs,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        default_local_dir=default_local_dir,
    )
    command = config.to_command(python_executable=args.python_executable)
    if args.output_script:
        target = Path(args.output_script)
        if args.portable_output_script:
            _write_portable_script(
                target,
                config=config,
                python_executable=args.python_executable,
                cuda_visible_devices=args.cuda_visible_devices,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(command + "\n", encoding="utf-8")
    print(command)


if __name__ == "__main__":
    main()
