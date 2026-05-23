"""Build a portable SkillRL/verl SFT launch script for XSkill warm start."""

from __future__ import annotations

import argparse
from pathlib import Path


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


def _sft_overrides(args, *, portable: bool) -> list[str]:
    train_file = _portable_path(args.train_file) if portable else args.train_file
    val_file = _portable_path(args.val_file) if portable else args.val_file
    output_dir = args.output_dir
    if portable and output_dir is None:
        output_dir = "${SKILLRL_REPO_ROOT}/checkpoints/xskill_sft/qwen3vl8b_sft"
    elif portable:
        output_dir = _portable_path(output_dir, root_var="SKILLRL_REPO_ROOT")

    return [
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "data.prompt_key=prompt",
        "data.response_key=response",
        "data.prompt_dict_keys=[]",
        "data.response_dict_keys=[]",
        f"data.max_length={args.max_length}",
        f"data.truncation={args.truncation}",
        f"model.partial_pretrain={args.model_path}",
        "model.trust_remote_code=True",
        f"model.attn_implementation={args.attn_implementation}",
        f"model.enable_gradient_checkpointing={str(args.enable_gradient_checkpointing)}",
        f"model.strategy={args.strategy}",
        f"use_remove_padding={str(args.use_remove_padding)}",
        f"optim.lr={args.learning_rate}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.total_training_steps={args.total_training_steps}",
        f"trainer.logger={args.logger}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.default_hdfs_dir=null",
    ]


def _write_script(target: Path, args) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    overrides = _sft_overrides(args, portable=args.portable_output_script)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export HYDRA_FULL_ERROR=1",
        "export TOKENIZERS_PARALLELISM=false",
        "unset ROCR_VISIBLE_DEVICES || true",
        "unset HIP_VISIBLE_DEVICES || true",
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'XSKILL_REPO_ROOT="${XSKILL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"',
        'MONOREPO_ROOT="${MONOREPO_ROOT:-$(cd "${XSKILL_REPO_ROOT}/.." && pwd)}"',
        'SKILLRL_REPO_ROOT="${SKILLRL_REPO_ROOT:-${MONOREPO_ROOT}/SkillRL}"',
        'export XSKILL_REPO_ROOT',
        'export PYTHONPATH="${SKILLRL_REPO_ROOT}:${XSKILL_REPO_ROOT}:${PYTHONPATH:-}"',
    ]
    if args.cuda_visible_devices:
        lines.append(f'export CUDA_VISIBLE_DEVICES="{args.cuda_visible_devices}"')
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
            (
                f"exec torchrun --standalone --nnodes=1 --nproc_per_node={args.nproc_per_node} "
                f"-m verl.trainer.fsdp_sft_trainer \"${{ARGS[@]}}\""
            ),
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable SkillRL SFT launch script.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-script", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--truncation", choices=["error", "left", "right"], default="right")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--total-training-steps", default="null")
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--strategy", choices=["fsdp", "fsdp2"], default="fsdp2")
    parser.add_argument("--enable-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-remove-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project-name", default="xskill_skillrl")
    parser.add_argument("--experiment-name", default="qwen3vl8b_sft")
    parser.add_argument("--logger", default="['console']")
    parser.add_argument("--portable-output-script", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _write_script(Path(args.output_script), args)
    command = "torchrun --standalone --nnodes=1 --nproc_per_node={} -m verl.trainer.fsdp_sft_trainer {}".format(
        args.nproc_per_node,
        " ".join(_bash_quote(override) for override in _sft_overrides(args, portable=args.portable_output_script)),
    )
    print(command)


if __name__ == "__main__":
    main()
