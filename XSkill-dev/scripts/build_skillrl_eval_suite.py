"""Build reusable SkillRL validation scripts for XSkill global_val.

The generated scripts evaluate the same global_val parquet across:
base / SFT / RL models, each with and without runtime SkillBank retrieval.
Evaluation uses SkillRL's multi-turn validation rollout path and keeps tools
enabled by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_GRPO_SCRIPT = ROOT / "scripts" / "build_skillrl_grpo_command.py"
PREPARE_GRPO_DATA_SCRIPT = ROOT / "scripts" / "prepare_skillrl_grpo_dataset.py"

DEFAULT_ENABLED_TOOLS = "web_search,visit,image_search,code_interpreter,zoom"


def _as_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return ROOT / value


def _portable_arg(path: str | Path) -> str:
    """Prefer XSkill-dev-relative paths for generated portable scripts."""

    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _bash_export_block(args: argparse.Namespace) -> str:
    required_tools = args.enabled_tools
    bocha_required = any(
        tool.strip() in {"web_search", "image_search"}
        for tool in required_tools.split(",")
    )
    lines = [
        "",
        "# XSkill tool backend defaults for evaluation.",
        'export SEARCH_API_PROVIDER="${SEARCH_API_PROVIDER:-bocha}"',
        'export IMAGE_SEARCH_PROVIDER="${IMAGE_SEARCH_PROVIDER:-bocha}"',
        'export VISIT_BACKEND="${VISIT_BACKEND:-local}"',
        f': "${{BOCHA_SEARCH_TIMEOUT:={args.bocha_search_timeout}}}"',
        f': "${{BOCHA_SEARCH_MAX_RETRIES:={args.bocha_search_max_retries}}}"',
        'export BOCHA_SEARCH_TIMEOUT BOCHA_SEARCH_MAX_RETRIES',
        f'export XSKILL_EVAL_REQUIRED_TOOLS="{required_tools}"',
    ]
    if bocha_required:
        lines.extend(
            [
                'if [[ -z "${BOCHA_API_KEY:-}" ]]; then',
                '  echo "BOCHA_API_KEY is required because web_search/image_search are enabled." >&2',
                '  echo "Run: export BOCHA_API_KEY=your_bocha_key" >&2',
                "  exit 1",
                "fi",
            ]
        )
    lines.extend(
        [
            'export BOCHA_API_KEY="${BOCHA_API_KEY:-}"',
            "",
            '"${PYTHON_BIN:-python3}" - <<\'PY\'',
            "import os, sys",
            "xskill_root = os.environ.get('XSKILL_REPO_ROOT')",
            "if xskill_root:",
            "    eval_root = os.path.join(xskill_root, 'eval')",
            "    for path in (xskill_root, eval_root):",
            "        if path not in sys.path:",
            "            sys.path.insert(0, path)",
            "required = [x.strip() for x in os.environ.get('XSKILL_EVAL_REQUIRED_TOOLS', '').split(',') if x.strip()]",
            "try:",
            "    from tools import list_tools",
            "except Exception as exc:",
            "    print(f'Tool preflight failed: cannot import tools: {exc}', file=sys.stderr)",
            "    raise SystemExit(1)",
            "available = set(list_tools())",
            "missing = [name for name in required if name not in available]",
            "print('Registered tools:', ', '.join(sorted(available)))",
            "if missing:",
            "    print('Missing enabled tools: ' + ', '.join(missing), file=sys.stderr)",
            "    print('Install tool dependencies or remove them from --enabled-tools.', file=sys.stderr)",
            "    raise SystemExit(1)",
            "PY",
            "",
        ]
    )
    return "\n".join(lines)


def _patch_generated_script(path: Path, args: argparse.Namespace) -> None:
    if args.dry_run or not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "\nexec "
    if "XSkill tool backend defaults for evaluation" in text or marker not in text:
        return
    text = text.replace(marker, _bash_export_block(args) + marker, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def _export_eval_parquet(args: argparse.Namespace, eval_parquet: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PREPARE_GRPO_DATA_SCRIPT),
        "--input-spec",
        _portable_arg(args.global_val_json),
        "--output-path",
        _portable_arg(eval_parquet),
        "--mixing-strategy",
        args.mixing_strategy,
        "--seed",
        str(args.seed),
    ]
    _run(cmd, dry_run=args.dry_run)
    return cmd


def _script_run_name(model_label: str, use_skill_bank: bool) -> str:
    suffix = "with_skill" if use_skill_bank else "no_skill"
    return f"{model_label}_{suffix}_val"


def _build_case_command(
    args: argparse.Namespace,
    *,
    run_name: str,
    model_path: str,
    eval_parquet: Path,
    script_path: Path,
    use_skill_bank: bool,
) -> list[str]:
    default_local_dir = f"checkpoints/xskill_eval/{run_name}"
    validation_data_dir = f"${{SKILLRL_REPO_ROOT}}/checkpoints/xskill_eval/{run_name}/validation_dump"
    rollout_data_dir = f"${{SKILLRL_REPO_ROOT}}/checkpoints/xskill_eval/{run_name}/rollout_dump"

    cmd = [
        sys.executable,
        str(BUILD_GRPO_SCRIPT),
        "--model-path",
        model_path,
        "--train-file",
        _portable_arg(eval_parquet),
        "--val-file",
        _portable_arg(eval_parquet),
        "--reward-mode",
        args.reward_mode,
        "--env-name",
        args.env_name,
        "--max-steps",
        str(args.max_steps),
        "--enabled-tools",
        args.enabled_tools,
        "--retrieval-mode",
        args.retrieval_mode,
        "--top-k",
        str(args.top_k),
        "--train-batch-size",
        str(args.val_batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--ppo-mini-batch-size",
        str(args.ppo_mini_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--group-size",
        str(args.group_size),
        "--val-rollout-n",
        str(args.val_rollout_n),
        "--val-temperature",
        str(args.val_temperature),
        "--n-gpus-per-node",
        str(args.n_gpus_per_node),
        "--nnodes",
        str(args.nnodes),
        "--model-dtype",
        args.model_dtype,
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-response-length",
        str(args.max_response_length),
        "--rollout-gpu-memory-utilization",
        str(args.rollout_gpu_memory_utilization),
        "--total-epochs",
        "1",
        "--save-freq",
        "-1",
        "--test-freq",
        "-1",
        "--val-only",
        "--val-before-train",
        "--resume-mode",
        "disable",
        "--validation-data-dir",
        validation_data_dir,
        "--rollout-data-dir",
        rollout_data_dir,
        "--log-val-generations",
        str(args.log_val_generations),
        "--project-name",
        args.project_name,
        "--experiment-name",
        run_name,
        "--default-local-dir",
        default_local_dir,
        "--output-script",
        str(script_path),
        "--cuda-visible-devices",
        args.cuda_visible_devices,
    ]
    if args.script_mode == "compute":
        cmd.extend(
            [
                "--compute-node-output-script",
                "--compute-conda-init",
                args.compute_conda_init,
                "--compute-conda-env",
                args.compute_conda_env,
                "--compute-monorepo-root",
                args.compute_monorepo_root,
            ]
        )
    else:
        cmd.extend(
            [
                "--no-compute-node-output-script",
                "--portable-output-script",
                "--python-executable",
                args.python_executable,
            ]
        )
    if args.preserve_proxy_env:
        cmd.append("--preserve-proxy-env")
    else:
        cmd.append("--no-preserve-proxy-env")
    if args.enable_tools:
        cmd.append("--enable-tools")
    else:
        cmd.append("--no-enable-tools")
    if args.val_do_sample:
        cmd.append("--val-do-sample")
    else:
        cmd.append("--no-val-do-sample")
    if args.rollout_max_model_len is not None:
        cmd.extend(["--rollout-max-model-len", str(args.rollout_max_model_len)])
    if args.rollout_max_num_batched_tokens is not None:
        cmd.extend(["--rollout-max-num-batched-tokens", str(args.rollout_max_num_batched_tokens)])
    if args.rollout_limit_images is not None:
        cmd.extend(["--rollout-limit-images", str(args.rollout_limit_images)])
    if args.task_specific_top_k is not None:
        cmd.extend(["--task-specific-top-k", str(args.task_specific_top_k)])
    if args.embedding_model_path:
        cmd.extend(["--embedding-model-path", args.embedding_model_path])
    if use_skill_bank:
        cmd.extend(["--skill-bank-json", _portable_arg(args.skill_bank_json)])
    return cmd


def _planned_models(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    return [
        ("base", args.base_model_path),
        ("sft", args.sft_model_path),
        ("rl", args.rl_model_path),
    ]


def _build_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _as_project_path(args.output_dir)
    eval_data_dir = _as_project_path(args.eval_data_dir)
    eval_parquet = eval_data_dir / args.eval_parquet_name
    manifest_path = output_dir / "eval_suite_manifest.json"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_data_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_data_export:
        data_export_command = []
        print(f"Skipping eval parquet export; expected file: {_portable_arg(eval_parquet)}")
    else:
        data_export_command = _export_eval_parquet(args, eval_parquet)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "global_val_json": _portable_arg(args.global_val_json),
        "eval_parquet": _portable_arg(eval_parquet),
        "skip_data_export": bool(args.skip_data_export),
        "data_export_command": data_export_command,
        "defaults": {
            "script_mode": args.script_mode,
            "python_executable": args.python_executable,
            "preserve_proxy_env": args.preserve_proxy_env,
            "bocha_search_timeout": args.bocha_search_timeout,
            "bocha_search_max_retries": args.bocha_search_max_retries,
            "n_gpus_per_node": args.n_gpus_per_node,
            "val_batch_size": args.val_batch_size,
            "group_size": args.group_size,
            "val_rollout_n": args.val_rollout_n,
            "val_do_sample": args.val_do_sample,
            "val_temperature": args.val_temperature,
            "max_steps": args.max_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "rollout_max_model_len": args.rollout_max_model_len,
            "rollout_max_num_batched_tokens": args.rollout_max_num_batched_tokens,
            "rollout_gpu_memory_utilization": args.rollout_gpu_memory_utilization,
            "log_val_generations": args.log_val_generations,
            "enable_tools": args.enable_tools,
            "enabled_tools": args.enabled_tools,
        },
        "cases": [],
        "skipped": [],
    }

    for model_label, model_path in _planned_models(args):
        if not model_path:
            manifest["skipped"].append(
                {
                    "model": model_label,
                    "reason": f"--{model_label}-model-path was not provided",
                }
            )
            continue

        for use_skill_bank in (False, True):
            run_name = _script_run_name(model_label, use_skill_bank)
            if use_skill_bank and not args.skill_bank_json:
                manifest["skipped"].append(
                    {
                        "case": run_name,
                        "reason": "--skill-bank-json was not provided",
                    }
                )
                continue

            script_path = output_dir / f"{run_name}.sh"
            command = _build_case_command(
                args,
                run_name=run_name,
                model_path=model_path,
                eval_parquet=eval_parquet,
                script_path=script_path,
                use_skill_bank=use_skill_bank,
            )
            _run(command, dry_run=args.dry_run)
            _patch_generated_script(script_path, args)
            manifest["cases"].append(
                {
                    "name": run_name,
                    "model_label": model_label,
                    "model_path": model_path,
                    "skill_bank_enabled": use_skill_bank,
                    "skill_bank_json": _portable_arg(args.skill_bank_json) if use_skill_bank else "",
                    "script_path": _portable_arg(script_path),
                    "default_local_dir": f"SkillRL/checkpoints/xskill_eval/{run_name}",
                    "validation_data_dir": f"SkillRL/checkpoints/xskill_eval/{run_name}/validation_dump",
                    "rollout_data_dir": f"SkillRL/checkpoints/xskill_eval/{run_name}/rollout_dump",
                    "command": command,
                }
            )

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Global Val SkillRL evaluation script suite.")
    parser.add_argument("--global-val-json", default="benchmark/_mixed_protocol/global_val.json")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--sft-model-path", default=None)
    parser.add_argument("--rl-model-path", default=None)
    parser.add_argument("--skill-bank-json", default=None)
    parser.add_argument("--output-dir", default="output/eval_runs")
    parser.add_argument("--eval-data-dir", default="output/eval_data")
    parser.add_argument("--eval-parquet-name", default="global_val_eval.parquet")
    parser.add_argument("--mixing-strategy", choices=["concat", "sqrt_size"], default="concat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-data-export",
        action="store_true",
        help="Only generate evaluation scripts and manifest; assume eval parquet already exists.",
    )

    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--n-gpus-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--val-rollout-n",
        type=int,
        default=None,
        help=(
            "Validation trajectories per sample. Defaults to --group-size. "
            "This is separate from --max-steps, which controls multi-turn dialogue length."
        ),
    )
    parser.add_argument("--val-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-temperature", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-prompt-length", type=int, default=16000)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--rollout-max-model-len", type=int, default=24576)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=24576)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--rollout-limit-images", type=int, default=None)
    parser.add_argument("--model-dtype", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"], default="bfloat16")

    parser.add_argument("--enable-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enabled-tools", default=DEFAULT_ENABLED_TOOLS)
    parser.add_argument("--reward-mode", choices=["contains", "exact"], default="contains")
    parser.add_argument("--env-name", default="xskill_visual")
    parser.add_argument("--retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--log-val-generations", type=int, default=20)
    parser.add_argument("--project-name", default="xskill_eval")

    parser.add_argument(
        "--script-mode",
        choices=["portable", "compute"],
        default="portable",
        help=(
            "portable writes scripts for a normal server and does not activate conda; "
            "compute writes scripts with conda activation and fixed compute-node roots."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default="python3",
        help="Python executable used inside portable generated scripts.",
    )
    parser.add_argument(
        "--preserve-proxy-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep HTTP_PROXY/HTTPS_PROXY/ALL_PROXY in generated scripts. This matches accumulation behavior.",
    )
    parser.add_argument("--bocha-search-timeout", type=int, default=60)
    parser.add_argument("--bocha-search-max-retries", type=int, default=3)
    parser.add_argument("--compute-conda-init", default="/data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh")
    parser.add_argument("--compute-conda-env", default="/data/home/scwb693/.conda/envs/skillrl")
    parser.add_argument("--compute-monorepo-root", default="/data/home/scwb693/run/luzy/XSKILLRL")

    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest without writing output files.")
    args = parser.parse_args()

    if args.val_rollout_n is None:
        args.val_rollout_n = args.group_size
    if args.val_rollout_n <= 0:
        raise SystemExit("--val-rollout-n must be positive")
    if args.max_steps <= 1 and args.enable_tools:
        raise SystemExit("--max-steps must be greater than 1 when tools are enabled for multi-turn evaluation")
    if args.n_gpus_per_node != 1:
        print("Warning: evaluation suite defaults to single-GPU evaluation; using requested n_gpus_per_node.")
    return args


def main() -> None:
    args = parse_args()
    _build_suite(args)


if __name__ == "__main__":
    main()
