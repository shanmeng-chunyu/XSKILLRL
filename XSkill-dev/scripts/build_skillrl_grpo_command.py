"""Build a SkillRL/verl GRPO command for an exported XSkill dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.skillrl.config import SkillRLGRPOConfig


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
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--project-name", default="xskill_skillrl")
    parser.add_argument("--experiment-name", default="xskill_grpo_skills")
    parser.add_argument("--default-local-dir", default=None)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--output-script", default=None)
    args = parser.parse_args()

    config = SkillRLGRPOConfig(
        model_path=args.model_path,
        train_file=args.train_file,
        val_file=args.val_file,
        skill_bank_json=args.skill_bank_json,
        xskill_repo_root=args.xskill_repo_root,
        image_root=args.image_root,
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
        n_gpus_per_node=args.n_gpus_per_node,
        nnodes=args.nnodes,
        total_epochs=args.total_epochs,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        default_local_dir=args.default_local_dir,
    )
    command = config.to_command(python_executable=args.python_executable)
    if args.output_script:
        target = Path(args.output_script)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(command + "\n", encoding="utf-8")
    print(command)


if __name__ == "__main__":
    main()
