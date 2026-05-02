"""Small GRPO utilities aligned with SkillRL's optimizer contract.

The actual parameter update is still delegated to an external GRPO trainer
such as the SkillRL/verl stack.  These helpers keep the data and manifests in
this repo explicit about group size, reward normalization, and KL settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class GRPOHyperParams:
    """Default GRPO settings from SkillRL's paper and run scripts."""

    learning_rate: float = 1e-6
    train_batch_size: int = 64
    group_size: int = 8
    gradient_accumulation_steps: int = 4
    kl_loss_coef: float = 0.01
    clip_ratio: float = 0.2
    invalid_action_penalty_coef: float = 0.1
    max_prompt_length: int = 6000
    max_response_length: int = 1024
    total_epochs: int = 150

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> List[float]:
    """Compute GRPO group-normalized advantages.

    SkillRL uses ``A_i = (R_i - mean(R)) / std(R)`` over each sampled group.
    If every reward in a group is identical, the group contributes no relative
    preference signal, so this helper returns zeros.
    """

    values = [float(reward) for reward in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = sqrt(variance)
    if std <= eps:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def summarize_group_rewards(groups: Iterable[Sequence[float]]) -> Dict:
    """Return lightweight diagnostics for GRPO reward groups."""

    group_summaries = []
    for rewards in groups:
        values = [float(value) for value in rewards]
        advantages = compute_group_advantages(values)
        group_summaries.append(
            {
                "count": len(values),
                "mean_reward": sum(values) / len(values) if values else 0.0,
                "advantages": advantages,
            }
        )
    return {"groups": group_summaries, "num_groups": len(group_summaries)}
