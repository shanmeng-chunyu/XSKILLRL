"""Utilities for SkillRL-style recursive SkillBank updates."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def select_failed_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    *,
    threshold: float = 0.4,
    max_failures: int = 10,
    category_key: str = "task_type",
) -> List[Dict[str, Any]]:
    """Diversity-aware failure selection for dynamic skill updates.

    The paper groups validation failures by category and samples in a
    round-robin manner.  This function implements that deterministic selection
    over already collected trajectory dictionaries.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        reward = float(trajectory.get("reward", trajectory.get("success", 0.0)))
        if reward >= threshold:
            continue
        category = str(trajectory.get(category_key) or "unknown")
        grouped[category].append(dict(trajectory))

    for category in grouped:
        grouped[category].sort(key=lambda item: float(item.get("reward", 0.0)))

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_failures and grouped:
        for category in sorted(list(grouped)):
            if grouped[category]:
                selected.append(grouped[category].pop(0))
                if len(selected) >= max_failures:
                    break
            if not grouped[category]:
                del grouped[category]
    return selected


def build_failure_skill_prompt(
    failed_trajectories: Sequence[Dict[str, Any]],
    current_skills: Dict[str, Any],
    *,
    max_new_skills: int = 3,
) -> str:
    existing_titles = []
    for skill in current_skills.get("general_skills", []):
        existing_titles.append(skill.get("title", ""))
    for category, skills in current_skills.get("task_specific_skills", {}).items():
        for skill in skills:
            existing_titles.append(f"[{category}] {skill.get('title', '')}")

    failures = []
    for index, trajectory in enumerate(failed_trajectories[:5], start=1):
        failures.append(
            "\n".join(
                [
                    f"Example {index}:",
                    f"Task: {trajectory.get('task') or trajectory.get('problem', '')}",
                    f"Task Type: {trajectory.get('task_type', 'unknown')}",
                    f"Trajectory: {trajectory.get('trajectory', trajectory)}",
                ]
            )
        )

    return f"""Analyze these failed visual reasoning agent trajectories and suggest NEW skills to add.

FAILED TRAJECTORIES:
{chr(10).join(failures)}

EXISTING SKILL TITLES:
{existing_titles}

Generate 1-{max_new_skills} NEW actionable skills that would help avoid these failures.
Each skill must have: skill_id, title, principle, when_to_apply.
Return ONLY a JSON array of skills, no other text.
"""


def parse_new_skills(response: str) -> List[Dict[str, Any]]:
    start = response.find("[")
    end = response.rfind("]") + 1
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(response[start:end])
    except json.JSONDecodeError:
        return []
    skills = []
    for skill in payload:
        if isinstance(skill, dict) and {"title", "principle"}.issubset(skill):
            skills.append(dict(skill))
    return skills


def next_dynamic_skill_id(skills: Dict[str, Any]) -> str:
    pattern = re.compile(r"^dyn_(\d+)$")
    max_id = 0
    for skill in skills.get("general_skills", []):
        match = pattern.match(str(skill.get("skill_id", "")))
        if match:
            max_id = max(max_id, int(match.group(1)))
    for task_skills in skills.get("task_specific_skills", {}).values():
        for skill in task_skills:
            match = pattern.match(str(skill.get("skill_id", "")))
            if match:
                max_id = max(max_id, int(match.group(1)))
    return f"dyn_{max_id + 1:03d}"
