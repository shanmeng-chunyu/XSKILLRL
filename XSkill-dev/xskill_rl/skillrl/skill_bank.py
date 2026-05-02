"""SkillRL-style hierarchical SkillBank support.

SkillRL stores skills as JSON with ``general_skills``,
``task_specific_skills``, and ``common_mistakes``.  This module mirrors the
official ``SkillsOnlyMemory`` behavior while staying independent from the
heavy SkillRL/verl runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Skill = Dict[str, Any]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "general"


def _compact(value: str, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_skill(skill: Skill, *, fallback_id: str) -> Skill:
    """Normalize one skill dict to SkillRL's required fields."""

    payload = dict(skill)
    payload.setdefault("skill_id", fallback_id)
    payload.setdefault("title", payload["skill_id"].replace("_", " ").title())
    payload.setdefault("principle", "")
    payload.setdefault("when_to_apply", "")
    return payload


def empty_skill_bank() -> Dict[str, Any]:
    return {
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
    }


def validate_skill_bank(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized SkillRL-style skill bank."""

    bank = empty_skill_bank()
    bank.update(payload or {})

    general = []
    for index, skill in enumerate(bank.get("general_skills") or [], start=1):
        general.append(normalize_skill(skill, fallback_id=f"gen_{index:03d}"))
    bank["general_skills"] = general

    task_specific: Dict[str, List[Skill]] = {}
    for category, skills in (bank.get("task_specific_skills") or {}).items():
        key = _slug(str(category))
        task_specific[key] = [
            normalize_skill(skill, fallback_id=f"{key}_{index:03d}")
            for index, skill in enumerate(skills or [], start=1)
        ]
    bank["task_specific_skills"] = task_specific

    mistakes = []
    for index, mistake in enumerate(bank.get("common_mistakes") or [], start=1):
        payload = dict(mistake)
        payload.setdefault("mistake_id", f"err_{index:03d}")
        payload.setdefault("description", "")
        payload.setdefault("why_it_happens", "")
        payload.setdefault("how_to_avoid", "")
        mistakes.append(payload)
    bank["common_mistakes"] = mistakes
    return bank


@dataclass
class SkillBank:
    """Container for SkillRL-compatible skill JSON."""

    payload: Dict[str, Any]

    def __post_init__(self) -> None:
        self.payload = validate_skill_bank(self.payload)

    @classmethod
    def load(cls, path: Path | str) -> "SkillBank":
        with Path(path).open("r", encoding="utf-8-sig") as f:
            return cls(json.load(f))

    @classmethod
    def from_xskill_markdown(
        cls,
        markdown: str,
        *,
        category: str = "xskill_visual_reasoning",
        max_skills: int = 24,
    ) -> "SkillBank":
        """Wrap an XSkill ``SKILL.md`` document as a SkillRL SkillBank.

        XSkill keeps a free-form skill document.  SkillRL expects structured
        JSON.  This converter keeps the content intact by turning Markdown
        sections into general skills.
        """

        sections = _split_markdown_sections(markdown)
        skills = []
        for index, (title, body) in enumerate(sections[:max_skills], start=1):
            principle = _compact(body or title)
            skills.append(
                {
                    "skill_id": f"xskill_{index:03d}",
                    "title": _compact(title, limit=80),
                    "principle": principle,
                    "when_to_apply": (
                        "When a visual reasoning task benefits from this "
                        "XSkill memory procedure."
                    ),
                }
            )
        return cls(
            {
                "general_skills": skills,
                "task_specific_skills": {_slug(category): []},
                "common_mistakes": [],
            }
        )

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.payload, f, ensure_ascii=False, indent=2)

    def counts(self) -> Dict[str, int]:
        return {
            "general": len(self.payload.get("general_skills", [])),
            "task_specific": sum(
                len(skills)
                for skills in self.payload.get("task_specific_skills", {}).values()
            ),
            "common_mistakes": len(self.payload.get("common_mistakes", [])),
        }

    def all_skill_ids(self) -> set[str]:
        ids = {
            skill.get("skill_id")
            for skill in self.payload.get("general_skills", [])
            if skill.get("skill_id")
        }
        for skills in self.payload.get("task_specific_skills", {}).values():
            ids.update(skill.get("skill_id") for skill in skills if skill.get("skill_id"))
        return ids

    def add_skills(self, new_skills: Iterable[Skill], *, category: str = "general") -> int:
        existing = self.all_skill_ids()
        added = 0
        for skill in new_skills:
            normalized = normalize_skill(skill, fallback_id=f"dyn_{added + 1:03d}")
            if normalized.get("skill_id") in existing:
                continue
            if category == "general":
                self.payload.setdefault("general_skills", []).append(normalized)
            else:
                key = _slug(category)
                self.payload.setdefault("task_specific_skills", {}).setdefault(key, []).append(
                    normalized
                )
            existing.add(normalized.get("skill_id"))
            added += 1
        return added


class SkillsOnlyMemory:
    """SkillRL-compatible skill retriever.

    Template mode is dependency-free and mirrors the official SkillRL behavior:
    include dynamic general skills first, fill the remaining general budget with
    static skills, and return task-specific skills for the detected category.
    Embedding mode lazily imports ``sentence_transformers`` and ``numpy``.
    """

    def __init__(
        self,
        skills_json_path: Path | str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
    ) -> None:
        if retrieval_mode not in {"template", "embedding"}:
            raise ValueError("retrieval_mode must be 'template' or 'embedding'")
        self.skill_bank = SkillBank.load(skills_json_path)
        self.skills = self.skill_bank.payload
        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self._embedding_model = None
        self._embedding_cache = None

    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        common_mistakes = self.skills.get("common_mistakes", [])[:5]
        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general_skills, task_skills = self._embedding_retrieve(
                task_description,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
            )
            task_type = self._detect_task_type(task_description, metadata=metadata)
            return {
                "general_skills": general_skills,
                "task_specific_skills": task_skills,
                "mistakes_to_avoid": common_mistakes,
                "task_type": task_type,
                "task_specific_examples": [],
                "retrieval_mode": "embedding",
            }

        task_type = self._detect_task_type(task_description, metadata=metadata)
        all_general = self.skills.get("general_skills", [])
        dynamic = [s for s in all_general if str(s.get("skill_id", "")).startswith("dyn_")]
        static = [s for s in all_general if not str(s.get("skill_id", "")).startswith("dyn_")]
        general_skills = dynamic + static[: max(0, top_k - len(dynamic))]

        all_task_skills = self.skills.get("task_specific_skills", {}).get(task_type, [])
        if self.task_specific_top_k is not None:
            task_skills = all_task_skills[: self.task_specific_top_k]
        else:
            task_skills = all_task_skills

        return {
            "general_skills": general_skills,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": common_mistakes,
            "task_type": task_type,
            "task_specific_examples": [],
            "retrieval_mode": "template",
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        sections = []
        mode = retrieved_memories.get("retrieval_mode", "template")
        task_type = retrieved_memories.get("task_type", "unknown")

        general = retrieved_memories.get("general_skills", [])
        if general:
            lines = ["### General Principles"]
            for skill in general:
                title = skill.get("title", "")
                principle = skill.get("principle", "")
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        task_skills = retrieved_memories.get("task_specific_skills", [])
        if task_skills:
            title = "### Task-Relevant Skills"
            if mode == "template":
                title = f"### {task_type.replace('_', ' ').title()} Skills"
            lines = [title]
            for skill in task_skills:
                lines.append(f"- **{skill.get('title', '')}**: {skill.get('principle', '')}")
                when = skill.get("when_to_apply", "")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        mistakes = retrieved_memories.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                description = mistake.get("description", "")
                fix = mistake.get("how_to_avoid", "")
                if description:
                    lines.append(f"- **Don't**: {description}")
                if fix:
                    lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    def add_skills(self, new_skills: List[Skill], category: str = "general") -> int:
        added = self.skill_bank.add_skills(new_skills, category=category)
        if added:
            self._embedding_cache = None
            self.skills = self.skill_bank.payload
        return added

    def save_skills(self, path: Path | str) -> None:
        self.skill_bank.save(path)

    def get_skill_count(self) -> Dict[str, int]:
        counts = self.skill_bank.counts()
        counts["total"] = counts["general"] + counts["task_specific"] + counts["common_mistakes"]
        return counts

    def _detect_task_type(
        self,
        task_description: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_specific = self.skills.get("task_specific_skills", {})
        metadata = metadata or {}
        candidates = [
            metadata.get("task"),
            metadata.get("prompt_category"),
            metadata.get("category"),
            metadata.get("subdomain"),
            metadata.get("domain"),
            metadata.get("benchmark_name"),
        ]
        for candidate in candidates:
            key = _slug(str(candidate or ""))
            if key in task_specific:
                return key

        text = task_description.lower()
        for key in task_specific:
            parts = key.replace("_", " ").split()
            if parts and any(part in text for part in parts):
                return key
        return next(iter(task_specific), "unknown")

    @staticmethod
    def _skill_to_text(skill: Skill) -> str:
        return ". ".join(
            str(skill.get(field, "")).strip()
            for field in ("title", "principle", "when_to_apply")
            if str(skill.get(field, "")).strip()
        )

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_path)
        return self._embedding_model

    def _compute_skill_embeddings(self):
        if self._embedding_cache is not None:
            return self._embedding_cache

        import numpy as np

        general = [("general", None, s) for s in self.skills.get("general_skills", [])]
        task_items = [
            ("task_specific", task_type, skill)
            for task_type, skills in self.skills.get("task_specific_skills", {}).items()
            for skill in skills
        ]
        items = general + task_items
        texts = [self._skill_to_text(item[2]) for item in items]
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._embedding_cache = {
            "items": items,
            "embeddings": np.asarray(embeddings),
            "n_general": len(general),
        }
        return self._embedding_cache

    def _embedding_retrieve(
        self,
        task_description: str,
        *,
        top_k_general: int,
        top_k_task_specific: int,
    ) -> Tuple[List[Skill], List[Skill]]:
        import numpy as np

        cache = self._compute_skill_embeddings()
        if not cache["items"]:
            return [], []
        query = self._get_embedding_model().encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        similarities = cache["embeddings"] @ query
        n_general = cache["n_general"]
        general_idx = np.argsort(similarities[:n_general])[::-1][:top_k_general]
        task_idx = np.argsort(similarities[n_general:])[::-1][:top_k_task_specific]
        general = [cache["items"][int(index)][2] for index in general_idx]
        task = [cache["items"][n_general + int(index)][2] for index in task_idx]
        return general, task


def _split_markdown_sections(markdown: str) -> List[Tuple[str, str]]:
    markdown = markdown.lstrip("\ufeff")
    matches = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    if not matches:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip()]
        return [(f"XSkill Memory {i:02d}", paragraph) for i, paragraph in enumerate(paragraphs, 1)]

    sections = []
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((title, body))
    return sections
