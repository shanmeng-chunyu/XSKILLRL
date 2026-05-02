"""Convert an XSkill SKILL.md file into SkillRL SkillBank JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.skillrl.skill_bank import SkillBank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-skill-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--category", default="xskill_visual_reasoning")
    parser.add_argument("--max-skills", type=int, default=24)
    args = parser.parse_args()

    markdown = Path(args.input_skill_md).read_text(encoding="utf-8-sig")
    bank = SkillBank.from_xskill_markdown(
        markdown,
        category=args.category,
        max_skills=args.max_skills,
    )
    bank.save(args.output_json)
    print(bank.counts())


if __name__ == "__main__":
    main()
