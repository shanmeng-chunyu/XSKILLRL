"""Check paths required by XSkill + SkillRL GRPO after moving to a new server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        records = data if isinstance(data, list) else [data]
    elif suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".parquet":
        import pandas as pd

        records = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return records[:limit] if limit is not None else records


def image_values(record: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for container in (record, record.get("env_kwargs") if isinstance(record.get("env_kwargs"), dict) else None):
        if not isinstance(container, dict):
            continue
        images = container.get("images")
        if images is None:
            continue
        if isinstance(images, list):
            values.extend(images)
        else:
            values.append(images)
    return list(_flatten_image_values(values))


def _flatten_image_values(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, dict):
        if any(key in values for key in ("image", "path", "url")):
            return [values]
        flattened: list[Any] = []
        for value in values.values():
            flattened.extend(_flatten_image_values(value))
        return flattened
    if isinstance(values, (list, tuple)):
        flattened = []
        for value in values:
            flattened.extend(_flatten_image_values(value))
        return flattened
    return [values]


def image_path(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("image") or value.get("path") or value.get("url")
    if value is None:
        return None
    text = str(value)
    if text.startswith(("http://", "https://", "data:image")):
        return None
    if text.startswith("file://"):
        return text[len("file://") :]
    return text


def resolve_image(path_text: str, roots: list[Path]) -> Path | None:
    path = Path(path_text)
    if path.is_absolute() and path.is_file():
        return path
    candidates = [path]
    if path.is_absolute():
        candidates = _image_suffixes(path)
    else:
        candidates.extend(_image_suffixes(path))
    for root in roots:
        for relative in candidates:
            candidate = root / relative
            if candidate.is_file():
                return candidate
    return None


def _image_suffixes(path: Path) -> list[Path]:
    parts = path.parts
    suffixes: list[Path] = []
    for marker in ("images", "benchmark"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                suffixes.append(Path(*parts[index + 1 :]))
            suffixes.append(Path(*parts[index:]))
    if path.name:
        suffixes.append(Path(path.name))
    return suffixes


def check_file(label: str, path: Path) -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xskill-root", default=None)
    parser.add_argument("--skillrl-root", default=None)
    parser.add_argument("--train-file", default="output/rl_data/skillrl_train.parquet")
    parser.add_argument("--val-file", default="output/rl_data/skillrl_val.parquet")
    parser.add_argument(
        "--skill-bank-json",
        default=None,
        help="Optional SkillBank JSON path. Omit this for no-memory GRPO baseline checks.",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    xskill_root = Path(args.xskill_root or os.environ.get("XSKILL_REPO_ROOT") or script_root).resolve()
    skillrl_root = Path(args.skillrl_root or os.environ.get("SKILLRL_REPO_ROOT") or xskill_root.parent / "SkillRL").resolve()
    image_root = Path(args.image_root or os.environ.get("XSKILL_IMAGE_ROOT") or xskill_root).resolve()
    roots = [image_root, xskill_root, xskill_root / "benchmark", xskill_root.parent / "images", Path.cwd()]

    ok = True
    ok &= check_file("XSkill root", xskill_root)
    ok &= check_file("SkillRL root", skillrl_root)
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    if not train_file.is_absolute():
        train_file = xskill_root / train_file
    if not val_file.is_absolute():
        val_file = xskill_root / val_file
    ok &= check_file("train file", train_file)
    ok &= check_file("val file", val_file)
    if args.skill_bank_json:
        skill_bank = Path(args.skill_bank_json)
        if not skill_bank.is_absolute():
            skill_bank = xskill_root / skill_bank
        ok &= check_file("skill bank", skill_bank)
    else:
        print("[SKIP] skill bank: not required for no-memory baseline")
    print(f"[INFO] image roots: {', '.join(str(root) for root in roots)}")

    missing_images: list[str] = []
    checked = 0
    for dataset_path in (train_file, val_file):
        if not dataset_path.exists():
            continue
        for record in read_records(dataset_path, limit=args.sample_limit):
            for value in image_values(record):
                path_text = image_path(value)
                if not path_text:
                    continue
                checked += 1
                if resolve_image(path_text, roots) is None:
                    missing_images.append(path_text)
                    if len(missing_images) >= 20:
                        break
            if len(missing_images) >= 20:
                break
    if missing_images:
        ok = False
        print("[MISSING] sample image paths:")
        for item in missing_images:
            print(f"  - {item}")
    print(f"[INFO] checked image refs: {checked}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
