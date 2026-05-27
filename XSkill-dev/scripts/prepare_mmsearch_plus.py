"""Prepare MMSearch-Plus into the local mixed-training protocol."""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xskill_rl.benchmark_protocol import (
    ensure_benchmark_prefixed_id,
    normalize_images,
    normalize_record,
    read_json,
    stratified_train_test_split,
    write_split_bundle,
)


def _derive_xor_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode("utf-8")).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _try_decrypt_text(value: Any, password: str) -> Any:
    """Decrypt MMSearch-Plus obfuscated strings when possible.

    The public dataset uses base64-encoded XOR text with the canary password
    "MMSearch-Plus". If a value is already plain text or cannot be decrypted,
    it is returned unchanged.
    """
    if isinstance(value, list):
        return [_try_decrypt_text(item, password) for item in value]
    if not isinstance(value, str) or not value:
        return value
    try:
        encrypted = base64.b64decode(value, validate=True)
        key = _derive_xor_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")
    except Exception:
        return value
    if not decrypted.strip():
        return value
    return decrypted


def _decrypt_record(item: Dict, password: str | None) -> Dict:
    if not password:
        return dict(item)
    updated = dict(item)
    for key in ("problem", "question", "solution", "answer"):
        if key in updated:
            updated[key] = _try_decrypt_text(updated[key], password)
    return updated


def _collect_images(item: Dict) -> List[str]:
    if item.get("images"):
        return normalize_images(item["images"])
    ordered = []
    for key in sorted(item):
        if key.startswith("img_") and item[key]:
            ordered.append(item[key])
    return normalize_images(ordered)


def normalize_mmsearch_plus(records: List[Dict], *, decrypt_password: str | None = "MMSearch-Plus") -> List[Dict]:
    normalized = []
    for index, item in enumerate(records):
        item = _decrypt_record(item, decrypt_password)
        doc_id = item.get("doc_id") or item.get("id") or f"mmsearch_plus_{index:04d}"
        doc_id = ensure_benchmark_prefixed_id(doc_id, "mmsearch_plus")
        problem = item.get("problem") or item.get("question") or ""
        solution = item.get("solution") or item.get("answer") or ""
        images = _collect_images(item)
        category = item.get("category") or "unknown"
        difficulty = item.get("difficulty") or "unknown"
        extra = dict(item)
        extra["category"] = category
        extra["difficulty"] = difficulty
        normalized.append(
            normalize_record(
                extra,
                benchmark_name="mmsearch_plus",
                doc_id=doc_id,
                problem=problem,
                solution=solution,
                images=images,
                extra_fields=extra,
            )
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--decrypt-password",
        default="MMSearch-Plus",
        help="Password/canary for MMSearch-Plus encrypted text fields. Use empty string to disable.",
    )
    args = parser.parse_args()

    decrypt_password = args.decrypt_password or None
    records = normalize_mmsearch_plus(read_json(args.input_json), decrypt_password=decrypt_password)
    train_records, test_records, manifest = stratified_train_test_split(
        records,
        test_ratio=args.test_ratio,
        seed=args.seed,
        candidate_key_groups=[["category", "difficulty"], ["category"]],
    )
    write_split_bundle(
        args.output_dir,
        benchmark_name="mmsearch_plus",
        all_records=records,
        train_records=train_records,
        test_records=test_records,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
