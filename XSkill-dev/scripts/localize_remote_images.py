"""Download remote image URLs in XSkill JSON data and rewrite them to local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_url(value: str) -> str:
    text = value.strip()
    if text.startswith("https:/") and not text.startswith("https://"):
        return "https://" + text[len("https:/") :]
    if text.startswith("http:/") and not text.startswith("http://"):
        return "http://" + text[len("http:/") :]
    return text


def is_http_url(value: Any) -> bool:
    return str(value).startswith(("http://", "https://"))


def image_url_from_item(item: Any) -> str | None:
    if isinstance(item, dict):
        value = item.get("image") or item.get("url") or item.get("path")
    else:
        value = item
    if value is None:
        return None
    text = normalize_url(str(value))
    return text if is_http_url(text) else None


def extension_from_url(url: str, content_type: str | None = None) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".png"


def target_relative_path(url: str, content_type: str | None = None) -> Path:
    parsed = urlparse(url)
    stem = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = extension_from_url(url, content_type)
    host = parsed.netloc.replace(":", "_") or "remote"
    parts = [part for part in Path(unquote(parsed.path)).parts if part not in {"/", "\\"}]
    if len(parts) >= 2:
        subdir = Path(host, *parts[:-1])
    else:
        subdir = Path(host)
    original_stem = Path(unquote(parsed.path)).stem or "image"
    return subdir / f"{original_stem}_{stem}{ext}"


def download_bytes(url: str, *, retries: int, timeout: int, sleep: float) -> tuple[bytes, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type")
                return response.read(), content_type
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def localize_url(
    url: str,
    *,
    cache_dir: Path,
    path_prefix: str,
    retries: int,
    timeout: int,
    sleep: float,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    url = normalize_url(url)
    rel_guess = target_relative_path(url)
    target = cache_dir / rel_guess
    prefix = path_prefix.rstrip("/")
    local_ref = f"{prefix}/{rel_guess.as_posix()}" if prefix else rel_guess.as_posix()
    if target.exists():
        return local_ref, {
            "url": url,
            "local_path": str(target),
            "relative_path": rel_guess.as_posix(),
            "status": "cached",
        }

    if dry_run:
        return local_ref, {
            "url": url,
            "local_path": str(target),
            "relative_path": rel_guess.as_posix(),
            "status": "dry_run",
        }

    data, content_type = download_bytes(url, retries=retries, timeout=timeout, sleep=sleep)
    rel_path = target_relative_path(url, content_type)
    target = cache_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    prefix = path_prefix.rstrip("/")
    local_ref = f"{prefix}/{rel_path.as_posix()}" if prefix else rel_path.as_posix()
    return local_ref, {
        "url": url,
        "local_path": str(target),
        "relative_path": rel_path.as_posix(),
        "bytes": len(data),
        "status": "downloaded",
    }


def localize_image_item(item: Any, *, url_map: dict[str, str], **kwargs) -> tuple[Any, dict[str, Any] | None]:
    url = image_url_from_item(item)
    if not url:
        return item, None
    if url in url_map:
        local_ref = url_map[url]
        return replace_image_value(item, local_ref), {"url": url, "replacement": local_ref, "status": "reused"}
    local_ref, row = localize_url(url, **kwargs)
    url_map[url] = local_ref
    row["replacement"] = local_ref
    return replace_image_value(item, local_ref), row


def replace_image_value(item: Any, value: str) -> Any:
    if isinstance(item, dict):
        updated = dict(item)
        for key in ("image", "url", "path"):
            if key in updated:
                updated[key] = value
                return updated
        updated["image"] = value
        return updated
    return value


def localize_record(record: Any, *, manifest: list[dict[str, Any]], url_map: dict[str, str], **kwargs) -> Any:
    if not isinstance(record, dict):
        return record
    updated = dict(record)
    images = updated.get("images")
    if isinstance(images, list):
        new_images = []
        for image in images:
            new_image, row = localize_image_item(image, url_map=url_map, **kwargs)
            new_images.append(new_image)
            if row is not None:
                manifest.append(row)
        updated["images"] = new_images
    else:
        new_image, row = localize_image_item(images, url_map=url_map, **kwargs)
        if row is not None:
            updated["images"] = [new_image]
            manifest.append(row)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", action="append", required=True, help="Input XSkill JSON file. Can be repeated.")
    parser.add_argument("--output-json", action="append", required=True, help="Output JSON file. Must match --input-json count.")
    parser.add_argument("--cache-dir", default="benchmark/_remote_images", help="Directory where downloaded images are stored.")
    parser.add_argument(
        "--path-prefix",
        default="_remote_images",
        help="Path prefix written into output JSON. Use a path relative to the runtime image root.",
    )
    parser.add_argument("--manifest-path", default="output/rl_data/remote_image_manifest.json")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.input_json) != len(args.output_json):
        raise ValueError("--input-json and --output-json must have the same count")

    cache_dir = Path(args.cache_dir)
    manifest: list[dict[str, Any]] = []
    url_map: dict[str, str] = {}

    for input_name, output_name in zip(args.input_json, args.output_json):
        input_path = Path(input_name)
        output_path = Path(output_name)
        payload = read_json(input_path)
        if isinstance(payload, list):
            localized = [
                localize_record(
                    record,
                    manifest=manifest,
                    url_map=url_map,
                    cache_dir=cache_dir,
                    path_prefix=args.path_prefix,
                    retries=args.retries,
                    timeout=args.timeout,
                    sleep=args.sleep,
                    dry_run=args.dry_run,
                )
                for record in payload
            ]
        else:
            localized = localize_record(
                payload,
                manifest=manifest,
                url_map=url_map,
                cache_dir=cache_dir,
                path_prefix=args.path_prefix,
                retries=args.retries,
                timeout=args.timeout,
                sleep=args.sleep,
                dry_run=args.dry_run,
            )
        write_json(output_path, localized)
        print(f"{input_path} -> {output_path}")

    write_json(Path(args.manifest_path), manifest)
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    print(f"manifest: {args.manifest_path}")
    print(f"unique remote urls: {len(url_map)}")
    print(f"records: {counts}")


if __name__ == "__main__":
    main()
