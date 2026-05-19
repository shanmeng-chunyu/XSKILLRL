# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO
import os
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def _is_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "file://", "data:image"))


def _normalize_url(value: str) -> str:
    if value.startswith("https:/") and not value.startswith("https://"):
        return "https://" + value[len("https:/") :]
    if value.startswith("http:/") and not value.startswith("http://"):
        return "http://" + value[len("http:/") :]
    return value


def _as_existing_file_uri(path: Path) -> str | None:
    try:
        if path.exists():
            return path.resolve().as_uri()
    except OSError:
        return None
    return None


def _resolve_image_string(image: str) -> str:
    image = _normalize_url(image.strip())
    if not image or _is_uri(image):
        return image

    raw_path = Path(image)
    if raw_path.is_absolute():
        return _as_existing_file_uri(raw_path) or image

    candidates: list[Path] = []
    for env_name in ("XSKILL_IMAGE_ROOT", "XSKILL_BENCHMARK_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value) / raw_path)

    repo_root = os.environ.get("XSKILL_REPO_ROOT")
    if repo_root:
        candidates.append(Path(repo_root) / raw_path)
        candidates.append(Path(repo_root) / "benchmark" / raw_path)

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / raw_path,
            cwd.parent / "XSkill-dev" / raw_path,
            cwd.parent / "XSkill-dev" / "benchmark" / raw_path,
        ]
    )

    for candidate in candidates:
        uri = _as_existing_file_uri(candidate)
        if uri:
            return uri
    return image


def process_image(image: Union[str, os.PathLike, dict, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, os.PathLike)):
        image = {"image": _resolve_image_string(os.fspath(image))}
    else:
        image = dict(image)
        if isinstance(image.get("image"), (str, os.PathLike)):
            image["image"] = _resolve_image_string(os.fspath(image["image"]))

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = BytesIO(image["bytes"])

    return fetch_image(image)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(video)
