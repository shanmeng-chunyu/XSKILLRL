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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import ast
from collections import defaultdict
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def _flatten_media_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                return _flatten_media_items(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                pass
        return [value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, str)):
        try:
            return _flatten_media_items(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        if any(key in value for key in ("image", "path", "url", "video")):
            return [value]
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_media_items(item))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_media_items(item))
        return flattened
    return [value]


def _make_text_position_ids(attention_mask: torch.Tensor, *, expand_to_3d: bool = False) -> torch.Tensor:
    position_ids = compute_position_id_with_mask(attention_mask)
    if expand_to_3d:
        return position_ids.unsqueeze(0).expand(3, -1)
    return position_ids


def _pad_tensor_list(tensors: list[torch.Tensor], *, pad_value=0):
    if not tensors:
        return None
    max_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        max_shape = [max(left, right) for left, right in zip(max_shape, tensor.shape)]
    padded = []
    masks = []
    for tensor in tensors:
        output = torch.full(max_shape, pad_value, dtype=tensor.dtype)
        slices = tuple(slice(0, size) for size in tensor.shape)
        output[slices] = tensor
        padded.append(output)
        mask = torch.zeros(max_shape[0], dtype=torch.bool)
        mask[: tensor.shape[0]] = True
        masks.append(mask)
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def sft_collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            tensors[key].append(value)

    batch = {}
    for key, values in tensors.items():
        if key in {"pixel_values", "image_grid_thw"}:
            padded, mask = _pad_tensor_list(values)
            batch[key] = padded
            batch[f"{key}_mask"] = mask
        else:
            batch[key] = torch.stack(values, dim=0)
    return batch


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config, processor=None):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        image_key = config.get("image_key", "images")
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)
        enable_multimodal = config.get("enable_multimodal", True)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin | None = processor if enable_multimodal else None

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []
        self.image_key = image_key

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()
        if self.image_key in self.dataframe.columns:
            self.images = self.dataframe[self.image_key].tolist()
        else:
            self.images = [[] for _ in self.responses]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        image_items = _flatten_media_items(self.images[item])
        use_multimodal = self.processor is not None and len(image_items) > 0

        if use_multimodal:
            from verl.utils.dataset.vision_utils import process_image

            processed_images = [process_image(image) for image in image_items]
            prompt_chat = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in processed_images]
                    + [{"type": "text", "text": str(prompt)}],
                }
            ]
            prompt_chat_str = self.processor.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            prompt_ids_output = self.processor(text=[prompt_chat_str], images=processed_images, return_tensors="pt")
            prompt_ids = prompt_ids_output.pop("input_ids")[0]
            prompt_attention_mask = prompt_ids_output.pop("attention_mask")[0]
            prompt_mm_inputs = dict(prompt_ids_output)
        else:
            # apply chat template
            prompt_chat = [{"role": "user", "content": prompt}]

            # string
            prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]
            prompt_mm_inputs = {}
        response_chat_str = response + tokenizer.eos_token

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]

        if use_multimodal:
            image_grid_thw = prompt_mm_inputs.get("image_grid_thw")
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_grid_thw is not None and image_token_id is not None:
                merge_size = self.processor.image_processor.merge_size
                expected_tokens = int((image_grid_thw[:, 0] * (image_grid_thw[:, 1] // merge_size) * (image_grid_thw[:, 2] // merge_size)).sum().item())
                actual_tokens = int((input_ids == image_token_id).sum().item())
                if actual_tokens != expected_tokens:
                    raise RuntimeError(
                        "Multimodal SFT sample has mismatched image tokens after truncation: "
                        f"{actual_tokens=} {expected_tokens=} {item=}."
                    )
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=prompt_mm_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
            )
        else:
            position_ids = _make_text_position_ids(attention_mask, expand_to_3d=self.processor is not None)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        if use_multimodal:
            if "pixel_values" in prompt_mm_inputs:
                output["pixel_values"] = prompt_mm_inputs["pixel_values"]
            if "image_grid_thw" in prompt_mm_inputs:
                output["image_grid_thw"] = prompt_mm_inputs["image_grid_thw"]
        else:
            output["pixel_values"] = torch.empty((0, 1), dtype=torch.float32)
            output["image_grid_thw"] = torch.empty((0, 3), dtype=torch.long)
        return output
