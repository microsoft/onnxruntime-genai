# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import os

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open


def load_qwen35_visual(model_class, config, model_name_or_path, **kwargs):
    def resolve(filename):
        if os.path.isdir(model_name_or_path):
            path = os.path.join(model_name_or_path, filename)
            return path if os.path.isfile(path) else None
        try:
            return hf_hub_download(model_name_or_path, filename, **kwargs)
        except EntryNotFoundError:
            return None

    for index_name, weight_name in (
        ("model.safetensors.index.json", "model.safetensors"),
        ("pytorch_model.bin.index.json", "pytorch_model.bin"),
    ):
        index_path = resolve(index_name)
        if index_path:
            with open(index_path) as index_file:
                weight_map = json.load(index_file)["weight_map"]
            filenames = sorted({filename for name, filename in weight_map.items() if name.startswith("model.visual.")})
            break
        if resolve(weight_name):
            filenames = [weight_name]
            break
    else:
        raise FileNotFoundError("Qwen3.5 vision export requires a safetensors or PyTorch checkpoint.")
    state = {}
    for filename in filenames:
        path = resolve(filename)
        if path is None:
            raise FileNotFoundError(f"Missing Qwen3.5 checkpoint shard: {filename}")
        if filename.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as weights:
                for name in weights.keys():  # noqa: SIM118 (safe_open is not iterable)
                    if name.startswith("model.visual."):
                        state[name.removeprefix("model.visual.")] = weights.get_tensor(name)
        else:
            weights = torch.load(path, map_location="cpu", weights_only=True)
            for name, tensor in weights.items():
                if name.startswith("model.visual."):
                    state[name.removeprefix("model.visual.")] = tensor
            del weights
    with torch.device("meta"):
        visual = model_class(config)
    visual.load_state_dict(state, strict=True, assign=True)
    rotary_config = config if hasattr(visual.rotary_pos_emb, "config") else config.hidden_size // config.num_heads // 2
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(rotary_config)
    return visual


class Qwen38VisionLoader:
    def load_visual(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig  # noqa: PLC0415
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: PLC0415
            Qwen4ExpVisionModel,
            Qwen4ExpVisionRotaryEmbedding,
        )

        try:
            config = Qwen4ExpVisionConfig(**self.vision_config)
            with torch.device("meta"):
                visual = Qwen4ExpVisionModel(config)
            prefix = "model.visual."
            state = {
                name[len(prefix) :]: self.get_tensor(name) for name in self.tensor_names() if name.startswith(prefix)
            }
            visual.load_state_dict(state, strict=True, assign=True)
            visual.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(config)
            return visual
        finally:
            self.close()
