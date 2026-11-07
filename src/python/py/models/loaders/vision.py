# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import math
import os
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

from .base import TensorModule


def load_visual_state(model_name_or_path, prefixes=("model.visual.",), **kwargs):
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
            filenames = sorted({filename for name, filename in weight_map.items() if name.startswith(prefixes)})
            break
        if resolve(weight_name):
            filenames = [weight_name]
            break
    else:
        raise FileNotFoundError("Vision export requires a safetensors or PyTorch checkpoint.")
    state = {}
    for filename in filenames:
        path = resolve(filename)
        if path is None:
            raise FileNotFoundError(f"Missing vision checkpoint shard: {filename}")
        if filename.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as weights:
                for name in weights.keys():  # noqa: SIM118 (safe_open is not iterable)
                    if name.startswith(prefixes):
                        state[name] = weights.get_tensor(name)
        else:
            weights = torch.load(path, map_location="cpu", weights_only=True)
            state.update({name: tensor for name, tensor in weights.items() if name.startswith(prefixes)})
            del weights
    return state


def load_qwen35_visual(model_class, config, model_name_or_path, **kwargs):
    state = {
        name.removeprefix("model.visual."): tensor
        for name, tensor in load_visual_state(model_name_or_path, **kwargs).items()
    }
    with torch.device("meta"):
        visual = model_class(config)
    visual.load_state_dict(state, strict=True, assign=True)
    rotary_config = config if hasattr(visual.rotary_pos_emb, "config") else config.hidden_size // config.num_heads // 2
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(rotary_config)
    return visual


def load_bailing_visual(config, text_hidden_size, model_name_or_path, **kwargs):
    state = load_visual_state(model_name_or_path, prefixes=("model.visual.", "linear_proj."), **kwargs)

    def module(name, shape):
        tensors = {}
        for suffix, expected in (("weight", shape), ("bias", shape[:1])):
            key = f"{name}.{suffix}"
            tensor = state.pop(key, None)
            if tensor is None or list(tensor.shape) != expected:
                raise ValueError(f"Ling vision requires {key} with shape {expected}")
            tensors[suffix] = tensor
        return TensorModule(**tensors)

    def norm(name):
        layer = module(name, [config.hidden_size])
        layer.eps = 1e-6
        return layer

    hidden = config.hidden_size
    grid_side = math.isqrt(config.num_position_embeddings)
    if grid_side**2 != config.num_position_embeddings:
        raise ValueError("Ling learned vision positions must form a square grid")
    rotary_dim = hidden // config.num_heads // 2
    blocks = []
    for index in range(config.depth):
        prefix = f"model.visual.blocks.{index}"
        blocks.append(
            SimpleNamespace(
                norm1=norm(f"{prefix}.norm1"),
                norm2=norm(f"{prefix}.norm2"),
                attn=SimpleNamespace(
                    qkv=module(f"{prefix}.attn.qkv", [3 * hidden, hidden]),
                    proj=module(f"{prefix}.attn.proj", [hidden, hidden]),
                    scaling=(hidden // config.num_heads) ** -0.5,
                ),
                mlp=SimpleNamespace(
                    linear_fc1=module(f"{prefix}.mlp.linear_fc1", [config.intermediate_size, hidden]),
                    linear_fc2=module(f"{prefix}.mlp.linear_fc2", [hidden, config.intermediate_size]),
                ),
            )
        )
    visual = SimpleNamespace(
        blocks=blocks,
        num_grid_per_side=grid_side,
        rotary_pos_emb=SimpleNamespace(inv_freq=1 / 10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)),
        patch_embed=SimpleNamespace(
            proj=module(
                "model.visual.patch_embed.proj",
                [hidden, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size],
            )
        ),
        pos_embed=TensorModule(weight=state.pop("model.visual.pos_embed.weight", None)),
        merger=SimpleNamespace(norm=norm("model.visual.merger.norm")),
        projector=SimpleNamespace(
            linear_fc1=module("linear_proj.0", [text_hidden_size, hidden * config.spatial_merge_size**2]),
            linear_fc2=module("linear_proj.2", [text_hidden_size, text_hidden_size]),
        ),
    )
    if visual.pos_embed.weight is None or list(visual.pos_embed.weight.shape) != [
        config.num_position_embeddings,
        hidden,
    ]:
        raise ValueError("Ling learned vision position shape does not match the vision configuration")
    if state:
        raise ValueError(f"Unsupported Ling vision tensors: {sorted(state)}")
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
