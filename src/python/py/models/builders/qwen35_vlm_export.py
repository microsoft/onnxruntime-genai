# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Qwen3.5 VLM auxiliary ONNX export helpers.

The text decoder is built by the regular model builder. These helpers export the
embedding merger and vision encoder models that the ORT GenAI multimodal runtime
loads alongside the decoder.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class ConfigView(Mapping):
    """Small attribute/dict-style config view for unreleased HF config classes."""

    def __init__(self, data: dict[str, Any], name_or_path: str | None = None):
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = self._wrap(value)

        if "rope_parameters" in self._data and "rope_scaling" not in self._data:
            self._data["rope_scaling"] = self._data["rope_parameters"]

        if name_or_path is not None:
            self._data["_name_or_path"] = name_or_path

    def _wrap(self, value):
        if isinstance(value, dict):
            return ConfigView(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self._data[name] = self._wrap(value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str):
        return self._data[key]

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value):
            if isinstance(value, ConfigView):
                return value.to_dict()
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return {key: unwrap(value) for key, value in self._data.items() if not key.startswith("_")}

    def save_pretrained(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)


def _is_qwen35_config_error(error: Exception) -> bool:
    return "qwen3_5" in str(error) or "Qwen3_5" in str(error)


def _resolve_hf_file(model_name_or_path: str, filename: str, cache_dir: str | None, token) -> str:
    if os.path.isdir(model_name_or_path):
        return os.path.join(model_name_or_path, filename)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name_or_path, filename, cache_dir=cache_dir, token=token)


def load_qwen35_config(model_name_or_path: str, token=True, cache_dir: str | None = None) -> ConfigView:
    config_path = _resolve_hf_file(model_name_or_path, "config.json", cache_dir, token)
    with open(config_path) as f:
        data = json.load(f)
    return ConfigView(data, name_or_path=model_name_or_path)


def resolve_qwen35_model_dir(model_name_or_path: str, token=True, cache_dir: str | None = None) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return os.path.dirname(_resolve_hf_file(model_name_or_path, "config.json", cache_dir, token))


def load_qwen35_state_dict(model_name_or_path: str, token=True, cache_dir: str | None = None) -> dict[str, torch.Tensor]:
    model_dir = resolve_qwen35_model_dir(model_name_or_path, token=token, cache_dir=cache_dir)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files and cache_dir is not None and not os.path.isdir(model_name_or_path):
        model_dir = resolve_qwen35_model_dir(model_name_or_path, token=token, cache_dir=None)
        safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for safetensor_file in safetensor_files:
        state_dict.update(load_file(safetensor_file))
    return state_dict


def maybe_load_qwen35_config(model_name_or_path: str, token=True, cache_dir: str | None = None, error: Exception | None = None):
    if error is not None and not _is_qwen35_config_error(error):
        raise error
    return load_qwen35_config(model_name_or_path, token=token, cache_dir=cache_dir)


def _activation(name: str):
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported Qwen3.5 vision activation: {name}")


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = _activation(config.hidden_act)

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).view(-1, self.embed_dim)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x):
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, position_embeddings):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if getattr(torch.compiler, "is_exporting", lambda: False)():
            query_3d = query_states.transpose(1, 2).reshape(query_states.shape[0], query_states.shape[2], -1)
            key_3d = key_states.transpose(1, 2).reshape(key_states.shape[0], key_states.shape[2], -1)
            value_3d = value_states.transpose(1, 2).reshape(value_states.shape[0], value_states.shape[2], -1)
            attn_output = torch.onnx.ops.symbolic(
                "com.microsoft::MultiHeadAttention",
                (query_3d, key_3d, value_3d),
                dict(scale=self.scaling, num_heads=self.num_heads),
                dtype=query_states.dtype,
                shape=(query_states.shape[0], query_states.shape[2], self.dim),
                version=1,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=False,
            ).transpose(1, 2)

        return self.proj(attn_output.reshape(seq_length, -1).contiguous())


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, position_embeddings):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), position_embeddings=position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen35VisionModel(nn.Module):
    def __init__(self, config, fixed_image_grid_thw: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        if fixed_image_grid_thw is None:
            self.fixed_image_grid_thw = None
        else:
            self.register_buffer("fixed_image_grid_thw", fixed_image_grid_thw)
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = grid_thw[:, 1:].max()
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        all_embeddings = []
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = (
                block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = (
                block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            coords = coords.repeat(num_frames, 1)
            all_embeddings.append(freq_table[coords].flatten(1))
        return torch.cat(all_embeddings, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw):
        merge_size = self.config.spatial_merge_size
        dev = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        n = self.num_grid_per_side
        all_pos_embeds = []
        for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
            h_idxs = torch.arange(h, dtype=torch.float32, device=dev) * ((n - 1) / (h - 1))
            w_idxs = torch.arange(w, dtype=torch.float32, device=dev) * ((n - 1) / (w - 1))
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clamp(max=n - 1)
            w_ceil = (w_floor + 1).clamp(max=n - 1)
            dh = (h_idxs - h_floor.float()).to(dtype)
            dw = (w_idxs - w_floor.float()).to(dtype)
            base_h = h_floor.long() * n
            base_hc = h_ceil.long() * n
            idx_00 = (base_h[:, None] + w_floor.long()[None]).reshape(-1)
            idx_01 = (base_h[:, None] + w_ceil.long()[None]).reshape(-1)
            idx_10 = (base_hc[:, None] + w_floor.long()[None]).reshape(-1)
            idx_11 = (base_hc[:, None] + w_ceil.long()[None]).reshape(-1)
            wt_00 = ((1.0 - dh)[:, None] * (1.0 - dw)[None]).reshape(-1)
            wt_01 = ((1.0 - dh)[:, None] * dw[None]).reshape(-1)
            wt_10 = (dh[:, None] * (1.0 - dw)[None]).reshape(-1)
            wt_11 = (dh[:, None] * dw[None]).reshape(-1)
            pos = (
                self.pos_embed(idx_00.to(dev)) * wt_00[:, None]
                + self.pos_embed(idx_01.to(dev)) * wt_01[:, None]
                + self.pos_embed(idx_10.to(dev)) * wt_10[:, None]
                + self.pos_embed(idx_11.to(dev)) * wt_11[:, None]
            )
            pos = pos.repeat(t, 1)
            pos = (
                pos.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            all_pos_embeds.append(pos)
        return torch.cat(all_pos_embeds)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        if self.fixed_image_grid_thw is not None:
            image_grid_thw = self.fixed_image_grid_thw
        if image_grid_thw is None:
            raise RuntimeError("image_grid_thw is required when Qwen35VisionModel is not exported with a fixed grid")
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(image_grid_thw)
        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings=position_embeddings)
        return self.merger(hidden_states)


class Qwen35EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_token_id = config.image_token_id
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        image_mask = input_ids == self.image_token_id
        safe_input_ids = torch.where(image_mask, torch.zeros_like(input_ids), input_ids)
        inputs_embeds = self.embed_tokens(safe_input_ids)

        mask_i64 = image_mask.to(torch.int64)
        image_offsets = torch.cumsum(mask_i64, dim=1) - 1
        zero_offsets = torch.zeros_like(image_offsets)
        image_offsets = torch.where(image_mask, image_offsets, zero_offsets)

        dummy_row = torch.zeros((1, image_features.shape[1]), dtype=image_features.dtype, device=image_features.device)
        padded_image_features = torch.cat((image_features, dummy_row), dim=0)
        dummy_index = torch.full_like(image_offsets, image_features.shape[0])
        gather_indices = torch.where(image_mask, image_offsets, dummy_index)
        image_embeds = padded_image_features[gather_indices.reshape(-1)].reshape(
            input_ids.shape[0], input_ids.shape[1], self.hidden_size
        )
        image_embeds = image_embeds.to(inputs_embeds.dtype)
        return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)


def _snapshot_dir(model_name_or_path: str, cache_dir: str | None, token) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.safetensors.index.json"],
    )


def _load_qwen35_aux_state(model_name_or_path: str, cache_dir: str | None, token) -> dict[str, torch.Tensor]:
    model_dir = _snapshot_dir(model_name_or_path, cache_dir, token)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for safetensor_file in safetensor_files:
        tensors = load_file(safetensor_file)
        for name, tensor in tensors.items():
            if name.startswith("model.visual."):
                state_dict["visual." + name[len("model.visual.") :]] = tensor
            elif name == "model.language_model.embed_tokens.weight":
                state_dict["embed_tokens.weight"] = tensor
    return state_dict


def _torch_dtype_from_io_dtype(io_dtype) -> torch.dtype:
    name = getattr(io_dtype, "name", str(io_dtype))
    if name == "BFLOAT16":
        return torch.bfloat16
    if name == "FLOAT":
        return torch.float32
    return torch.float16


def _export_onnx(model, args, out_path: str, input_names, output_names, dynamic_shapes=None):
    if os.path.exists(out_path):
        os.remove(out_path)
    data_path = out_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            model,
            args,
            out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=20,
            dynamo=True,
            external_data=True,
            dynamic_shapes=dynamic_shapes,
            optimize=True,
            verbose=False,
        )


def _validate_no_ops(model_path: str, blocked_ops: set[str]):
    model = onnx.load(model_path, load_external_data=False)
    present = sorted({node.op_type for node in model.graph.node if node.op_type in blocked_ops})
    if present:
        raise RuntimeError(f"{model_path} contains unsupported ops after export: {present}")


def _write_processor_config(out_dir: str, vision_config):
    processor_config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "width": 960,
                            "height": 672,
                            "smart_resize": 1,
                            "min_pixels": 65536,
                            "max_pixels": 16777216,
                            "patch_size": vision_config.patch_size,
                            "merge_size": vision_config.spatial_merge_size,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {"rescale_factor": 1.0 / 255.0},
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "qwen2_5_vl": 1},
                    }
                },
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": vision_config.patch_size,
                            "temporal_patch_size": vision_config.temporal_patch_size,
                            "merge_size": vision_config.spatial_merge_size,
                        },
                    }
                },
            ],
        }
    }

    with open(os.path.join(out_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)


def _patch_genai_config(out_dir: str, config, execution_provider: str):
    genai_path = os.path.join(out_dir, "genai_config.json")
    with open(genai_path) as f:
        genai_config = json.load(f)

    model_config = genai_config["model"]
    model_config["type"] = "qwen3_5"
    model_config["image_token_id"] = config.image_token_id
    model_config["video_token_id"] = getattr(config, "video_token_id", 0)
    model_config["vision_start_token_id"] = config.vision_start_token_id
    model_config["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    model_config["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": config.vision_config.spatial_merge_size,
        "tokens_per_second": 2.0,
        "patch_size": config.vision_config.patch_size,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
    }

    genai_config.setdefault("search", {})
    genai_config["search"]["past_present_share_buffer"] = True
    genai_config["search"]["top_k"] = 1
    genai_config["search"]["top_p"] = 1.0

    if execution_provider == "trt-rtx":
        model_config["embedding"]["session_options"] = {
            "log_id": "onnxruntime-genai",
            "provider_options": [
                {
                    "NvTensorRtRtx": {
                        "enable_cuda_graph": "0",
                        "nv_profile_min_shapes": "input_ids:1x1,image_features:0x1024",
                        "nv_profile_opt_shapes": "input_ids:1x226,image_features:192x1024",
                        "nv_profile_max_shapes": "input_ids:1x1024,image_features:2520x1024",
                    }
                }
            ],
        }
        model_config["vision"]["session_options"] = {
            "log_id": "onnxruntime-genai",
            "provider_options": [
                {
                    "NvTensorRtRtx": {
                        "enable_cuda_graph": "0",
                        "nv_profile_min_shapes": "pixel_values:600x1536",
                        "nv_profile_opt_shapes": "pixel_values:600x1536",
                        "nv_profile_max_shapes": "pixel_values:600x1536",
                    }
                }
            ],
        }

    with open(genai_path, "w") as f:
        json.dump(genai_config, f, indent=4)


def export_qwen35_vlm_components(
    model_name_or_path: str,
    out_dir: str,
    cache_dir: str | None,
    token,
    execution_provider: str,
    io_dtype,
):
    config = load_qwen35_config(model_name_or_path, token=token, cache_dir=cache_dir)
    dtype = _torch_dtype_from_io_dtype(io_dtype)
    state_dict = _load_qwen35_aux_state(model_name_or_path, cache_dir, token)

    print("Exporting Qwen3.5 embedding.onnx...")
    embedding = Qwen35EmbeddingModel(config)
    missing, unexpected = embedding.load_state_dict(
        {"embed_tokens.weight": state_dict["embed_tokens.weight"]}, strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"Unexpected embedding state dict keys. Missing={missing}, unexpected={unexpected}")
    embedding = embedding.to(dtype).eval()
    input_ids = torch.randint(0, config.image_token_id, (2, 216), dtype=torch.int64)
    patches_per_image = 187
    input_ids[:, 2] = config.vision_start_token_id
    input_ids[:, 3 : 3 + patches_per_image] = config.image_token_id
    image_features = torch.randn((input_ids.shape[0] * patches_per_image, config.vision_config.out_hidden_size), dtype=dtype)
    _export_onnx(
        embedding,
        (input_ids, image_features),
        os.path.join(out_dir, "embedding.onnx"),
        ["input_ids", "image_features"],
        ["inputs_embeds"],
        dynamic_shapes={
            "input_ids": {0: torch.export.Dim("batch_size"), 1: torch.export.Dim("sequence_length")},
            "image_features": {0: torch.export.Dim("num_logical_patches")},
        },
    )
    _validate_no_ops(os.path.join(out_dir, "embedding.onnx"), {"NonZero", "ScatterND"})

    print("Exporting Qwen3.5 vision.onnx...")
    image_grid_thw = torch.tensor([[1, 20, 30]], dtype=torch.int64)
    vision = Qwen35VisionModel(config.vision_config, fixed_image_grid_thw=image_grid_thw)
    visual_state = {name[len("visual.") :]: tensor for name, tensor in state_dict.items() if name.startswith("visual.")}
    missing, unexpected = vision.load_state_dict(visual_state, strict=False)
    missing = [key for key in missing if key != "fixed_image_grid_thw"]
    if missing or unexpected:
        raise RuntimeError(f"Unexpected vision state dict keys. Missing={missing}, unexpected={unexpected}")
    vision = vision.to(dtype).eval()
    pixel_values = torch.randn((600, 3 * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2), dtype=dtype)

    _export_onnx(
        vision,
        (pixel_values,),
        os.path.join(out_dir, "vision.onnx"),
        ["pixel_values"],
        ["image_features"],
        dynamic_shapes=None,
    )
    _validate_no_ops(os.path.join(out_dir, "vision.onnx"), {"Loop", "MemcpyToHost", "MemcpyFromHost"})

    _write_processor_config(out_dir, config.vision_config)
    _patch_genai_config(out_dir, config, execution_provider)
    print("Qwen3.5 VLM auxiliary ONNX export complete.")
