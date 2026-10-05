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

import glob
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import onnx
import onnx_ir as ir
import torch
from safetensors.torch import load_file

from ..base import Model
from .trt_rtx import TRT_RTX

_MAX_VISION_PATCHES = 4096


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


def load_qwen35_state_dict(
    model_name_or_path: str, token=True, cache_dir: str | None = None
) -> dict[str, torch.Tensor]:
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


def maybe_load_qwen35_config(
    model_name_or_path: str, token=True, cache_dir: str | None = None, error: Exception | None = None
):
    if error is not None and not _is_qwen35_config_error(error):
        raise error
    return load_qwen35_config(model_name_or_path, token=token, cache_dir=cache_dir)


class Qwen35VLMModel(Model):
    """Build auxiliary graphs with the same IR and serialization as the decoder."""

    constant = TRT_RTX.make_expansion_constant

    def __init__(self, config, state_dict, io_dtype, filename, out_dir):
        self.config = config
        self.state_dict = state_dict
        self.io_dtype = ir.DataType[getattr(io_dtype, "name", str(io_dtype))]
        self.onnx_dtype = self.io_dtype
        self.quant_type = None
        self.filename = filename
        self.cache_dir = out_dir
        self.values = {}
        self.node_names = set()
        self.model = ir.Model(
            ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1}, name=filename),
            ir_version=10,
        )

    def node(self, op_type, inputs, name, **attributes):
        output = f"{name}/output_0"
        self.make_node(op_type, inputs, [output], name=name, **attributes)
        return output

    def input(self, name, dtype, shape):
        self.model.graph.inputs.append(self.make_value(name, dtype, shape))
        return name

    def output(self, root_input, name, shape):
        self.make_node("Identity", [root_input], [name], name=name)
        self.model.graph.outputs.append(self.make_value(name, self.io_dtype, shape))

    def weight(self, name, tensor=None):
        self.make_initializer(self.state_dict[name] if tensor is None else tensor, name, to=self.io_dtype)
        return name

    def linear(self, root_input, name):
        weight = self.state_dict[f"{name}.weight"]
        weight = self.weight(f"{name}.weight", weight.reshape(weight.shape[0], -1).T)
        output = self.node("MatMul", [root_input, weight], f"{name}/MatMul")
        if f"{name}.bias" in self.state_dict:
            output = self.node("Add", [output, self.weight(f"{name}.bias")], f"{name}/Add")
        return output

    def norm(self, root_input, name):
        return self.node(
            "LayerNormalization",
            [root_input, self.weight(f"{name}.weight"), self.weight(f"{name}.bias")],
            f"{name}/LayerNormalization",
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )

    def make_embedding(self):
        config = self.config
        hidden_size = config.text_config.hidden_size
        ids = self.input("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        features = self.input("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        zero = self.constant("embedding/zero", 0)
        one = self.constant("embedding/one", 1)
        image_token = self.constant("embedding/image_token", config.image_token_id)
        mask = self.node("Equal", [ids, image_token], "embedding/image_mask")
        safe_ids = self.node("Where", [mask, zero, ids], "embedding/safe_ids")
        embedded = self.node("Gather", [self.weight("embed_tokens.weight"), safe_ids], "embedding/Gather", axis=0)
        flat_shape = self.constant("embedding/flat_shape", [-1])
        flat_mask = self.node("Reshape", [mask, flat_shape], "embedding/flat_mask")
        flat_mask = self.node("Cast", [flat_mask], "embedding/Cast", to=ir.DataType.INT64)
        offsets = self.node("CumSum", [flat_mask, zero], "embedding/CumSum")
        offsets = self.node("Sub", [offsets, one], "embedding/offsets")
        ids_shape = self.node("Shape", [ids], "embedding/ids_shape")
        offsets = self.node("Reshape", [offsets, ids_shape], "embedding/indices")
        feature_shape = self.node("Shape", [features], "embedding/feature_shape")
        dummy_index = self.node("Gather", [feature_shape, zero], "embedding/dummy_index", axis=0)
        indices = self.node("Where", [mask, offsets, dummy_index], "embedding/gather_indices")
        dummy = self.weight("embedding/dummy", torch.zeros(1, hidden_size))
        padded = self.node("Concat", [features, dummy], "embedding/padded_features", axis=0)
        image_embeds = self.node("Gather", [padded, indices], "embedding/image_embeddings", axis=0)
        axes = self.constant("embedding/mask_axis", [-1])
        mask = self.node("Unsqueeze", [mask, axes], "embedding/broadcast_mask")
        merged = self.node("Where", [mask, image_embeds, embedded], "embedding/Where")
        self.output(merged, "inputs_embeds", ["batch_size", "sequence_length", hidden_size])

    def make_vision_positions(self, pixels, grid):
        config = self.config.vision_config
        merge = self.constant("vision/merge_size", config.spatial_merge_size)
        merge_unit = self.constant("vision/merge_unit", config.spatial_merge_size**2)
        zero = self.constant("vision/zero", 0)
        one = self.constant("vision/one", 1)
        axis0 = self.constant("vision/axis0", [0])
        axis1 = self.constant("vision/axis1", [1])
        shape = self.node("Shape", [pixels], "vision/pixel_shape")
        count = self.node("Gather", [shape, zero], "vision/patch_count", axis=0)
        patches = self.node("Range", [zero, count, one], "vision/patch_indices")
        sizes = self.node("ReduceProd", [grid, axis1], "vision/image_sizes", keepdims=0)
        ends = self.node("CumSum", [sizes, zero], "vision/image_ends")
        starts = self.node("Sub", [ends, sizes], "vision/image_starts")
        patch_column = self.node("Unsqueeze", [patches, axis1], "vision/patch_column")
        image_ends = self.node("Unsqueeze", [ends, axis0], "vision/image_ends_row")
        previous = self.node("GreaterOrEqual", [patch_column, image_ends], "vision/previous_images")
        previous = self.node("Cast", [previous], "vision/previous_images_int", to=ir.DataType.INT64)
        image_ids = self.node("ReduceSum", [previous, axis1], "vision/image_ids", keepdims=0)
        start = self.node("Gather", [starts, image_ids], "vision/image_start", axis=0)
        local_patch = self.node("Sub", [patches, start], "vision/local_patch")
        dimensions = []
        for index, label in enumerate(("frames", "height", "width")):
            index_value = self.constant(f"vision/{label}_axis", index)
            dimensions.append(self.node("Gather", [grid, index_value], f"vision/{label}", axis=1))
        frames, heights, widths = dimensions
        height = self.node("Gather", [heights, image_ids], "vision/patch_height", axis=0)
        width = self.node("Gather", [widths, image_ids], "vision/patch_width", axis=0)
        frame_size = self.node("Mul", [height, width], "vision/frame_size")
        frame = self.node("Div", [local_patch, frame_size], "vision/local_frame")
        frame_ends = self.node("CumSum", [frames, zero], "vision/frame_ends")
        frame_starts = self.node("Sub", [frame_ends, frames], "vision/frame_starts")
        frame_start = self.node("Gather", [frame_starts, image_ids], "vision/frame_start", axis=0)
        frame_ids = self.node("Add", [frame_start, frame], "vision/frame_ids")
        local_patch = self.node("Mod", [local_patch, frame_size], "vision/frame_patch")
        block = self.node("Div", [local_patch, merge_unit], "vision/block")
        block_width = self.node("Div", [width, merge], "vision/block_width")
        block_row = self.node("Div", [block, block_width], "vision/block_row")
        block_col = self.node("Mod", [block, block_width], "vision/block_col")
        intra = self.node("Mod", [local_patch, merge_unit], "vision/intra_block")
        intra_row = self.node("Div", [intra, merge], "vision/intra_row")
        intra_col = self.node("Mod", [intra, merge], "vision/intra_col")
        row = self.node("Mul", [block_row, merge], "vision/row_start")
        col = self.node("Mul", [block_col, merge], "vision/col_start")
        row = self.node("Add", [row, intra_row], "vision/row")
        col = self.node("Add", [col, intra_col], "vision/col")
        return row, col, height, width, frame_ids

    def make_vision_position_embeddings(self, row, col, height, width):
        config = self.config.vision_config
        side = int(config.num_position_embeddings**0.5)
        side_value = self.constant("vision/position_side", side)
        maximum = self.constant("vision/position_max", side - 1)
        one = self.constant("vision/one", 1)
        one_float = self.constant("vision/one_float", 1.0, np.float32)
        maximum_float = self.constant("vision/position_max_float", side - 1, np.float32)
        axis1 = self.constant("vision/axis1", [1])
        positions = []
        for coordinate_input, dimension_input, label in ((row, height, "row"), (col, width, "col")):
            coordinate = self.node("Cast", [coordinate_input], f"vision/{label}_float", to=ir.DataType.FLOAT)
            dimension = self.node("Sub", [dimension_input, one], f"vision/{label}_extent")
            dimension = self.node("Cast", [dimension], f"vision/{label}_extent_float", to=ir.DataType.FLOAT)
            scale = self.node("Div", [maximum_float, dimension], f"vision/{label}_scale")
            position = self.node("Mul", [coordinate, scale], f"vision/{label}_position")
            floor = self.node("Cast", [position], f"vision/{label}_floor", to=ir.DataType.INT64)
            ceil = self.node("Add", [floor, one], f"vision/{label}_next")
            ceil = self.node("Min", [ceil, maximum], f"vision/{label}_ceil")
            floor_float = self.node("Cast", [floor], f"vision/{label}_floor_float", to=ir.DataType.FLOAT)
            fraction = self.node("Sub", [position, floor_float], f"vision/{label}_fraction")
            complement = self.node("Sub", [one_float, fraction], f"vision/{label}_complement")
            positions.append(((floor, ceil), (complement, fraction)))
        positional_weight = self.weight("pos_embed.weight")
        terms = []
        for h in range(2):
            for w in range(2):
                prefix = f"vision/position_{h}{w}"
                base = self.node("Mul", [positions[0][0][h], side_value], f"{prefix}/base")
                indices = self.node("Add", [base, positions[1][0][w]], f"{prefix}/indices")
                weight = self.node("Mul", [positions[0][1][h], positions[1][1][w]], f"{prefix}/weight")
                weight = self.node("Cast", [weight], f"{prefix}/Cast", to=self.io_dtype)
                weight = self.node("Unsqueeze", [weight, axis1], f"{prefix}/Unsqueeze")
                table = self.node("Gather", [positional_weight, indices], f"{prefix}/Gather", axis=0)
                terms.append(self.node("Mul", [table, weight], f"{prefix}/Mul"))
        output = terms[0]
        for index, term in enumerate(terms[1:]):
            output = self.node("Add", [output, term], f"vision/position_sum_{index}")
        head_dim = config.hidden_size // config.num_heads
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))
        frequencies = self.constant("vision/inv_freq", inv_freq, np.float32)
        coordinates = []
        for coordinate_input, label in ((row, "row"), (col, "col")):
            coordinate = self.node("Cast", [coordinate_input], f"vision/rotary_{label}_float", to=ir.DataType.FLOAT)
            coordinate = self.node("Unsqueeze", [coordinate, axis1], f"vision/rotary_{label}_column")
            coordinates.append(self.node("Mul", [coordinate, frequencies], f"vision/rotary_{label}_freq"))
        angles = self.node("Concat", coordinates, "vision/rotary_angles", axis=-1)
        angles = self.node("Concat", [angles, angles], "vision/rotary_angles_full", axis=-1)
        cos = self.node("Cos", [angles], "vision/rotary_cos")
        sin = self.node("Sin", [angles], "vision/rotary_sin")
        cos = self.node("Unsqueeze", [cos, axis1], "vision/rotary_cos_heads")
        sin = self.node("Unsqueeze", [sin, axis1], "vision/rotary_sin_heads")
        return output, cos, sin

    def make_vision(self):
        config = self.config.vision_config
        hidden_size = config.hidden_size
        head_dim = hidden_size // config.num_heads
        patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixels = self.input("pixel_values", self.io_dtype, ["num_patches", patch_size])
        grid = self.input("image_grid_thw", ir.DataType.INT64, ["num_images", 3])
        row, col, height, width, frame_ids = self.make_vision_positions(pixels, grid)
        position, cos, sin = self.make_vision_position_embeddings(row, col, height, width)
        hidden = self.linear(pixels, "patch_embed.proj")
        hidden = self.node("Add", [hidden, position], "vision/position_Add")
        frame_column = self.node("Unsqueeze", [frame_ids, self.constant("vision/axis1", [1])], "vision/frame_column")
        frame_row = self.node("Unsqueeze", [frame_ids, self.constant("vision/axis0", [0])], "vision/frame_row")
        same_frame = self.node("Equal", [frame_column, frame_row], "vision/same_frame")
        zero = self.constant("vision/mask_zero", 0.0, np.float32)
        negative = self.constant("vision/mask_negative", -10000.0, np.float32)
        bias = self.node("Where", [same_frame, zero, negative], "vision/attention_mask")
        bias = self.node("Cast", [bias], "vision/attention_mask_Cast", to=self.io_dtype)
        bias = self.node("Unsqueeze", [bias, self.constant("vision/mask_axes", [0, 1])], "vision/attention_bias")
        split_sizes = self.constant("vision/qkv_split", [hidden_size] * 3)
        head_shape = self.constant("vision/head_shape", [-1, config.num_heads, head_dim])
        attention_shape = self.constant("vision/attention_shape", [1, -1, hidden_size])
        hidden_shape = self.constant("vision/hidden_shape", [-1, hidden_size])
        half_sizes = self.constant("vision/half_split", [head_dim // 2] * 2)
        for layer in range(config.depth):
            prefix = f"blocks.{layer}"
            normalized = self.norm(hidden, f"{prefix}.norm1")
            qkv = self.linear(normalized, f"{prefix}.attn.qkv")
            q, k, v = [f"{prefix}/qkv/{label}" for label in ("q", "k", "v")]
            self.make_node("Split", [qkv, split_sizes], [q, k, v], name=f"{prefix}/qkv/Split", axis=-1)
            queries = []
            for root_input, label in ((q, "q"), (k, "k")):
                name = f"{prefix}/{label}"
                value = self.node("Reshape", [root_input, head_shape], f"{name}/Reshape")
                value = self.node("Cast", [value], f"{name}/Cast", to=ir.DataType.FLOAT)
                first, second = f"{name}/first_half", f"{name}/second_half"
                self.make_node("Split", [value, half_sizes], [first, second], name=f"{name}/Split", axis=-1)
                negative = self.node("Neg", [second], f"{name}/Neg")
                rotated = self.node("Concat", [negative, first], f"{name}/rotate_half", axis=-1)
                direct = self.node("Mul", [value, cos], f"{name}/cos_Mul")
                rotated = self.node("Mul", [rotated, sin], f"{name}/sin_Mul")
                rotated = self.node("Add", [direct, rotated], f"{name}/rotary_Add")
                rotated = self.node("Cast", [rotated], f"{name}/rotary_Cast", to=self.io_dtype)
                queries.append(self.node("Reshape", [rotated, attention_shape], f"{name}/attention_Reshape"))
            v = self.node("Reshape", [v, attention_shape], f"{prefix}/v/Reshape")
            attention = self.node(
                "MultiHeadAttention",
                [*queries, v, "", "", bias],
                f"{prefix}/MultiHeadAttention",
                domain="com.microsoft",
                num_heads=config.num_heads,
                scale=head_dim**-0.5,
            )
            attention = self.node("Reshape", [attention, hidden_shape], f"{prefix}/attention_Reshape")
            attention = self.linear(attention, f"{prefix}.attn.proj")
            hidden = self.node("Add", [hidden, attention], f"{prefix}/attention_residual")
            normalized = self.norm(hidden, f"{prefix}.norm2")
            mlp = self.linear(normalized, f"{prefix}.mlp.linear_fc1")
            if config.hidden_act == "silu":
                sigmoid = self.node("Sigmoid", [mlp], f"{prefix}/mlp/Sigmoid")
                mlp = self.node("Mul", [mlp, sigmoid], f"{prefix}/mlp/SiLU")
            else:
                mlp = self.node(
                    "Gelu", [mlp], f"{prefix}/mlp/Gelu", approximate="tanh" if "tanh" in config.hidden_act else "none"
                )
            mlp = self.linear(mlp, f"{prefix}.mlp.linear_fc2")
            hidden = self.node("Add", [hidden, mlp], f"{prefix}/mlp_residual")
        hidden = self.norm(hidden, "merger.norm")
        merged_size = hidden_size * config.spatial_merge_size**2
        hidden = self.node(
            "Reshape", [hidden, self.constant("vision/merger_shape", [-1, merged_size])], "merger/Reshape"
        )
        hidden = self.linear(hidden, "merger.linear_fc1")
        hidden = self.node("Gelu", [hidden], "merger/Gelu", approximate="none")
        hidden = self.linear(hidden, "merger.linear_fc2")
        self.output(hidden, "image_features", ["num_image_tokens", config.out_hidden_size])


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


def _validate_no_ops(model_path: str, blocked_ops: set[str]):
    model = onnx.load(model_path, load_external_data=False)
    present = sorted({node.op_type for node in model.graph.node if node.op_type in blocked_ops})
    if present:
        raise RuntimeError(f"{model_path} contains unsupported ops after export: {present}")


def _write_processor_config(out_dir: str, vision_config, execution_provider: str):
    max_pixels = 16777216
    if execution_provider == "trt-rtx":
        max_pixels = _MAX_VISION_PATCHES * vision_config.patch_size**2
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
                            "min_pixels": min(65536, max_pixels),
                            "max_pixels": max_pixels,
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
    model_config["type"] = getattr(config, "model_type", "qwen3_5")
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
        hidden_size = config.text_config.hidden_size
        vision_config = config.vision_config
        patch_size = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
        model_config["embedding"]["session_options"] = {
            "log_id": "onnxruntime-genai",
            "provider_options": [
                {
                    "NvTensorRtRtx": {
                        "enable_cuda_graph": "0",
                        "nv_profile_min_shapes": f"input_ids:1x1,image_features:0x{hidden_size}",
                        "nv_profile_opt_shapes": f"input_ids:1x609,image_features:576x{hidden_size}",
                        "nv_profile_max_shapes": f"input_ids:1x4096,image_features:{_MAX_VISION_PATCHES // vision_config.spatial_merge_size**2}x{hidden_size}",
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
                        "nv_profile_min_shapes": f"pixel_values:4x{patch_size},image_grid_thw:1x3",
                        "nv_profile_opt_shapes": f"pixel_values:2304x{patch_size},image_grid_thw:1x3",
                        "nv_profile_max_shapes": f"pixel_values:{_MAX_VISION_PATCHES}x{patch_size},image_grid_thw:8x3",
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
    state_dict = _load_qwen35_aux_state(model_name_or_path, cache_dir, token)
    print("Exporting Qwen3.5 embedding.onnx...")
    embedding = Qwen35VLMModel(
        config, {"embed_tokens.weight": state_dict["embed_tokens.weight"]}, io_dtype, "embedding.onnx", out_dir
    )
    embedding.make_embedding()
    embedding.save_model(out_dir)
    print("Exporting Qwen3.5 vision.onnx...")
    visual_state = {name[len("visual.") :]: tensor for name, tensor in state_dict.items() if name.startswith("visual.")}
    vision = Qwen35VLMModel(config, visual_state, io_dtype, "vision.onnx", out_dir)
    vision.make_vision()
    vision.save_model(out_dir)
    for filename in ("embedding.onnx", "vision.onnx"):
        _validate_no_ops(
            os.path.join(out_dir, filename), {"Loop", "NonZero", "ScatterND", "MemcpyToHost", "MemcpyFromHost"}
        )
    _write_processor_config(out_dir, config.vision_config, execution_provider)
    _patch_genai_config(out_dir, config, execution_provider)
    print("Qwen3.5 VLM auxiliary ONNX export complete.")
