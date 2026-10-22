# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Qwen3.5/3.6 embedding and vision graph builders, coordinated by Qwen35Model."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch
from safetensors import safe_open

from .base import Model

_MAX_VISION_PATCHES = 4096


class Qwen35EmbeddingModel(Model):
    """Merge token embeddings and image features, including image-free decode steps."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.image_token_id = config.image_token_id
        options = {"filename": "embedding.onnx", "hf_token": extra_options.get("hf_token", True)}
        super().__init__(copy.deepcopy(config), io_dtype, io_dtype, ep, cache_dir, options)

    def load_weights(self, input_path):
        source = input_path if os.path.isdir(input_path) else self.model_name_or_path
        return load_qwen35_component_weights(
            source, "model.language_model.embed_tokens.", self.cache_dir, self.hf_token
        )

    def make_model(self, input_path):
        self.weights = self.load_weights(input_path)
        hidden_size = self.hidden_size
        self.model.graph.inputs.append(
            self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        )
        ids = "input_ids"
        self.model.graph.inputs.append(
            self.make_value("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        )
        features = "image_features"
        zero = "/model/constants/INT64/0"
        one = "/model/constants/INT64/1"
        image_token = f"/model/constants/INT64/{self.image_token_id}"
        self.make_equal("embedding/image_mask", [ids, image_token], shape=None)
        mask = "embedding/image_mask/output_0"
        self.make_where("embedding/safe_ids", [mask, zero, ids], dtype=ir.DataType.INT64, shape=None)
        safe_ids = "embedding/safe_ids/output_0"
        self.make_initializer(self.weights["weight"], "embed_tokens.weight", to=self.io_dtype)
        self.make_gather("embedding/Gather", ["embed_tokens.weight", safe_ids], dtype=self.io_dtype, shape=None, axis=0)
        embedded = "embedding/Gather/output_0"
        flat_shape = "/model/constants/INT64/[-1]"
        self.make_reshape("embedding/flat_mask", [mask, flat_shape], dtype=ir.DataType.BOOL, shape=None)
        flat_mask = "embedding/flat_mask/output_0"
        self.make_cast("embedding/Cast", flat_mask, dtype=ir.DataType.INT64, shape=None)
        flat_mask = "embedding/Cast/output_0"
        self.make_cumsum("embedding/CumSum", [flat_mask, zero], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/CumSum/output_0"
        self.make_sub("embedding/offsets", [offsets, one], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/offsets/output_0"
        self.make_shape("embedding/ids_shape", ids, shape=None)
        ids_shape = "embedding/ids_shape/output_0"
        self.make_reshape("embedding/indices", [offsets, ids_shape], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/indices/output_0"
        self.make_shape("embedding/feature_shape", features, shape=None)
        feature_shape = "embedding/feature_shape/output_0"
        self.make_gather("embedding/dummy_index", [feature_shape, zero], dtype=ir.DataType.INT64, shape=None, axis=0)
        dummy_index = "embedding/dummy_index/output_0"
        self.make_where("embedding/gather_indices", [mask, offsets, dummy_index], dtype=ir.DataType.INT64, shape=None)
        indices = "embedding/gather_indices/output_0"
        self.make_initializer(torch.zeros(1, hidden_size), "embedding/dummy", to=self.io_dtype)
        dummy = "embedding/dummy"
        self.make_concat("embedding/padded_features", [features, dummy], dtype=self.io_dtype, shape=None, axis=0)
        padded = "embedding/padded_features/output_0"
        self.make_gather("embedding/image_embeddings", [padded, indices], dtype=self.io_dtype, shape=None, axis=0)
        image_embeds = "embedding/image_embeddings/output_0"
        axes = "/model/constants/INT64/[-1]"
        self.make_unsqueeze("embedding/broadcast_mask", [mask, axes], dtype=ir.DataType.BOOL, shape=None)
        mask = "embedding/broadcast_mask/output_0"
        self.make_where("embedding/Where", [mask, image_embeds, embedded], dtype=self.io_dtype, shape=None)
        merged = "embedding/Where/output_0"
        self.make_node("Identity", [merged], ["inputs_embeds"], name="inputs_embeds")
        self.model.graph.outputs.append(
            self.make_value("inputs_embeds", self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        )
        del self.weights


class Qwen35VisionModel(Model):
    """Build the vision encoder with dynamic image grids and no control-flow operators."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.vision_config = config.vision_config
        options = {"filename": "vision.onnx", "hf_token": extra_options.get("hf_token", True)}
        super().__init__(copy.deepcopy(config), io_dtype, io_dtype, ep, cache_dir, options)
        self.num_attn_heads = self.vision_config.num_heads
        self.head_size = self.vision_config.hidden_size // self.num_attn_heads
        self.attention_attrs["scale"] = self.head_size ** (-0.5)
        self.attention_attrs["unidirectional"] = False

    def load_weights(self, input_path):
        source = input_path if os.path.isdir(input_path) else self.model_name_or_path
        return load_qwen35_component_weights(source, "model.visual.", self.cache_dir, self.hf_token)

    def make_linear(self, root_input, name):
        weight = self.weights[f"{name}.weight"]
        weight_name = f"{name}.weight"
        self.make_initializer(weight.reshape(weight.shape[0], -1).T, weight_name, to=self.io_dtype)
        self.make_node("MatMul", [root_input, weight_name], [f"{name}/MatMul/output_0"], name=f"{name}/MatMul")
        self.make_value(f"{name}/MatMul/output_0", self.io_dtype)
        output = f"{name}/MatMul/output_0"
        if f"{name}.bias" in self.weights:
            bias = f"{name}.bias"
            self.make_initializer(self.weights[bias], bias, to=self.io_dtype)
            self.make_add(f"{name}/Add", [output, bias], dtype=self.io_dtype, shape=None)
            output = f"{name}/Add/output_0"
        return output

    def make_normalization(self, root_input, name):
        weight, bias = (f"{name}.weight", f"{name}.bias")
        self.make_initializer(self.weights[weight], weight, to=self.io_dtype)
        self.make_initializer(self.weights[bias], bias, to=self.io_dtype)
        self.make_node(
            "LayerNormalization",
            [root_input, weight, bias],
            [f"{name}/LayerNormalization/output_0"],
            name=f"{name}/LayerNormalization",
            axis=-1,
            epsilon=1e-06,
            stash_type=1,
        )
        self.make_value(f"{name}/LayerNormalization/output_0", self.io_dtype)
        return f"{name}/LayerNormalization/output_0"

    def make_vision_positions(self, pixels, grid):
        config = self.vision_config
        merge = f"/model/constants/INT64/{config.spatial_merge_size}"
        merge_unit = f"/model/constants/INT64/{config.spatial_merge_size ** 2}"
        zero = "/model/constants/INT64/0"
        one = "/model/constants/INT64/1"
        axis0 = "/model/constants/INT64/[0]"
        axis1 = "/model/constants/INT64/[1]"
        self.make_shape("vision/pixel_shape", pixels, shape=None)
        shape = "vision/pixel_shape/output_0"
        self.make_gather("vision/patch_count", [shape, zero], dtype=ir.DataType.INT64, shape=None, axis=0)
        count = "vision/patch_count/output_0"
        self.make_range("vision/patch_indices", [zero, count, one], dtype=ir.DataType.INT64, shape=None)
        patches = "vision/patch_indices/output_0"
        self.make_node(
            "ReduceProd", [grid, axis1], ["vision/image_sizes/output_0"], name="vision/image_sizes", keepdims=0
        )
        self.make_value("vision/image_sizes/output_0", ir.DataType.INT64)
        sizes = "vision/image_sizes/output_0"
        self.make_cumsum("vision/image_ends", [sizes, zero], dtype=ir.DataType.INT64, shape=None)
        ends = "vision/image_ends/output_0"
        self.make_sub("vision/image_starts", [ends, sizes], dtype=ir.DataType.INT64, shape=None)
        starts = "vision/image_starts/output_0"
        self.make_unsqueeze("vision/patch_column", [patches, axis1], dtype=ir.DataType.INT64, shape=None)
        patch_column = "vision/patch_column/output_0"
        self.make_unsqueeze("vision/image_ends_row", [ends, axis0], dtype=ir.DataType.INT64, shape=None)
        image_ends = "vision/image_ends_row/output_0"
        self.make_greater_or_equal("vision/previous_images", [patch_column, image_ends], shape=None)
        previous = "vision/previous_images/output_0"
        self.make_cast("vision/previous_images_int", previous, dtype=ir.DataType.INT64, shape=None)
        previous = "vision/previous_images_int/output_0"
        self.make_reduce_sum("vision/image_ids", [previous, axis1], dtype=ir.DataType.INT64, shape=None, keepdims=0)
        image_ids = "vision/image_ids/output_0"
        self.make_gather("vision/image_start", [starts, image_ids], dtype=ir.DataType.INT64, shape=None, axis=0)
        start = "vision/image_start/output_0"
        self.make_sub("vision/local_patch", [patches, start], dtype=ir.DataType.INT64, shape=None)
        local_patch = "vision/local_patch/output_0"
        dimensions = []
        for index, label in enumerate(("frames", "height", "width")):
            index_value = f"/model/constants/INT64/{index}"
            self.make_gather(f"vision/{label}", [grid, index_value], dtype=ir.DataType.INT64, shape=None, axis=1)
            dimensions.append(f"vision/{label}/output_0")
        frames, heights, widths = dimensions
        self.make_gather("vision/patch_height", [heights, image_ids], dtype=ir.DataType.INT64, shape=None, axis=0)
        height = "vision/patch_height/output_0"
        self.make_gather("vision/patch_width", [widths, image_ids], dtype=ir.DataType.INT64, shape=None, axis=0)
        width = "vision/patch_width/output_0"
        self.make_mul("vision/frame_size", [height, width], dtype=ir.DataType.INT64, shape=None)
        frame_size = "vision/frame_size/output_0"
        self.make_div("vision/local_frame", [local_patch, frame_size], dtype=ir.DataType.INT64, shape=None)
        frame = "vision/local_frame/output_0"
        self.make_cumsum("vision/frame_ends", [frames, zero], dtype=ir.DataType.INT64, shape=None)
        frame_ends = "vision/frame_ends/output_0"
        self.make_sub("vision/frame_starts", [frame_ends, frames], dtype=ir.DataType.INT64, shape=None)
        frame_starts = "vision/frame_starts/output_0"
        self.make_gather("vision/frame_start", [frame_starts, image_ids], dtype=ir.DataType.INT64, shape=None, axis=0)
        frame_start = "vision/frame_start/output_0"
        self.make_add("vision/frame_ids", [frame_start, frame], dtype=ir.DataType.INT64, shape=None)
        frame_ids = "vision/frame_ids/output_0"
        self.make_node("Mod", [local_patch, frame_size], ["vision/frame_patch/output_0"], name="vision/frame_patch")
        self.make_value("vision/frame_patch/output_0", ir.DataType.INT64)
        local_patch = "vision/frame_patch/output_0"
        self.make_div("vision/block", [local_patch, merge_unit], dtype=ir.DataType.INT64, shape=None)
        block = "vision/block/output_0"
        self.make_div("vision/block_width", [width, merge], dtype=ir.DataType.INT64, shape=None)
        block_width = "vision/block_width/output_0"
        self.make_div("vision/block_row", [block, block_width], dtype=ir.DataType.INT64, shape=None)
        block_row = "vision/block_row/output_0"
        self.make_node("Mod", [block, block_width], ["vision/block_col/output_0"], name="vision/block_col")
        self.make_value("vision/block_col/output_0", ir.DataType.INT64)
        block_col = "vision/block_col/output_0"
        self.make_node("Mod", [local_patch, merge_unit], ["vision/intra_block/output_0"], name="vision/intra_block")
        self.make_value("vision/intra_block/output_0", ir.DataType.INT64)
        intra = "vision/intra_block/output_0"
        self.make_div("vision/intra_row", [intra, merge], dtype=ir.DataType.INT64, shape=None)
        intra_row = "vision/intra_row/output_0"
        self.make_node("Mod", [intra, merge], ["vision/intra_col/output_0"], name="vision/intra_col")
        self.make_value("vision/intra_col/output_0", ir.DataType.INT64)
        intra_col = "vision/intra_col/output_0"
        self.make_mul("vision/row_start", [block_row, merge], dtype=ir.DataType.INT64, shape=None)
        row = "vision/row_start/output_0"
        self.make_mul("vision/col_start", [block_col, merge], dtype=ir.DataType.INT64, shape=None)
        col = "vision/col_start/output_0"
        self.make_add("vision/row", [row, intra_row], dtype=ir.DataType.INT64, shape=None)
        row = "vision/row/output_0"
        self.make_add("vision/col", [col, intra_col], dtype=ir.DataType.INT64, shape=None)
        col = "vision/col/output_0"
        return (row, col, height, width, frame_ids)

    def make_vision_position_embeddings(self, row, col, height, width):
        config = self.vision_config
        side = int(config.num_position_embeddings**0.5)
        side_value = f"/model/constants/INT64/{side}"
        maximum = f"/model/constants/INT64/{side - 1}"
        one = "/model/constants/INT64/1"
        one_float = "/model/constants/FLOAT/1.0"
        maximum_float = f"/model/constants/FLOAT/{side - 1}"
        axis1 = "/model/constants/INT64/[1]"
        positions = []
        for coordinate_input, dimension_input, label in ((row, height, "row"), (col, width, "col")):
            self.make_cast(f"vision/{label}_float", coordinate_input, dtype=ir.DataType.FLOAT, shape=None)
            coordinate = f"vision/{label}_float/output_0"
            self.make_sub(f"vision/{label}_extent", [dimension_input, one], dtype=ir.DataType.INT64, shape=None)
            dimension = f"vision/{label}_extent/output_0"
            self.make_cast(f"vision/{label}_extent_float", dimension, dtype=ir.DataType.FLOAT, shape=None)
            dimension = f"vision/{label}_extent_float/output_0"
            self.make_div(f"vision/{label}_scale", [maximum_float, dimension], dtype=ir.DataType.FLOAT, shape=None)
            scale = f"vision/{label}_scale/output_0"
            self.make_mul(f"vision/{label}_position", [coordinate, scale], dtype=ir.DataType.FLOAT, shape=None)
            position = f"vision/{label}_position/output_0"
            self.make_cast(f"vision/{label}_floor", position, dtype=ir.DataType.INT64, shape=None)
            floor = f"vision/{label}_floor/output_0"
            self.make_add(f"vision/{label}_next", [floor, one], dtype=ir.DataType.INT64, shape=None)
            ceil = f"vision/{label}_next/output_0"
            self.make_min(f"vision/{label}_ceil", [ceil, maximum], dtype=ir.DataType.INT64, shape=None)
            ceil = f"vision/{label}_ceil/output_0"
            self.make_cast(f"vision/{label}_floor_float", floor, dtype=ir.DataType.FLOAT, shape=None)
            floor_float = f"vision/{label}_floor_float/output_0"
            self.make_sub(f"vision/{label}_fraction", [position, floor_float], dtype=ir.DataType.FLOAT, shape=None)
            fraction = f"vision/{label}_fraction/output_0"
            self.make_sub(f"vision/{label}_complement", [one_float, fraction], dtype=ir.DataType.FLOAT, shape=None)
            complement = f"vision/{label}_complement/output_0"
            positions.append(((floor, ceil), (complement, fraction)))
        self.make_initializer(self.weights["pos_embed.weight"], "pos_embed.weight", to=self.io_dtype)
        positional_weight = "pos_embed.weight"
        terms = []
        for h in range(2):
            for w in range(2):
                prefix = f"vision/position_{h}{w}"
                self.make_mul(f"{prefix}/base", [positions[0][0][h], side_value], dtype=ir.DataType.INT64, shape=None)
                base = f"{prefix}/base/output_0"
                self.make_add(f"{prefix}/indices", [base, positions[1][0][w]], dtype=ir.DataType.INT64, shape=None)
                indices = f"{prefix}/indices/output_0"
                self.make_mul(
                    f"{prefix}/weight", [positions[0][1][h], positions[1][1][w]], dtype=ir.DataType.FLOAT, shape=None
                )
                weight = f"{prefix}/weight/output_0"
                self.make_cast(f"{prefix}/Cast", weight, dtype=self.io_dtype, shape=None)
                weight = f"{prefix}/Cast/output_0"
                self.make_unsqueeze(f"{prefix}/Unsqueeze", [weight, axis1], dtype=self.io_dtype, shape=None)
                weight = f"{prefix}/Unsqueeze/output_0"
                self.make_gather(
                    f"{prefix}/Gather", [positional_weight, indices], dtype=self.io_dtype, shape=None, axis=0
                )
                table = f"{prefix}/Gather/output_0"
                self.make_mul(f"{prefix}/Mul", [table, weight], dtype=self.io_dtype, shape=None)
                terms.append(f"{prefix}/Mul/output_0")
        output = terms[0]
        for index, term in enumerate(terms[1:]):
            self.make_add(f"vision/position_sum_{index}", [output, term], dtype=self.io_dtype, shape=None)
            output = f"vision/position_sum_{index}/output_0"
        head_dim = config.hidden_size // config.num_heads
        inv_freq = 1.0 / 10000.0 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2))
        frequencies = f"/model/constants/FLOAT/{inv_freq.tolist()}"
        coordinates = []
        for coordinate_input, label in ((row, "row"), (col, "col")):
            self.make_cast(f"vision/rotary_{label}_float", coordinate_input, dtype=ir.DataType.FLOAT, shape=None)
            coordinate = f"vision/rotary_{label}_float/output_0"
            self.make_unsqueeze(
                f"vision/rotary_{label}_column", [coordinate, axis1], dtype=ir.DataType.FLOAT, shape=None
            )
            coordinate = f"vision/rotary_{label}_column/output_0"
            self.make_mul(f"vision/rotary_{label}_freq", [coordinate, frequencies], dtype=ir.DataType.FLOAT, shape=None)
            coordinates.append(f"vision/rotary_{label}_freq/output_0")
        self.make_concat("vision/rotary_angles", coordinates, dtype=ir.DataType.FLOAT, shape=None, axis=-1)
        angles = "vision/rotary_angles/output_0"
        self.make_concat("vision/rotary_angles_full", [angles, angles], dtype=ir.DataType.FLOAT, shape=None, axis=-1)
        angles = "vision/rotary_angles_full/output_0"
        self.make_cos("vision/rotary_cos", angles, dtype=ir.DataType.FLOAT, shape=None)
        cos = "vision/rotary_cos/output_0"
        self.make_sin("vision/rotary_sin", angles, dtype=ir.DataType.FLOAT, shape=None)
        sin = "vision/rotary_sin/output_0"
        self.make_unsqueeze("vision/rotary_cos_heads", [cos, axis1], dtype=ir.DataType.FLOAT, shape=None)
        cos = "vision/rotary_cos_heads/output_0"
        self.make_unsqueeze("vision/rotary_sin_heads", [sin, axis1], dtype=ir.DataType.FLOAT, shape=None)
        sin = "vision/rotary_sin_heads/output_0"
        return (output, cos, sin)

    def make_model(self, input_path):
        self.weights = self.load_weights(input_path)
        config = self.vision_config
        hidden_size = config.hidden_size
        head_dim = hidden_size // config.num_heads
        patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.model.graph.inputs.append(self.make_value("pixel_values", self.io_dtype, ["num_patches", patch_size]))
        pixels = "pixel_values"
        self.model.graph.inputs.append(self.make_value("image_grid_thw", ir.DataType.INT64, ["num_images", 3]))
        grid = "image_grid_thw"
        row, col, height, width, frame_ids = self.make_vision_positions(pixels, grid)
        position, cos, sin = self.make_vision_position_embeddings(row, col, height, width)
        hidden = self.make_linear(pixels, "patch_embed.proj")
        self.make_add("vision/position_Add", [hidden, position], dtype=self.io_dtype, shape=None)
        hidden = "vision/position_Add/output_0"
        self.make_unsqueeze(
            "vision/frame_column", [frame_ids, "/model/constants/INT64/[1]"], dtype=ir.DataType.INT64, shape=None
        )
        frame_column = "vision/frame_column/output_0"
        self.make_unsqueeze(
            "vision/frame_row", [frame_ids, "/model/constants/INT64/[0]"], dtype=ir.DataType.INT64, shape=None
        )
        frame_row = "vision/frame_row/output_0"
        self.make_equal("vision/same_frame", [frame_column, frame_row], shape=None)
        same_frame = "vision/same_frame/output_0"
        zero = "/model/constants/FLOAT/0.0"
        negative = "/model/constants/FLOAT/-10000.0"
        self.make_where("vision/attention_mask", [same_frame, zero, negative], dtype=ir.DataType.FLOAT, shape=None)
        bias = "vision/attention_mask/output_0"
        self.make_cast("vision/attention_mask_Cast", bias, dtype=self.io_dtype, shape=None)
        bias = "vision/attention_mask_Cast/output_0"
        self.make_unsqueeze(
            "vision/attention_bias", [bias, "/model/constants/INT64/[0, 1]"], dtype=self.io_dtype, shape=None
        )
        bias = "vision/attention_bias/output_0"
        split_sizes = f"/model/constants/INT64/{[hidden_size] * 3}"
        head_shape = f"/model/constants/INT64/{[-1, config.num_heads, head_dim]}"
        attention_shape = f"/model/constants/INT64/{[1, -1, hidden_size]}"
        hidden_shape = f"/model/constants/INT64/{[-1, hidden_size]}"
        half_sizes = f"/model/constants/INT64/{[head_dim // 2] * 2}"
        for layer in range(config.depth):
            prefix = f"blocks.{layer}"
            normalized = self.make_normalization(hidden, f"{prefix}.norm1")
            qkv = self.make_linear(normalized, f"{prefix}.attn.qkv")
            q, k, v = [f"{prefix}/qkv/{label}" for label in ("q", "k", "v")]
            self.make_split(
                f"{prefix}/qkv/Split",
                [qkv, split_sizes],
                [q, k, v],
                [self.io_dtype] * 3,
                [["num_patches", hidden_size]] * 3,
            )
            queries = []
            for root_input, label in ((q, "q"), (k, "k")):
                name = f"{prefix}/{label}"
                self.make_reshape(f"{name}/Reshape", [root_input, head_shape], dtype=self.io_dtype, shape=None)
                value = f"{name}/Reshape/output_0"
                self.make_cast(f"{name}/Cast", value, dtype=ir.DataType.FLOAT, shape=None)
                value = f"{name}/Cast/output_0"
                first, second = (f"{name}/first_half", f"{name}/second_half")
                self.make_split(
                    f"{name}/Split",
                    [value, half_sizes],
                    [first, second],
                    [ir.DataType.FLOAT] * 2,
                    [["num_patches", config.num_heads, head_dim // 2]] * 2,
                )
                self.make_neg(f"{name}/Neg", second, dtype=ir.DataType.FLOAT, shape=None)
                negative = f"{name}/Neg/output_0"
                self.make_concat(f"{name}/rotate_half", [negative, first], dtype=ir.DataType.FLOAT, shape=None, axis=-1)
                rotated = f"{name}/rotate_half/output_0"
                self.make_mul(f"{name}/cos_Mul", [value, cos], dtype=ir.DataType.FLOAT, shape=None)
                direct = f"{name}/cos_Mul/output_0"
                self.make_mul(f"{name}/sin_Mul", [rotated, sin], dtype=ir.DataType.FLOAT, shape=None)
                rotated = f"{name}/sin_Mul/output_0"
                self.make_add(f"{name}/rotary_Add", [direct, rotated], dtype=ir.DataType.FLOAT, shape=None)
                rotated = f"{name}/rotary_Add/output_0"
                self.make_cast(f"{name}/rotary_Cast", rotated, dtype=self.io_dtype, shape=None)
                rotated = f"{name}/rotary_Cast/output_0"
                self.make_reshape(
                    f"{name}/attention_Reshape", [rotated, attention_shape], dtype=self.io_dtype, shape=None
                )
                queries.append(f"{name}/attention_Reshape/output_0")
            self.make_reshape(f"{prefix}/v/Reshape", [v, attention_shape], dtype=self.io_dtype, shape=None)
            v = f"{prefix}/v/Reshape/output_0"
            self.make_multi_head_attention(
                f"{prefix}/MultiHeadAttention", q_path=queries[0], k_path=queries[1], v_path=v, add_qk=bias
            )
            attention = f"{prefix}/MultiHeadAttention/output_0"
            self.make_value(attention, self.io_dtype, [1, "num_patches", hidden_size])
            self.make_reshape(f"{prefix}/attention_Reshape", [attention, hidden_shape], dtype=self.io_dtype, shape=None)
            attention = f"{prefix}/attention_Reshape/output_0"
            attention = self.make_linear(attention, f"{prefix}.attn.proj")
            self.make_add(f"{prefix}/attention_residual", [hidden, attention], dtype=self.io_dtype, shape=None)
            hidden = f"{prefix}/attention_residual/output_0"
            normalized = self.make_normalization(hidden, f"{prefix}.norm2")
            mlp = self.make_linear(normalized, f"{prefix}.mlp.linear_fc1")
            if config.hidden_act == "silu":
                self.make_sigmoid(f"{prefix}/mlp/Sigmoid", mlp, dtype=self.io_dtype, shape=None)
                sigmoid = f"{prefix}/mlp/Sigmoid/output_0"
                self.make_mul(f"{prefix}/mlp/SiLU", [mlp, sigmoid], dtype=self.io_dtype, shape=None)
                mlp = f"{prefix}/mlp/SiLU/output_0"
            else:
                self.make_node(
                    "Gelu",
                    [mlp],
                    [f"{prefix}/mlp/Gelu/output_0"],
                    name=f"{prefix}/mlp/Gelu",
                    approximate="tanh" if "tanh" in config.hidden_act else "none",
                )
                self.make_value(f"{prefix}/mlp/Gelu/output_0", self.io_dtype)
                mlp = f"{prefix}/mlp/Gelu/output_0"
            mlp = self.make_linear(mlp, f"{prefix}.mlp.linear_fc2")
            self.make_add(f"{prefix}/mlp_residual", [hidden, mlp], dtype=self.io_dtype, shape=None)
            hidden = f"{prefix}/mlp_residual/output_0"
        hidden = self.make_normalization(hidden, "merger.norm")
        merged_size = hidden_size * config.spatial_merge_size**2
        self.make_reshape(
            "merger/Reshape", [hidden, f"/model/constants/INT64/{[-1, merged_size]}"], dtype=self.io_dtype, shape=None
        )
        hidden = "merger/Reshape/output_0"
        hidden = self.make_linear(hidden, "merger.linear_fc1")
        self.make_node("Gelu", [hidden], ["merger/Gelu/output_0"], name="merger/Gelu", approximate="none")
        self.make_value("merger/Gelu/output_0", self.io_dtype)
        hidden = "merger/Gelu/output_0"
        hidden = self.make_linear(hidden, "merger.linear_fc2")
        self.make_node("Identity", [hidden], ["image_features"], name="image_features")
        self.model.graph.outputs.append(
            self.make_value("image_features", self.io_dtype, ["num_image_tokens", config.out_hidden_size])
        )
        del self.weights


def load_qwen35_component_weights(model_name_or_path: str, prefix: str, cache_dir: str | None, token):
    """Load only the requested component, without materializing the decoder shards."""
    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            model_name_or_path, cache_dir=cache_dir, token=token, allow_patterns=["*.json", "*.safetensors"]
        )
    safetensor_files = sorted(Path(model_dir).glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")
    state_dict = {}
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as tensors:
            for name in sorted(tensors.keys()):
                if name.startswith(prefix):
                    state_dict[name[len(prefix) :]] = tensors.get_tensor(name)
    if not state_dict:
        raise ValueError(f"No weights with prefix {prefix} found under {model_dir}")
    return state_dict


def save_qwen35_processor(out_dir: str, vision_config, execution_provider: str):
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
                {"operation": {"name": "rescale", "type": "Rescale", "attrs": {"rescale_factor": 1.0 / 255.0}}},
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


def make_qwen35_vlm_config(out_dir: str, config, execution_provider: str):
    genai_path = os.path.join(out_dir, "genai_config.json")
    with open(genai_path) as f:
        genai_config = json.load(f)
    model_config = genai_config["model"]
    model_config["type"] = config.model_type
    model_config["image_token_id"] = config.image_token_id
    model_config["video_token_id"] = config.video_token_id
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
    if execution_provider == "cuda":
        for component in ("embedding", "vision"):
            model_config[component]["session_options"] = {
                "provider_options": [{"cuda": {"enable_cuda_graph": "0"}}],
            }
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
                        "nv_profile_max_shapes": f"input_ids:1x4096,image_features:{_MAX_VISION_PATCHES // vision_config.spatial_merge_size ** 2}x{hidden_size}",
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
