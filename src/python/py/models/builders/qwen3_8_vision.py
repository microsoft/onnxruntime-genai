# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir

from .expansions.qwen3_8 import Qwen38Vision
from .qwen3_8_graph import Qwen4ExpGraphModel


class Qwen4ExpVisionModel(Qwen38Vision, Qwen4ExpGraphModel):
    def __init__(self, config, visual, io_dtype):
        super().__init__(io_dtype, "vision.onnx", "qwen4_exp_vision")
        self.config = config
        self.visual = visual
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.merge_size = config.spatial_merge_size
        patch_size = config.patch_size[0] if isinstance(config.patch_size, (list, tuple)) else config.patch_size
        temporal_size = (
            config.temporal_patch_size[0]
            if isinstance(config.temporal_patch_size, (list, tuple))
            else config.temporal_patch_size
        )
        self.patch_dim = config.in_channels * temporal_size * patch_size * patch_size
        self.make_vision_model()

    def make_vision_model(self):
        pixel_values = self.make_value("pixel_values", self.io_dtype, ["num_patches", self.patch_dim])
        image_grid = self.make_value("image_grid_thw", ir.DataType.INT64, [1, 3])
        image_features = self.make_value(
            "image_features", self.io_dtype, ["num_image_tokens", self.config.out_hidden_size]
        )
        self.graph.inputs.extend([pixel_values, image_grid])
        self.graph.outputs.append(image_features)

        shape = "/vision/pixel_values/Shape/output_0"
        self.make_shape("/vision/pixel_values/Shape", "pixel_values", [2])
        self.make_gather(
            "/vision/pixel_values/num_patches/Gather",
            [shape, "/model/constants/INT64/0"],
            ir.DataType.INT64,
            [],
            axis=0,
        )
        num_patches = "/vision/pixel_values/num_patches/Gather/output_0"
        _, height, width = self.make_grid_values()
        row, col = self.make_patch_positions(num_patches, height, width)

        patch_weight = self.visual.patch_embed.proj.weight.reshape(self.hidden_size, -1).T
        patch_weight_name = "visual.patch_embed.proj.weight"
        patch_bias_name = "visual.patch_embed.proj.bias"
        self.make_initializer(patch_weight, patch_weight_name, to=self.io_dtype)
        self.make_initializer(self.visual.patch_embed.proj.bias, patch_bias_name, to=self.io_dtype)
        patch_matmul = self.make_binary(
            "MatMul",
            "/visual/patch_embed/proj/MatMul",
            ["pixel_values", patch_weight_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/proj/Add",
            [patch_matmul, patch_bias_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        position_embeddings = self.make_position_embeddings(row, col, height, width)
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/add_position/Add",
            [hidden_states, position_embeddings],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        cos, sin = self.make_rotary_embeddings(row, col)

        for layer_id, block in enumerate(self.visual.blocks):
            norm1 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm1", block.norm1, hidden_states, ["num_patches", self.hidden_size]
            )
            attention = self.make_attention(layer_id, block.attn, norm1, cos, sin)
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/attn/residual/Add",
                [hidden_states, attention],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )
            norm2 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm2", block.norm2, hidden_states, ["num_patches", self.hidden_size]
            )
            fc1 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc1",
                block.mlp.linear_fc1,
                norm2,
                ["num_patches", self.config.intermediate_size],
            )
            activated = self.make_gelu(
                f"/visual/blocks/{layer_id}/mlp/Gelu",
                fc1,
                ["num_patches", self.config.intermediate_size],
                approximate="tanh",
            )
            fc2 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc2",
                block.mlp.linear_fc2,
                activated,
                ["num_patches", self.hidden_size],
            )
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/mlp/residual/Add",
                [hidden_states, fc2],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )

        merger = self.visual.merger
        merged_norm = self.make_layer_norm(
            "visual.merger.norm", merger.norm, hidden_states, ["num_patches", self.hidden_size]
        )
        merged_input = "/visual/merger/Reshape/output_0"
        merged_hidden_size = self.hidden_size * self.merge_size * self.merge_size
        self.make_reshape(
            "/visual/merger/Reshape",
            [merged_norm, f"/model/constants/INT64/[-1, {merged_hidden_size}]"],
            self.io_dtype,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_fc1 = self.make_linear(
            "visual.merger.linear_fc1",
            merger.linear_fc1,
            merged_input,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_activated = self.make_gelu("/visual/merger/Gelu", merger_fc1, ["num_image_tokens", merged_hidden_size])
        self.make_linear(
            "visual.merger.linear_fc2",
            merger.linear_fc2,
            merger_activated,
            ["num_image_tokens", self.config.out_hidden_size],
            output="image_features",
        )
