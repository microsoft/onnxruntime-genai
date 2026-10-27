# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import numpy as np
import onnx_ir as ir

from .graph import Qwen4ExpGraphModel


class Qwen4ExpVisionModel(Qwen4ExpGraphModel):
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

    def make_unary(self, op_type, name, root_input, dtype, shape, **attributes):
        output = f"{name}/output_0"
        self.make_node(op_type, [root_input], [output], name=name, **attributes)
        self.make_value(output, dtype, shape)
        return output

    def make_grid_values(self):
        flat = "/vision/grid/Reshape/output_0"
        self.make_reshape(
            "/vision/grid/Reshape",
            ["image_grid_thw", "/model/constants/INT64/[-1]"],
            ir.DataType.INT64,
            [3],
        )
        values = []
        for index, label in enumerate(("t", "h", "w")):
            name = f"/vision/grid/{label}/Gather"
            self.make_gather(
                name,
                [flat, f"/model/constants/INT64/{index}"],
                ir.DataType.INT64,
                [],
                axis=0,
            )
            values.append(f"{name}/output_0")
        return values

    def make_patch_positions(self, num_patches, height, width):
        positions = "/vision/positions/Range/output_0"
        self.make_node(
            "Range",
            ["/model/constants/INT64/0", num_patches, "/model/constants/INT64/1"],
            [positions],
            name="/vision/positions/Range",
        )
        self.make_value(positions, ir.DataType.INT64, ["num_patches"])
        frame_size = self.make_binary("Mul", "/vision/grid/frame_size/Mul", [height, width], ir.DataType.INT64, [])
        within = self.make_binary(
            "Mod", "/vision/positions/within/Mod", [positions, frame_size], ir.DataType.INT64, ["num_patches"]
        )
        merge = f"/model/constants/INT64/{self.merge_size}"
        merge_sq = f"/model/constants/INT64/{self.merge_size * self.merge_size}"
        blocks_w = self.make_binary("Div", "/vision/grid/blocks_w/Div", [width, merge], ir.DataType.INT64, [])
        in_col = self.make_binary(
            "Mod", "/vision/positions/in_col/Mod", [within, merge], ir.DataType.INT64, ["num_patches"]
        )
        within_div_merge = self.make_binary(
            "Div",
            "/vision/positions/within_div_merge/Div",
            [within, merge],
            ir.DataType.INT64,
            ["num_patches"],
        )
        in_row = self.make_binary(
            "Mod",
            "/vision/positions/in_row/Mod",
            [within_div_merge, merge],
            ir.DataType.INT64,
            ["num_patches"],
        )
        within_div_block = self.make_binary(
            "Div",
            "/vision/positions/within_div_block/Div",
            [within, merge_sq],
            ir.DataType.INT64,
            ["num_patches"],
        )
        block_col = self.make_binary(
            "Mod",
            "/vision/positions/block_col/Mod",
            [within_div_block, blocks_w],
            ir.DataType.INT64,
            ["num_patches"],
        )
        row_denominator = self.make_binary(
            "Mul", "/vision/positions/row_denominator/Mul", [blocks_w, merge_sq], ir.DataType.INT64, []
        )
        block_row = self.make_binary(
            "Div",
            "/vision/positions/block_row/Div",
            [within, row_denominator],
            ir.DataType.INT64,
            ["num_patches"],
        )
        row_base = self.make_binary(
            "Mul", "/vision/positions/row_base/Mul", [block_row, merge], ir.DataType.INT64, ["num_patches"]
        )
        col_base = self.make_binary(
            "Mul", "/vision/positions/col_base/Mul", [block_col, merge], ir.DataType.INT64, ["num_patches"]
        )
        row = self.make_binary(
            "Add", "/vision/positions/row/Add", [row_base, in_row], ir.DataType.INT64, ["num_patches"]
        )
        col = self.make_binary(
            "Add", "/vision/positions/col/Add", [col_base, in_col], ir.DataType.INT64, ["num_patches"]
        )
        return row, col

    def make_axis_interpolation(self, label, position, size):
        position_float = self.make_cast(
            f"/vision/interpolation/{label}/position/Cast", position, ir.DataType.FLOAT, ["num_patches"]
        )
        size_minus_one = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/size_minus_one/Sub",
            [size, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
        )
        denominator = self.make_binary(
            "Max",
            f"/vision/interpolation/{label}/denominator/Max",
            [size_minus_one, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
        )
        denominator_float = self.make_cast(
            f"/vision/interpolation/{label}/denominator/Cast", denominator, ir.DataType.FLOAT, []
        )
        scaled = self.make_binary(
            "Mul",
            f"/vision/interpolation/{label}/scaled/Mul",
            [position_float, f"/model/constants/FLOAT/{self.visual.num_grid_per_side - 1}.0"],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        source = self.make_binary(
            "Div",
            f"/vision/interpolation/{label}/source/Div",
            [scaled, denominator_float],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        floor_float = self.make_unary(
            "Floor", f"/vision/interpolation/{label}/floor/Floor", source, ir.DataType.FLOAT, ["num_patches"]
        )
        floor_int = self.make_cast(
            f"/vision/interpolation/{label}/floor/Cast", floor_float, ir.DataType.INT64, ["num_patches"]
        )
        ceil_int = self.make_binary(
            "Add",
            f"/vision/interpolation/{label}/ceil/Add",
            [floor_int, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["num_patches"],
        )
        taps = []
        for tap_name, tap in (("floor", floor_int), ("ceil", ceil_int)):
            clipped = f"/vision/interpolation/{label}/{tap_name}/Clip/output_0"
            self.make_node(
                "Clip",
                [
                    tap,
                    "/model/constants/INT64/0",
                    f"/model/constants/INT64/{self.visual.num_grid_per_side - 1}",
                ],
                [clipped],
                name=f"/vision/interpolation/{label}/{tap_name}/Clip",
            )
            self.make_value(clipped, ir.DataType.INT64, ["num_patches"])
            unsqueezed = f"/vision/interpolation/{label}/{tap_name}/Unsqueeze/output_0"
            self.make_unsqueeze(
                f"/vision/interpolation/{label}/{tap_name}/Unsqueeze",
                [clipped, "/model/constants/INT64/[1]"],
                ir.DataType.INT64,
                ["num_patches", 1],
            )
            taps.append(unsqueezed)
        tap_indices = f"/vision/interpolation/{label}/taps/Concat/output_0"
        self.make_concat(
            f"/vision/interpolation/{label}/taps/Concat",
            taps,
            ir.DataType.INT64,
            ["num_patches", 2],
            axis=1,
        )
        fraction = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/fraction/Sub",
            [source, floor_float],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        lower_weight = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/lower_weight/Sub",
            ["/model/constants/FLOAT/1.0", fraction],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        weight_parts = []
        for weight_name, weight in (("lower", lower_weight), ("upper", fraction)):
            unsqueezed = f"/vision/interpolation/{label}/{weight_name}_weight/Unsqueeze/output_0"
            self.make_unsqueeze(
                f"/vision/interpolation/{label}/{weight_name}_weight/Unsqueeze",
                [weight, "/model/constants/INT64/[1]"],
                ir.DataType.FLOAT,
                ["num_patches", 1],
            )
            weight_parts.append(unsqueezed)
        tap_weights = f"/vision/interpolation/{label}/weights/Concat/output_0"
        self.make_concat(
            f"/vision/interpolation/{label}/weights/Concat",
            weight_parts,
            ir.DataType.FLOAT,
            ["num_patches", 2],
            axis=1,
        )
        return tap_indices, tap_weights

    def make_position_embeddings(self, row, col, height, width):
        h_taps, h_weights = self.make_axis_interpolation("h", row, height)
        w_taps, w_weights = self.make_axis_interpolation("w", col, width)
        h_taps_3d = "/vision/interpolation/h/taps/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/h/taps/Unsqueeze",
            [h_taps, "/model/constants/INT64/[2]"],
            ir.DataType.INT64,
            ["num_patches", 2, 1],
        )
        w_taps_3d = "/vision/interpolation/w/taps/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/w/taps/Unsqueeze",
            [w_taps, "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            ["num_patches", 1, 2],
        )
        h_offset = self.make_binary(
            "Mul",
            "/vision/interpolation/h_offset/Mul",
            [h_taps_3d, f"/model/constants/INT64/{self.visual.num_grid_per_side}"],
            ir.DataType.INT64,
            ["num_patches", 2, 1],
        )
        indices_3d = self.make_binary(
            "Add",
            "/vision/interpolation/indices/Add",
            [h_offset, w_taps_3d],
            ir.DataType.INT64,
            ["num_patches", 2, 2],
        )
        indices = "/vision/interpolation/indices/Reshape/output_0"
        self.make_reshape(
            "/vision/interpolation/indices/Reshape",
            [indices_3d, "/model/constants/INT64/[-1, 4]"],
            ir.DataType.INT64,
            ["num_patches", 4],
        )
        h_weights_3d = "/vision/interpolation/h/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/h/weights/Unsqueeze",
            [h_weights, "/model/constants/INT64/[2]"],
            ir.DataType.FLOAT,
            ["num_patches", 2, 1],
        )
        w_weights_3d = "/vision/interpolation/w/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/w/weights/Unsqueeze",
            [w_weights, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1, 2],
        )
        weights_3d = self.make_binary(
            "Mul",
            "/vision/interpolation/weights/Mul",
            [h_weights_3d, w_weights_3d],
            ir.DataType.FLOAT,
            ["num_patches", 2, 2],
        )
        weights = "/vision/interpolation/weights/Reshape/output_0"
        self.make_reshape(
            "/vision/interpolation/weights/Reshape",
            [weights_3d, "/model/constants/INT64/[-1, 4]"],
            ir.DataType.FLOAT,
            ["num_patches", 4],
        )
        table_name = "visual.pos_embed.weight"
        self.make_initializer(self.visual.pos_embed.weight, table_name, to=self.io_dtype)
        gathered = "/vision/pos_embed/Gather/output_0"
        self.make_gather(
            "/vision/pos_embed/Gather",
            [table_name, indices],
            self.io_dtype,
            ["num_patches", 4, self.hidden_size],
            axis=0,
        )
        if self.io_dtype != ir.DataType.FLOAT:
            weights = self.make_cast(
                "/vision/interpolation/weights/Cast",
                weights,
                self.io_dtype,
                ["num_patches", 4],
            )
        weights_expanded = "/vision/interpolation/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/weights/Unsqueeze",
            [weights, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            ["num_patches", 4, 1],
        )
        weighted = self.make_binary(
            "Mul",
            "/vision/pos_embed/Mul",
            [gathered, weights_expanded],
            self.io_dtype,
            ["num_patches", 4, self.hidden_size],
        )
        output = "/vision/pos_embed/ReduceSum/output_0"
        self.make_reduce_sum(
            "/vision/pos_embed/ReduceSum",
            [weighted, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        return output

    def make_rotary_embeddings(self, row, col):
        row_float = self.make_cast("/vision/rotary/row/Cast", row, ir.DataType.FLOAT, ["num_patches"])
        col_float = self.make_cast("/vision/rotary/col/Cast", col, ir.DataType.FLOAT, ["num_patches"])
        row_2d = "/vision/rotary/row/Unsqueeze/output_0"
        col_2d = "/vision/rotary/col/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/rotary/row/Unsqueeze",
            [row_float, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1],
        )
        self.make_unsqueeze(
            "/vision/rotary/col/Unsqueeze",
            [col_float, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1],
        )
        position_ids = "/vision/rotary/position_ids/Concat/output_0"
        self.make_concat(
            "/vision/rotary/position_ids/Concat",
            [row_2d, col_2d],
            ir.DataType.FLOAT,
            ["num_patches", 2],
            axis=1,
        )
        position_ids_3d = "/vision/rotary/position_ids/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/rotary/position_ids/Unsqueeze",
            [position_ids, "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            ["num_patches", 2, 1],
        )
        inv_freq_name = "visual.rotary_pos_emb.inv_freq"
        self.make_initializer(self.visual.rotary_pos_emb.inv_freq, inv_freq_name, to=ir.DataType.FLOAT)
        frequencies = self.make_binary(
            "Mul",
            "/vision/rotary/frequencies/Mul",
            [position_ids_3d, inv_freq_name],
            ir.DataType.FLOAT,
            ["num_patches", 2, self.head_size // 4],
        )
        flattened = "/vision/rotary/frequencies/Reshape/output_0"
        self.make_reshape(
            "/vision/rotary/frequencies/Reshape",
            [frequencies, f"/model/constants/INT64/[-1, {self.head_size // 2}]"],
            ir.DataType.FLOAT,
            ["num_patches", self.head_size // 2],
        )
        full = "/vision/rotary/frequencies/Concat/output_0"
        self.make_concat(
            "/vision/rotary/frequencies/Concat",
            [flattened, flattened],
            ir.DataType.FLOAT,
            ["num_patches", self.head_size],
            axis=1,
        )
        cos = self.make_unary("Cos", "/vision/rotary/Cos", full, ir.DataType.FLOAT, ["num_patches", self.head_size])
        sin = self.make_unary("Sin", "/vision/rotary/Sin", full, ir.DataType.FLOAT, ["num_patches", self.head_size])
        if self.io_dtype != ir.DataType.FLOAT:
            cos = self.make_cast("/vision/rotary/cos/Cast", cos, self.io_dtype, ["num_patches", self.head_size])
            sin = self.make_cast("/vision/rotary/sin/Cast", sin, self.io_dtype, ["num_patches", self.head_size])
        return cos, sin

    def apply_rotary(self, layer_id, label, tensor, cos, sin):
        basename = f"/visual/blocks/{layer_id}/attn/{label}_rotary"
        first = f"{basename}/Split/output_0"
        second = f"{basename}/Split/output_1"
        self.make_split(
            f"{basename}/Split",
            [tensor, f"/model/constants/INT64/[{self.head_size // 2}, {self.head_size // 2}]"],
            [first, second],
            [self.io_dtype, self.io_dtype],
            [["num_patches", self.num_heads, self.head_size // 2]] * 2,
            axis=-1,
        )
        negative = self.make_unary(
            "Neg", f"{basename}/Neg", second, self.io_dtype, ["num_patches", self.num_heads, self.head_size // 2]
        )
        rotated = f"{basename}/Concat/output_0"
        self.make_concat(
            f"{basename}/Concat",
            [negative, first],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
            axis=-1,
        )
        cos_3d = f"{basename}/cos/Unsqueeze/output_0"
        sin_3d = f"{basename}/sin/Unsqueeze/output_0"
        self.make_unsqueeze(
            f"{basename}/cos/Unsqueeze",
            [cos, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", 1, self.head_size],
        )
        self.make_unsqueeze(
            f"{basename}/sin/Unsqueeze",
            [sin, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", 1, self.head_size],
        )
        direct = self.make_binary(
            "Mul",
            f"{basename}/direct/Mul",
            [tensor, cos_3d],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )
        crossed = self.make_binary(
            "Mul",
            f"{basename}/crossed/Mul",
            [rotated, sin_3d],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )
        return self.make_binary(
            "Add",
            f"{basename}/Add",
            [direct, crossed],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )

    def make_frame_attention_mask(self, layer_id):
        name = f"/visual/blocks/{layer_id}/attn/frame_mask"
        frame_ids = self.make_binary(
            "Div",
            f"{name}/Div",
            ["/vision/positions/Range/output_0", "/vision/grid/frame_size/Mul/output_0"],
            ir.DataType.INT64,
            ["num_patches"],
        )
        for label, axis, shape in (("rows", 1, ["num_patches", 1]), ("cols", 0, [1, "num_patches"])):
            self.make_unsqueeze(
                f"{name}/{label}",
                [frame_ids, f"/model/constants/INT64/[{axis}]"],
                ir.DataType.INT64,
                shape,
            )
        return self.make_binary(
            "Equal",
            f"{name}/Equal",
            [f"{name}/rows/output_0", f"{name}/cols/output_0"],
            ir.DataType.BOOL,
            ["num_patches", "num_patches"],
        )

    def make_attention(self, layer_id, attention, root_input, cos, sin):
        qkv = self.make_linear(
            f"visual.blocks.{layer_id}.attn.qkv",
            attention.qkv,
            root_input,
            ["num_patches", 3 * self.hidden_size],
        )
        qkv_4d = f"/visual/blocks/{layer_id}/attn/qkv/Reshape/output_0"
        self.make_reshape(
            f"/visual/blocks/{layer_id}/attn/qkv/Reshape",
            [qkv, f"/model/constants/INT64/[-1, 3, {self.num_heads}, {self.head_size}]"],
            self.io_dtype,
            ["num_patches", 3, self.num_heads, self.head_size],
        )
        split_outputs = [f"/visual/blocks/{layer_id}/attn/qkv/Split/output_{index}" for index in range(3)]
        self.make_split(
            f"/visual/blocks/{layer_id}/attn/qkv/Split",
            [qkv_4d, "/model/constants/INT64/[1, 1, 1]"],
            split_outputs,
            [self.io_dtype] * 3,
            [["num_patches", 1, self.num_heads, self.head_size]] * 3,
            axis=1,
        )
        squeezed = []
        for label, value in zip(("q", "k", "v"), split_outputs, strict=True):
            output = f"/visual/blocks/{layer_id}/attn/{label}/Squeeze/output_0"
            self.make_squeeze(
                f"/visual/blocks/{layer_id}/attn/{label}/Squeeze",
                [value, "/model/constants/INT64/[1]"],
                self.io_dtype,
                ["num_patches", self.num_heads, self.head_size],
            )
            squeezed.append(output)
        query = self.apply_rotary(layer_id, "q", squeezed[0], cos, sin)
        key = self.apply_rotary(layer_id, "k", squeezed[1], cos, sin)
        query_t = f"/visual/blocks/{layer_id}/attn/q/Transpose/output_0"
        key_t = f"/visual/blocks/{layer_id}/attn/k/Transpose/output_0"
        value_t = f"/visual/blocks/{layer_id}/attn/v/Transpose/output_0"
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/q/Transpose",
            query,
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
            [1, 0, 2],
        )
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/k/Transpose",
            key,
            self.io_dtype,
            [self.num_heads, self.head_size, "num_patches"],
            [1, 2, 0],
        )
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/v/Transpose",
            squeezed[2],
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
            [1, 0, 2],
        )
        scores = self.make_binary(
            "MatMul",
            f"/visual/blocks/{layer_id}/attn/scores/MatMul",
            [query_t, key_t],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        scaled = self.make_binary(
            "Mul",
            f"/visual/blocks/{layer_id}/attn/scores/Mul",
            [scores, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{attention.scaling}"],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        probabilities = f"/visual/blocks/{layer_id}/attn/Softmax/output_0"
        frame_mask = self.make_frame_attention_mask(layer_id)
        masked = f"/visual/blocks/{layer_id}/attn/scores/Where"
        mask_value = f"/visual/blocks/{layer_id}/attn/mask_value"
        self.make_initializer(np.asarray(-np.inf, dtype=np.float32), mask_value, to=self.io_dtype)
        self.make_where(
            masked,
            [frame_mask, scaled, mask_value],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        self.make_softmax(
            f"/visual/blocks/{layer_id}/attn/Softmax",
            f"{masked}/output_0",
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
            axis=-1,
        )
        context = self.make_binary(
            "MatMul",
            f"/visual/blocks/{layer_id}/attn/context/MatMul",
            [probabilities, value_t],
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
        )
        context_t = f"/visual/blocks/{layer_id}/attn/context/Transpose/output_0"
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/context/Transpose",
            context,
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
            [1, 0, 2],
        )
        context_flat = f"/visual/blocks/{layer_id}/attn/context/Reshape/output_0"
        self.make_reshape(
            f"/visual/blocks/{layer_id}/attn/context/Reshape",
            [context_t, f"/model/constants/INT64/[-1, {self.hidden_size}]"],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        return self.make_linear(
            f"visual.blocks.{layer_id}.attn.proj",
            attention.proj,
            context_flat,
            ["num_patches", self.hidden_size],
        )

    def make_gelu(self, name, root_input, shape, approximate="none"):
        output = f"{name}/output_0"
        self.make_node("Gelu", [root_input], [output], name=name, approximate=approximate)
        self.make_value(output, self.io_dtype, shape)
        return output
