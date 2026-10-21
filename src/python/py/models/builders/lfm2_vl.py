# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import copy
import json
import os

import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import to_torch_dtype
from transformers.image_processing_base import ImageProcessingMixin

from .base import Model
from .lfm2 import LFM2Model


class LFM2VLVisionEncoder(Model):
    # The SigLIP2 NaFlex vision tower and the LFM2-VL projector:
    # pixel_values, pixel_attention_mask, spatial_shapes --> image_features (the tokens of all images, in order)

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # The projector is built with an exact Gelu
        if config.projector_hidden_act != "gelu":
            raise ValueError(f"Unsupported projector_hidden_act {config.projector_hidden_act}: expected gelu")

        # Standardize config attributes for the vision layers to simplify layer building logic
        vision_config = config.vision_config
        config.hidden_size = vision_config.hidden_size
        config.intermediate_size = vision_config.intermediate_size
        config.num_attention_heads = vision_config.num_attention_heads
        config.num_key_value_heads = vision_config.num_attention_heads
        config.head_dim = vision_config.hidden_size // vision_config.num_attention_heads
        config.num_hidden_layers = vision_config.num_hidden_layers
        config.layer_types = ["full_attention"] * vision_config.num_hidden_layers
        config.hidden_act = vision_config.hidden_act

        self.patch_dim = vision_config.num_channels * vision_config.patch_size**2
        self.position_embedding_size = int(vision_config.num_patches**0.5)
        self.downsample_factor = config.downsample_factor
        self.text_hidden_size = config.text_config.hidden_size

        extra_options["filename"] = "vision.onnx"
        # Model sets model_type from this, and "lfm2_vl" loads the weights through Lfm2VlForConditionalGeneration
        config.architectures = ["lfm2_vl"]
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.layernorm_attrs["simple"] = False
        self.layernorm_attrs["epsilon"] = vision_config.layer_norm_eps
        self.attention_attrs["rope"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def is_gqa_supported(self):
        # Every patch attends to every real patch, through an attention bias
        return False

    def is_packed_attn_supported(self):
        # MultiHeadAttention is the attention op with an attention bias that runs on CPU, CUDA and WebGPU
        return False

    def make_inputs_and_outputs(self):
        # Set input dicts
        self.input_names = {
            "pixel_values": "pixel_values",
            "pixel_attention_mask": "pixel_attention_mask",
            "spatial_shapes": "spatial_shapes",
        }
        self.input_types = {
            "pixel_values": self.io_dtype,
            "pixel_attention_mask": ir.DataType.INT64,
            "spatial_shapes": ir.DataType.INT64,
        }
        # batch_size is the number of images and sequence_length their padded number of patches
        self.input_shapes = {
            "pixel_values": ["batch_size", "sequence_length", self.patch_dim],
            "pixel_attention_mask": ["batch_size", "sequence_length"],
            "spatial_shapes": ["batch_size", 2],  # Patch rows and columns of each image
        }

        # Set output dicts
        self.output_names = {"image_features": "image_features"}
        self.output_types = {"image_features": self.io_dtype}
        self.output_shapes = {"image_features": ["num_image_tokens", self.text_hidden_size]}

        # Now set inputs and outputs
        super().make_inputs_and_outputs()

    def make_hidden_state_shape(self, seq_dim="sequence_length", last_dim=None):
        # The projector runs on the 2D [num_image_tokens, *] tokens of all images
        if seq_dim == "num_image_tokens":
            return [seq_dim, self.hidden_size if last_dim is None else last_dim]
        return super().make_hidden_state_shape(seq_dim, last_dim)

    def make_image_sizes(self):
        # Each image's patch rows and columns, as [batch_size, 1] columns
        sizes = []
        for axis, name in enumerate(("rows", "cols")):
            gather_name = f"/model/spatial_shapes/{name}/Gather"
            inputs = [self.input_names["spatial_shapes"], f"/model/constants/INT64/[{axis}]"]
            self.make_gather(gather_name, inputs, ir.DataType.INT64, ["batch_size", 1], axis=1)
            sizes.append(f"{gather_name}/output_0")
        return sizes

    def make_num_patches(self):
        # The padded patch count of every image, as a 1D tensor
        shape_name = "/model/num_patches/Shape"
        self.make_shape(shape_name, self.input_names["pixel_attention_mask"], shape=[1], start=1, end=2)
        return f"{shape_name}/output_0"

    def make_preprocessing_nodes(self):
        # Make the following subgraph:
        #
        #   pixel_values --> MatMul --> Add ---------------------------------------------------+
        #                                                                                       |
        #   spatial_shapes --> position weights [B,L,S*S] --> MatMul (position table) --> Cast --> Add --> layer 0
        #
        #   pixel_attention_mask --> attention bias [B,1,L,L] for every layer
        basename = "/model/embeddings"
        embeddings = self.vision_tower(self.weights).embeddings
        shape = self.make_hidden_state_shape()

        patch_name = f"{basename}/patch_embedding/MatMul"
        patch_name = self.make_matmul(embeddings.patch_embedding, patch_name, self.input_names["pixel_values"])
        patch_add_name = f"{basename}/patch_embedding/Add"
        self.make_add_bias(embeddings.patch_embedding.bias, patch_add_name, root_input=f"{patch_name}/output_0")

        # The resize runs in fp32, as in transformers on CPU
        table = "model.embeddings.position_embedding.weight"
        self.make_initializer(embeddings.position_embedding.weight, table, to=ir.DataType.FLOAT)
        weights = self.make_position_weights(f"{basename}/position_embedding")
        position_name = f"{basename}/position_embedding/MatMul"
        position_output = f"{position_name}/output_0"
        self.make_node("MatMul", inputs=[weights, table], outputs=[position_output], name=position_name)
        self.make_value(position_output, ir.DataType.FLOAT, shape=shape)
        if self.io_dtype != ir.DataType.FLOAT:
            cast_name = f"{basename}/position_embedding/Cast"
            self.make_cast(cast_name, position_output, self.io_dtype, shape)
            position_output = f"{cast_name}/output_0"

        add_name = f"{basename}/Add"
        self.make_add(add_name, [f"{patch_add_name}/output_0", position_output], self.io_dtype, shape)
        self.layernorm_attrs["root_input"] = f"{add_name}/output_0"
        self.layernorm_attrs["skip_input"] = f"{add_name}/output_0"

        self.make_attention_bias()

    def make_position_weights(self, basename):
        # Make the following subgraph, the weights each patch puts on the S x S position table. They are those of
        # transformers' F.interpolate(mode="bilinear", antialias=True) to each image's own grid, so the graph needs
        # no Resize node (WebGPU has no antialiased Resize):
        #
        #   Range(L) --> Where(patch < rows * cols, patch, 0) --> Div(cols) --> row weights [B,L,S] --> Unsqueeze --+
        #                                                     \--> Mod(cols) --> col weights [B,L,S] --> Unsqueeze --+
        #     --> Mul --> Reshape [B,L,S*S]
        i64, fp32, size = ir.DataType.INT64, ir.DataType.FLOAT, self.position_embedding_size
        patches = ["batch_size", "sequence_length"]
        rows, cols = self.make_image_sizes()

        squeeze_name = f"{basename}/Squeeze"
        self.make_squeeze(squeeze_name, [self.make_num_patches()], i64, shape=[])
        range_name = f"{basename}/Range"
        range_inputs = ["/model/constants/INT64/0", f"{squeeze_name}/output_0", "/model/constants/INT64/1"]
        self.make_range(range_name, range_inputs, i64, ["sequence_length"])

        # Padding patches take the position of patch 0, as in transformers
        area_name = f"{basename}/Mul"
        self.make_mul(area_name, [rows, cols], i64, ["batch_size", 1])
        less_name = f"{basename}/Less"
        self.make_less(less_name, [f"{range_name}/output_0", f"{area_name}/output_0"])
        where_name = f"{basename}/Where"
        where_inputs = [f"{less_name}/output_0", f"{range_name}/output_0", "/model/constants/INT64/0"]
        self.make_where(where_name, where_inputs, i64, patches)

        row_name = f"{basename}/row_index/Div"
        self.make_div(row_name, [f"{where_name}/output_0", cols], i64, patches)
        col_name = f"{basename}/col_index/Mod"
        self.make_mod(col_name, [f"{where_name}/output_0", cols], i64, patches)
        row_weights = self.make_axis_weights(f"{basename}/row", f"{row_name}/output_0", rows)
        col_weights = self.make_axis_weights(f"{basename}/col", f"{col_name}/output_0", cols)

        row_unsqueeze_name = f"{basename}/row/Unsqueeze"
        self.make_unsqueeze(row_unsqueeze_name, [row_weights, "/model/constants/INT64/[3]"], fp32, [*patches, size, 1])
        col_unsqueeze_name = f"{basename}/col/Unsqueeze"
        self.make_unsqueeze(col_unsqueeze_name, [col_weights, "/model/constants/INT64/[2]"], fp32, [*patches, 1, size])
        outer_name = f"{basename}/weights/Mul"
        outer_inputs = [f"{row_unsqueeze_name}/output_0", f"{col_unsqueeze_name}/output_0"]
        self.make_mul(outer_name, outer_inputs, fp32, [*patches, size, size])
        reshape_name = f"{basename}/weights/Reshape"
        reshape_inputs = [f"{outer_name}/output_0", f"/model/constants/INT64/[0, 0, {size * size}]"]
        self.make_reshape(reshape_name, reshape_inputs, fp32, [*patches, size * size])
        return f"{reshape_name}/output_0"

    def make_axis_weights(self, basename, index, length):
        # Make the following subgraph, the triangle filter of a bilinear resize from S to `length` samples along one
        # axis. When it shrinks (scale = S / length > 1), antialias widens the filter by the scale:
        #
        #   weight[k] = max(0, 1 - |k + 0.5 - (index + 0.5) * scale| / max(scale, 1)), normalized over k
        #
        #   length --> Cast --> Div(S, .) = scale --> Max(1) = support --> Unsqueeze ------------------+
        #   index --> Cast --> Add(0.5) --> Mul(scale) --> Unsqueeze --> Sub(k + 0.5, .) --> Div(support) --> Abs
        #     --> Sub(1, .) --> Max(0) --> Div(ReduceSum) --> [B,L,S]
        fp32, size = ir.DataType.FLOAT, self.position_embedding_size
        column, patches = ["batch_size", 1], ["batch_size", "sequence_length"]
        taps = [*patches, size]

        length_name = f"{basename}/length/Cast"
        self.make_cast(length_name, length, fp32, column)
        scale_name = f"{basename}/scale/Div"
        self.make_div(scale_name, [f"/model/constants/FLOAT/{size}", f"{length_name}/output_0"], fp32, column)
        support_name = f"{basename}/support/Max"
        self.make_max(support_name, [f"{scale_name}/output_0", "/model/constants/FLOAT/1"], fp32, column)
        support_unsqueeze_name = f"{basename}/support/Unsqueeze"
        support_unsqueeze_inputs = [f"{support_name}/output_0", "/model/constants/INT64/[2]"]
        self.make_unsqueeze(support_unsqueeze_name, support_unsqueeze_inputs, fp32, ["batch_size", 1, 1])

        index_name = f"{basename}/index/Cast"
        self.make_cast(index_name, index, fp32, patches)
        middle_name = f"{basename}/index/Add"
        self.make_add(middle_name, [f"{index_name}/output_0", "/model/constants/FLOAT/0.5"], fp32, patches)
        center_name = f"{basename}/center/Mul"
        self.make_mul(center_name, [f"{middle_name}/output_0", f"{scale_name}/output_0"], fp32, patches)
        center_unsqueeze_name = f"{basename}/center/Unsqueeze"
        center_unsqueeze_inputs = [f"{center_name}/output_0", "/model/constants/INT64/[2]"]
        self.make_unsqueeze(center_unsqueeze_name, center_unsqueeze_inputs, fp32, [*patches, 1])

        taps_name = f"{basename}/distance/Sub"
        taps_inputs = [f"/model/constants/FLOAT/{[k + 0.5 for k in range(size)]}", f"{center_unsqueeze_name}/output_0"]
        self.make_sub(taps_name, taps_inputs, fp32, taps)
        scaled_name = f"{basename}/distance/Div"
        self.make_div(scaled_name, [f"{taps_name}/output_0", f"{support_unsqueeze_name}/output_0"], fp32, taps)
        abs_name = f"{basename}/distance/Abs"
        self.make_abs(abs_name, f"{scaled_name}/output_0", fp32, taps)
        triangle_name = f"{basename}/triangle/Sub"
        self.make_sub(triangle_name, ["/model/constants/FLOAT/1", f"{abs_name}/output_0"], fp32, taps)
        relu_name = f"{basename}/triangle/Max"
        self.make_max(relu_name, [f"{triangle_name}/output_0", "/model/constants/FLOAT/0"], fp32, taps)

        sum_name = f"{basename}/ReduceSum"
        sum_inputs = [f"{relu_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_reduce_sum(sum_name, sum_inputs, fp32, [*patches, 1], keepdims=True)
        normalize_name = f"{basename}/Div"
        self.make_div(normalize_name, [f"{relu_name}/output_0", f"{sum_name}/output_0"], fp32, taps)
        return f"{normalize_name}/output_0"

    def make_attention_bias(self):
        # Make the following subgraph, the additive mask that hides the padding patches from every patch:
        #
        #   pixel_attention_mask --> Cast(bool) --> Where(0, lowest) --> Unsqueeze [B,1,1,L] --> Expand [B,1,L,L]
        basename = "/model/attn_mask"
        dtype, patches = self.to_str_dtype(self.io_dtype), ["batch_size", "sequence_length"]

        cast_name = f"{basename}/Cast"
        self.make_cast(cast_name, self.input_names["pixel_attention_mask"], ir.DataType.BOOL, patches)
        where_name = f"{basename}/Where"
        lowest = torch.finfo(to_torch_dtype(self.io_dtype)).min
        where_inputs = [f"{cast_name}/output_0", f"/model/constants/{dtype}/0", f"/model/constants/{dtype}/{lowest}"]
        self.make_where(where_name, where_inputs, self.io_dtype, patches)
        unsqueeze_name = f"{basename}/Unsqueeze"
        unsqueeze_inputs = [f"{where_name}/output_0", "/model/constants/INT64/[1, 2]"]
        self.make_unsqueeze(unsqueeze_name, unsqueeze_inputs, self.io_dtype, ["batch_size", 1, 1, "sequence_length"])

        # MultiHeadAttention needs the full [B,1,L,L] bias, not one row to broadcast
        concat_name = f"{basename}/Concat"
        concat_inputs = ["/model/constants/INT64/[1, 1]", self.make_num_patches(), "/model/constants/INT64/[1]"]
        self.make_concat(concat_name, concat_inputs, ir.DataType.INT64, [4])
        expand_name = f"{basename}/Expand"
        expand_shape = ["batch_size", 1, "sequence_length", "sequence_length"]
        self.make_expand(
            expand_name, [f"{unsqueeze_name}/output_0", f"{concat_name}/output_0"], self.io_dtype, expand_shape
        )
        self.mask_attrs["mask_name"] = expand_name

    def make_embedding(self, embedding):
        # The vision model has no token embedding
        pass

    def is_layer(self, module):
        return module.__class__.__name__.endswith("EncoderLayer")

    def make_layer(self, layer_id, layer):
        layer.input_layernorm = layer.layer_norm1
        layer.post_attention_layernorm = layer.layer_norm2
        super().make_layer(layer_id, layer)

    def make_key_value_cache_names(self, layer_id):
        # The vision model does not use any KV caches
        return "", "", "", ""

    def vision_tower(self, model):
        # Transformers before 5.6 nests the SigLIP2 tower under vision_model
        return getattr(model.model.vision_tower, "vision_model", model.model.vision_tower)

    def has_final_norm(self, module, model):
        return module is self.vision_tower(model).post_layernorm

    def has_lm_head(self, module):
        # The vision model ends with the projector
        return False

    def make_postprocessing_nodes(self):
        # Make the following subgraph, which drops the padding patches, merges each f x f block of patches into one
        # token in transformers' pixel-unshuffle order and projects the tokens of all images:
        #
        #   post_layernorm [B,L,D] --> Reshape [B*L,D] --> Gather(patch indices [T,f*f]) --> Reshape [T,f*f*D]
        #     --> LayerNorm (if the checkpoint has one) --> linear_1 --> Gelu --> linear_2 --> image_features [T,H]
        basename = "/model/projector"
        projector = self.weights.model.multi_modal_projector
        blocks, token_dim = self.downsample_factor**2, self.downsample_factor**2 * self.hidden_size

        flat_name = f"{basename}/patches/Reshape"
        flat_inputs = [self.layernorm_attrs["output_0"], f"/model/constants/INT64/[-1, {self.hidden_size}]"]
        self.make_reshape(flat_name, flat_inputs, self.io_dtype, ["num_patches", self.hidden_size])
        gather_name = f"{basename}/patches/Gather"
        gather_inputs = [f"{flat_name}/output_0", self.make_unshuffle_indices(f"{basename}/indices")]
        self.make_gather(
            gather_name, gather_inputs, self.io_dtype, ["num_image_tokens", blocks, self.hidden_size], axis=0
        )
        tokens_name = f"{basename}/tokens/Reshape"
        tokens_inputs = [f"{gather_name}/output_0", f"/model/constants/INT64/[-1, {token_dim}]"]
        self.make_reshape(tokens_name, tokens_inputs, self.io_dtype, ["num_image_tokens", token_dim])
        root_input = f"{tokens_name}/output_0"

        if projector.layer_norm is not None:
            layernorm_name = f"{basename}/layer_norm/LayerNorm"
            layernorm_output = f"{layernorm_name}/output_0"
            weight, bias = "model.projector.layer_norm.weight", "model.projector.layer_norm.bias"
            self.make_initializer(projector.layer_norm.weight, weight, to=self.io_dtype)
            self.make_initializer(projector.layer_norm.bias, bias, to=self.io_dtype)
            # nn.LayerNorm's default epsilon (1e-5), not the vision tower's
            epsilon = projector.layer_norm.eps
            layernorm_inputs = [root_input, weight, bias]
            self.make_node(
                "LayerNormalization",
                layernorm_inputs,
                [layernorm_output],
                name=layernorm_name,
                axis=-1,
                epsilon=epsilon,
                stash_type=1,
            )
            self.make_value(layernorm_output, self.io_dtype, shape=["num_image_tokens", token_dim])
            root_input = layernorm_output

        root_input = self.make_projector_linear(projector.linear_1, f"{basename}/linear_1", root_input)
        gelu_name = f"{basename}/act_fn/Gelu"
        gelu_output = f"{gelu_name}/output_0"
        self.make_node("Gelu", [root_input], [gelu_output], name=gelu_name, approximate="none")
        self.make_value(gelu_output, self.io_dtype, shape=["num_image_tokens", projector.linear_1.out_features])
        root_input = self.make_projector_linear(projector.linear_2, f"{basename}/linear_2", gelu_output)
        self.make_node("Identity", [root_input], [self.output_names["image_features"]], name=f"{basename}/Identity")

    def make_unshuffle_indices(self, basename):
        # Make the following subgraph, the rows of the [B*L,D] patches that make up each output token. Image b has
        # (rows / f) * (cols / f) tokens, and its token (I, J) holds patches (f*I + di, f*J + dj) in row-major
        # (di, dj) order:
        #
        #   spatial_shapes --> tokens per image --> CumSum = ends --+
        #   Range(sum of tokens) = t --------------------------------+--> image b = count(ends <= t)
        #     --> j = t - start of b --> I = j / (cols / f), J = j % (cols / f)
        #     --> b * L + (f*I + di) * cols + f*J + dj --> [T,f*f]
        i64, f = ir.DataType.INT64, self.downsample_factor
        column, token_column, blocks = ["batch_size", 1], ["num_image_tokens", 1], ["num_image_tokens", f * f]
        rows, cols = self.make_image_sizes()

        block_rows_name = f"{basename}/block_rows/Div"
        self.make_div(block_rows_name, [rows, f"/model/constants/INT64/{f}"], i64, column)
        block_cols_name = f"{basename}/block_cols/Div"
        self.make_div(block_cols_name, [cols, f"/model/constants/INT64/{f}"], i64, column)
        count_name = f"{basename}/count/Mul"
        self.make_mul(count_name, [f"{block_rows_name}/output_0", f"{block_cols_name}/output_0"], i64, column)
        ends_name = f"{basename}/ends/CumSum"
        self.make_cumsum(ends_name, [f"{count_name}/output_0", "/model/constants/INT64/0"], i64, column)
        starts_name = f"{basename}/starts/Sub"
        self.make_sub(starts_name, [f"{ends_name}/output_0", f"{count_name}/output_0"], i64, column)
        total_name = f"{basename}/total/ReduceSum"
        self.make_reduce_sum(total_name, [f"{count_name}/output_0"], i64, shape=[])
        range_name = f"{basename}/Range"
        range_inputs = ["/model/constants/INT64/0", f"{total_name}/output_0", "/model/constants/INT64/1"]
        self.make_range(range_name, range_inputs, i64, ["num_image_tokens"])

        # The image of each token is the number of images that end at or before it
        ended_name = f"{basename}/image/GreaterOrEqual"
        ended_inputs = [f"{range_name}/output_0", f"{ends_name}/output_0"]
        self.make_greater_or_equal(ended_name, ended_inputs, ["batch_size", "num_image_tokens"])
        ended_cast_name = f"{basename}/image/Cast"
        self.make_cast(ended_cast_name, f"{ended_name}/output_0", i64, ["batch_size", "num_image_tokens"])
        image_name = f"{basename}/image/ReduceSum"
        image_inputs = [f"{ended_cast_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_reduce_sum(image_name, image_inputs, i64, ["num_image_tokens"])
        image = f"{image_name}/output_0"

        token_name = f"{basename}/token/Unsqueeze"
        self.make_unsqueeze(token_name, [f"{range_name}/output_0", "/model/constants/INT64/[1]"], i64, token_column)
        start_name = f"{basename}/start/Gather"
        self.make_gather(start_name, [f"{starts_name}/output_0", image], i64, token_column, axis=0)
        index_name = f"{basename}/index/Sub"
        self.make_sub(index_name, [f"{token_name}/output_0", f"{start_name}/output_0"], i64, token_column)
        image_block_cols_name = f"{basename}/image_block_cols/Gather"
        self.make_gather(image_block_cols_name, [f"{block_cols_name}/output_0", image], i64, token_column, axis=0)
        image_cols_name = f"{basename}/image_cols/Gather"
        self.make_gather(image_cols_name, [cols, image], i64, token_column, axis=0)

        # Block (I, J) of the token, then the patch rows and columns of the block
        block_inputs = [f"{index_name}/output_0", f"{image_block_cols_name}/output_0"]
        block_row_name = f"{basename}/block_row/Div"
        self.make_div(block_row_name, block_inputs, i64, token_column)
        block_col_name = f"{basename}/block_col/Mod"
        self.make_mod(block_col_name, block_inputs, i64, token_column)
        top_name = f"{basename}/top/Mul"
        self.make_mul(top_name, [f"{block_row_name}/output_0", f"/model/constants/INT64/{f}"], i64, token_column)
        left_name = f"{basename}/left/Mul"
        self.make_mul(left_name, [f"{block_col_name}/output_0", f"/model/constants/INT64/{f}"], i64, token_column)
        patch_rows_name = f"{basename}/patch_rows/Add"
        row_steps = f"/model/constants/INT64/{[di for di in range(f) for _ in range(f)]}"
        self.make_add(patch_rows_name, [f"{top_name}/output_0", row_steps], i64, blocks)
        patch_cols_name = f"{basename}/patch_cols/Add"
        col_steps = f"/model/constants/INT64/{[dj for _ in range(f) for dj in range(f)]}"
        self.make_add(patch_cols_name, [f"{left_name}/output_0", col_steps], i64, blocks)

        # Row of each patch in the [B*L,D] patches
        row_start_name = f"{basename}/row_start/Mul"
        self.make_mul(row_start_name, [f"{patch_rows_name}/output_0", f"{image_cols_name}/output_0"], i64, blocks)
        patch_name = f"{basename}/patch/Add"
        self.make_add(patch_name, [f"{row_start_name}/output_0", f"{patch_cols_name}/output_0"], i64, blocks)
        image_start_name = f"{basename}/image_start/Mul"
        self.make_mul(image_start_name, [image, self.make_num_patches()], i64, ["num_image_tokens"])
        image_start_unsqueeze_name = f"{basename}/image_start/Unsqueeze"
        image_start_inputs = [f"{image_start_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_unsqueeze(image_start_unsqueeze_name, image_start_inputs, i64, token_column)
        indices_name = f"{basename}/Add"
        indices_inputs = [f"{image_start_unsqueeze_name}/output_0", f"{patch_name}/output_0"]
        self.make_add(indices_name, indices_inputs, i64, blocks)
        return f"{indices_name}/output_0"

    def make_projector_linear(self, linear, basename, root_input):
        matmul_name = self.make_matmul(linear, f"{basename}/MatMul", root_input, seq_dim="num_image_tokens")
        if linear.bias is None:
            return f"{matmul_name}/output_0"
        add_name = f"{basename}/Add"
        self.make_add_bias(linear.bias, add_name, root_input=f"{matmul_name}/output_0", seq_dim="num_image_tokens")
        return f"{add_name}/output_0"


class LFM2VLEmbedding(Model):
    # input_ids, image_features --> inputs_embeds, with the image features in place of the image tokens

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config.hidden_act = None  # Model reads the decoder's activation, which this model does not have
        extra_options["filename"] = "embeddings.onnx"
        # Model sets model_type from this, and "lfm2_vl" loads the weights through Lfm2VlForConditionalGeneration
        config.architectures = ["lfm2_vl"]
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.embed_attrs["image_token_id"] = config.image_token_id  # Token whose embeddings the image features replace
        self.embed_attrs["image_features_dtype"] = self.io_dtype  # Type of the vision model's image_features

    def make_inputs_and_outputs(self):
        # Set input dicts
        self.input_names = {"input_ids": "input_ids", "image_features": "image_features"}
        self.input_types = {"input_ids": ir.DataType.INT64, "image_features": self.embed_attrs["image_features_dtype"]}
        self.input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],
            "image_features": ["num_image_tokens", self.hidden_size],
        }

        # Set output dicts
        self.output_names = {"inputs_embeds": "inputs_embeds"}
        self.output_types = {"inputs_embeds": self.io_dtype}
        self.output_shapes = {"inputs_embeds": self.make_hidden_state_shape()}

        # Now set inputs and outputs
        super().make_inputs_and_outputs()

    def make_model(self, input_path):
        # Only the token embedding of the checkpoint, then the image features
        self.make_inputs_and_outputs()
        self.make_embedding(self.load_weights(input_path).get_input_embeddings().weight)
        self.make_postprocessing_nodes()

    def make_postprocessing_nodes(self):
        # Make the following subgraph, which writes the image features over the embeddings of the image tokens in
        # order (transformers' masked_scatter):
        #
        #   input_ids --> Equal(image_token_id) --> Reshape [-1] --> NonZero --> Transpose ---------+
        #                                                                                           |
        #   embeddings --> Reshape [-1,H] --> ScatterND <-- image_features (--> Cast) <--------------+
        #                                         |
        #                                      Reshape(Shape(embeddings)) --> inputs_embeds
        basename = "/model/embed_tokens/image_features"
        embeddings = self.layernorm_attrs["root_input"]
        flat, features_shape = ["num_tokens", self.hidden_size], ["num_image_tokens", self.hidden_size]

        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, embeddings, shape=[3])
        flat_name = f"{basename}/Reshape"
        self.make_reshape(
            flat_name, [embeddings, f"/model/constants/INT64/[-1, {self.hidden_size}]"], self.io_dtype, flat
        )

        equal_name = f"{basename}/Equal"
        image_token = f"/model/constants/INT64/{self.embed_attrs['image_token_id']}"
        self.make_equal(equal_name, [self.input_names["input_ids"], image_token], ["batch_size", "sequence_length"])
        mask_name = f"{basename}/mask/Reshape"
        self.make_reshape(
            mask_name, [f"{equal_name}/output_0", "/model/constants/INT64/[-1]"], ir.DataType.BOOL, ["num_tokens"]
        )
        nonzero_name = f"{basename}/NonZero"
        self.make_node("NonZero", [f"{mask_name}/output_0"], [f"{nonzero_name}/output_0"], name=nonzero_name)
        self.make_value(f"{nonzero_name}/output_0", ir.DataType.INT64, shape=[1, "num_image_tokens"])
        transpose_name = f"{basename}/Transpose"
        self.make_transpose(
            transpose_name, f"{nonzero_name}/output_0", ir.DataType.INT64, ["num_image_tokens", 1], [1, 0]
        )

        features = self.input_names["image_features"]
        if self.embed_attrs["image_features_dtype"] != self.io_dtype:
            cast_name = f"{basename}/Cast"
            self.make_cast(cast_name, features, self.io_dtype, features_shape)
            features = f"{cast_name}/output_0"

        scatter_name = f"{basename}/ScatterND"
        scatter_inputs = [f"{flat_name}/output_0", f"{transpose_name}/output_0", features]
        self.make_node("ScatterND", scatter_inputs, [f"{scatter_name}/output_0"], name=scatter_name)
        self.make_value(f"{scatter_name}/output_0", self.io_dtype, shape=flat)
        output_inputs = [f"{scatter_name}/output_0", f"{shape_name}/output_0"]
        self.make_node(
            "Reshape", output_inputs, [self.output_names["inputs_embeds"]], name=f"{basename}/output/Reshape"
        )


class LFM2VLModel:
    # The lfm2_vl pipeline (exclude_embeds=true): the LFM2 decoder, plus the vision and embedding models
    # that turn images and token ids into its inputs_embeds

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # The vision and embedding models are never quantized. A bf16 vision tower drifts much further from fp32 than
        # an fp16 one, so it runs in fp16 and the embedding model casts its output.
        vision_dtype = ir.DataType.FLOAT16 if io_dtype == ir.DataType.BFLOAT16 else io_dtype
        # The remaining options configure the decoder
        options = {key: extra_options[key] for key in ("hf_token", "hf_remote") if key in extra_options}
        self.vision = LFM2VLVisionEncoder(
            copy.deepcopy(config), vision_dtype, vision_dtype, ep, cache_dir, dict(options)
        )
        self.embedding = LFM2VLEmbedding(copy.deepcopy(config), io_dtype, io_dtype, ep, cache_dir, dict(options))
        self.embedding.embed_attrs["image_features_dtype"] = self.vision.io_dtype
        self.decoder = LFM2Model(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.decoder.model_type = "lfm2_vl"
        if self.decoder.filename in (self.vision.filename, self.embedding.filename):
            raise ValueError(
                f"The decoder cannot be named {self.decoder.filename}: the vision or embedding model has that name"
            )

        # Read the image processor settings before the build, so an unsupported checkpoint fails early
        self.image_processor, _ = ImageProcessingMixin.get_image_processor_dict(
            self.decoder.model_name_or_path, cache_dir=cache_dir, token=self.decoder.hf_token
        )
        self.processor_config = self.make_processor_config(self.image_processor)

    def make_processor_config(self, image_processor):
        # The Hugging Face Lfm2VlImageProcessor without image splitting (tiling)
        interpolation = {2: "LINEAR", 3: "CUBIC"}  # PIL resample codes of the Hugging Face image processor
        resample = image_processor["resample"]
        if resample not in interpolation:
            raise ValueError(f"Unsupported resample {resample}: expected 2 (bilinear) or 3 (bicubic)")
        patch_size, merge_size = image_processor["encoder_patch_size"], image_processor["downsample_factor"]
        pixels_per_token = (patch_size * merge_size) ** 2

        def step(name, kind, **attrs):
            return {"operation": {"name": name, "type": kind, **({"attrs": attrs} if attrs else {})}}

        size = image_processor["size"]
        transforms = [
            step("decode_image", "DecodeImage", color_space="RGB"),
            step(
                "resize",
                "Resize",
                height=size["height"],
                width=size["width"],
                interpolation=interpolation[resample],
                smart_resize=1,
                min_pixels=image_processor["min_image_tokens"] * pixels_per_token,
                max_pixels=image_processor["max_image_tokens"] * pixels_per_token,
                patch_size=patch_size,
                merge_size=merge_size,
            ),
            step("rescale", "Rescale", rescale_factor=image_processor["rescale_factor"]),
            step("normalize", "Normalize", mean=image_processor["image_mean"], std=image_processor["image_std"]),
            step("to_channel_first", "Permute3D", dims=[2, 0, 1]),
            step("image_sizes", "PixtralImageSizes"),
        ]
        return {"processor": {"name": "lfm2_vl_image_processor", "transforms": transforms}}

    def make_model(self, input_path):
        self.vision.make_model(input_path)
        self.embedding.make_model(input_path)
        self.decoder.make_model(input_path)

    def save_model(self, out_dir):
        self.vision.save_model(out_dir)
        self.embedding.save_model(out_dir)
        self.decoder.save_model(out_dir)

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(config, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        # The vision and embedding models run on the decoder's EP. Graph capture replays one recorded run, so it
        # stays on the decoder: the other models get new shapes for every prompt.
        model = genai_config["model"]
        graph_capture = {"enable_cuda_graph", "enable_graph_capture", "enableGraphCapture"}
        provider_options = [
            {name: {key: value for key, value in options.items() if key not in graph_capture}}
            for provider in model["decoder"]["session_options"]["provider_options"]
            for name, options in provider.items()
        ]
        session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
        model["embedding"] = {
            "filename": self.embedding.filename,
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
            "session_options": session_options,
        }
        model["vision"] = {
            "filename": self.vision.filename,
            "config_filename": "processor_config.json",
            "patch_size": self.image_processor["encoder_patch_size"],
            "spatial_merge_size": self.image_processor["downsample_factor"],
            "max_num_patches": self.image_processor["max_num_patches"],
            "inputs": {
                "pixel_values": "pixel_values",
                "attention_mask": "pixel_attention_mask",
                "image_sizes": "spatial_shapes",
            },
            "outputs": {"image_features": "image_features"},
            "session_options": session_options,
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
        with open(os.path.join(out_dir, "processor_config.json"), "w") as f:
            json.dump(self.processor_config, f, indent=2)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)
