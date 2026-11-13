# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import math

import onnx_ir as ir
import torch
from transformers.image_processing_base import ImageProcessingMixin

from .lfm2 import LFM3Model
from .lfm2_vl import LFM2VLModel, LFM2VLVisionEncoder


class LFM3VLVisionEncoder(LFM2VLVisionEncoder):
    # The LFM3-VL vision tower and projector, for one image per run:
    # pixel_values, image_grid_thw --> image_features (the tokens of the image, block by block)

    def make_vision_config_init(self, config):
        vision_config = config.vision_config
        unsupported = sorted(set(vision_config.layer_types) - {"conv", "full_attention"})
        if unsupported:
            raise ValueError(f"Unsupported vision layer types {unsupported}: expected 'conv' or 'full_attention'")
        # The runtime's image processor writes RGB patches
        if vision_config.in_channels != 3:
            raise ValueError(f"Unsupported in_channels {vision_config.in_channels}: expected 3")

        # Standardize config attributes for the vision layers to simplify layer building logic. Model fills whatever
        # is missing here from text_config.
        config.hidden_size = vision_config.hidden_size
        config.intermediate_size = vision_config.intermediate_size
        config.num_attention_heads = vision_config.num_attention_heads
        config.num_key_value_heads = vision_config.num_key_value_heads
        config.head_dim = vision_config.hidden_size // vision_config.num_attention_heads
        config.num_hidden_layers = vision_config.num_hidden_layers
        config.layer_types = list(vision_config.layer_types)
        config.hidden_act = "silu"

        self.patch_dim = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
        self.downsample_factor = config.downsample_factor
        self.text_hidden_size = config.text_config.hidden_size

    def make_vision_attrs_init(self, vision_config):
        self.layernorm_attrs["epsilon"] = vision_config.norm_eps

    def make_rope_init(self, config):
        rope_parameters = config.vision_config.rope_parameters
        if rope_parameters["rope_type"] != "axial":
            raise ValueError(f"Unsupported vision RoPE {rope_parameters['rope_type']}: expected axial")
        self.rope_attrs["theta"] = rope_parameters["rope_theta"]

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)

    def resolve_model_loader(self):
        return self.resolve_transformers_class(LFM3Model.MULTIMODAL[self.model_type][0])

    def make_vision_inputs(self):
        self.input_names = {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"}
        self.input_types = {"pixel_values": self.io_dtype, "image_grid_thw": ir.DataType.INT64}
        self.input_shapes = {"pixel_values": ["num_patches", self.patch_dim], "image_grid_thw": [1, 3]}

    def make_image_sizes(self):
        # Make the following subgraph, which holds the image's patch rows and columns and its block rows as 1D tensors:
        #
        #   image_grid_thw [1,3] --> Reshape [3] --> Gather = rows, cols --> Div(f) = block rows, block cols
        i64, f = ir.DataType.INT64, self.downsample_factor
        grid_name = "/model/grid/Reshape"
        self.make_reshape(grid_name, [self.input_names["image_grid_thw"], "/model/constants/INT64/[-1]"], i64, [3])
        sizes = []
        for axis, name in ((1, "rows"), (2, "cols")):
            gather_name = f"/model/grid/{name}/Gather"
            gather_inputs = [f"{grid_name}/output_0", f"/model/constants/INT64/[{axis}]"]
            self.make_gather(gather_name, gather_inputs, i64, [1], axis=0)
            div_name = f"/model/grid/block_{name}/Div"
            self.make_div(div_name, [f"{gather_name}/output_0", f"/model/constants/INT64/{f}"], i64, [1])
            sizes.append(f"{gather_name}/output_0")
        return sizes

    def make_preprocessing_nodes(self):
        # Make the following subgraph, which puts the patches in raster order for every later layer and embeds them:
        #
        #   pixel_values [N,D] --> Reshape [h/f,w/f,f,f,D] --> Transpose [h/f,f,w/f,f,D] --> Reshape [1,N,D]
        #     --> patchifier --> embedding_norm --> layer 0
        basename = "/model/embeddings"
        i64, f, patch_dim = ir.DataType.INT64, self.downsample_factor, self.patch_dim
        rows, cols = self.make_image_sizes()

        blocks_shape_name = f"{basename}/blocks/Concat"
        blocks_shape_inputs = [
            "/model/grid/block_rows/Div/output_0",
            "/model/grid/block_cols/Div/output_0",
            f"/model/constants/INT64/[{f}, {f}, {patch_dim}]",
        ]
        self.make_concat(blocks_shape_name, blocks_shape_inputs, i64, [5])
        blocks_name = f"{basename}/blocks/Reshape"
        blocks_inputs = [self.input_names["pixel_values"], f"{blocks_shape_name}/output_0"]
        self.make_reshape(blocks_name, blocks_inputs, self.io_dtype, ["block_rows", "block_cols", f, f, patch_dim])
        transpose_name = f"{basename}/Transpose"
        transposed = ["block_rows", f, "block_cols", f, patch_dim]
        self.make_transpose(transpose_name, f"{blocks_name}/output_0", self.io_dtype, transposed, [0, 2, 1, 3, 4])
        raster_name = f"{basename}/raster/Reshape"
        raster_inputs = [f"{transpose_name}/output_0", f"/model/constants/INT64/[1, -1, {patch_dim}]"]
        self.make_reshape(raster_name, raster_inputs, self.io_dtype, self.make_hidden_state_shape(last_dim=patch_dim))

        tower = self.vision_tower(self.weights)
        patchifier_name = self.make_matmul(tower.patchifier, f"{basename}/patchifier/MatMul", f"{raster_name}/output_0")
        self.layernorm_attrs["root_input"] = f"{patchifier_name}/output_0"
        self.make_layernorm(0, tower.embedding_norm, skip=False, simple=True, location="embedding")
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["output_0"]

        conv_shape_inputs = [f"/model/constants/INT64/[1, {self.hidden_size}]", rows, cols]
        self.make_concat("/model/grid/conv_shape/Concat", conv_shape_inputs, i64, [4])
        self.make_rope_caches(rows, cols)

    def make_index_range(self, basename, length, dim):
        # Range(length), for a length held in a 1D tensor
        squeeze_name = f"{basename}/Squeeze"
        self.make_squeeze(squeeze_name, [length], ir.DataType.INT64, shape=[])
        range_name = f"{basename}/Range"
        range_inputs = ["/model/constants/INT64/0", f"{squeeze_name}/output_0", "/model/constants/INT64/1"]
        self.make_range(range_name, range_inputs, ir.DataType.INT64, [dim])
        return f"{range_name}/output_0"

    def make_axis_angles(self, basename, table, positions, unsqueezed, expand_shape):
        # table [S,A] --> Gather(positions) --> Unsqueeze --> Expand [h,w,A]
        axis = unsqueezed.index(1)
        angles = unsqueezed[-1]
        gather_name = f"{basename}/Gather"
        length = unsqueezed[1 - axis]
        self.make_gather(gather_name, [table, positions], self.io_dtype, [length, angles], axis=0)
        unsqueeze_name = f"{basename}/Unsqueeze"
        unsqueeze_inputs = [f"{gather_name}/output_0", f"/model/constants/INT64/[{axis}]"]
        self.make_unsqueeze(unsqueeze_name, unsqueeze_inputs, self.io_dtype, unsqueezed)
        expand_name = f"{basename}/Expand"
        self.make_expand(
            expand_name, [f"{unsqueeze_name}/output_0", expand_shape], self.io_dtype, ["rows", "cols", angles]
        )
        return f"{expand_name}/output_0"

    def make_rope_caches(self, rows, cols):
        # Make the following subgraph, the cos and sin of every patch's row and column angles, which the
        # RotaryEmbedding of each attention layer reads through position_ids = Range(N):
        #
        #   table [S,A] --> rows angles [h,w,A] --+
        #   table [S,A] --> cols angles [h,w,A] --+--> Concat [h,w,2A] --> Reshape [N,2A]
        basename = "/model/rope"
        i64 = ir.DataType.INT64
        angles = self.head_size // 4
        inv_freq = 1.0 / self.rope_attrs["theta"] ** (torch.arange(0, 2 * angles, 2).float() / (2 * angles))
        freqs = torch.outer(torch.arange(self.rope_attrs["cache_length"]).float(), inv_freq)

        rows_range = self.make_index_range(f"{basename}/rows", rows, "rows")
        cols_range = self.make_index_range(f"{basename}/cols", cols, "cols")
        count_name = f"{basename}/num_patches/Mul"
        self.make_mul(count_name, [rows, cols], i64, [1])
        position_ids = self.make_index_range(f"{basename}/position_ids", f"{count_name}/output_0", "num_patches")
        position_ids_name = f"{basename}/position_ids/Unsqueeze"
        self.make_unsqueeze(position_ids_name, [position_ids, "/model/constants/INT64/[0]"], i64, [1, "num_patches"])
        # The graph makes position_ids itself, after make_inputs_and_outputs, for the attention layers to read
        self.input_names["position_ids"] = f"{position_ids_name}/output_0"

        shape_name = f"{basename}/grid/Concat"
        self.make_concat(shape_name, [rows, cols, f"/model/constants/INT64/[{angles}]"], i64, [3])
        for kind, table in (("cos", freqs.cos()), ("sin", freqs.sin())):
            table_name = f"model.rope.{kind}_table"
            self.make_initializer(table, table_name, to=self.io_dtype)
            shape = f"{shape_name}/output_0"
            row_angles = self.make_axis_angles(
                f"{basename}/{kind}/rows", table_name, rows_range, ["rows", 1, angles], shape
            )
            col_angles = self.make_axis_angles(
                f"{basename}/{kind}/cols", table_name, cols_range, [1, "cols", angles], shape
            )
            concat_name = f"{basename}/{kind}/Concat"
            concat_shape = ["rows", "cols", 2 * angles]
            self.make_concat(concat_name, [row_angles, col_angles], self.io_dtype, concat_shape, axis=-1)
            reshape_name = f"{basename}/{kind}/Reshape"
            reshape_inputs = [f"{concat_name}/output_0", f"/model/constants/INT64/[-1, {2 * angles}]"]
            self.make_reshape(reshape_name, reshape_inputs, self.io_dtype, ["num_patches", 2 * angles])

    def make_rotary_embedding_caches(self, **kwargs):
        return "/model/rope/cos/Reshape/output_0", "/model/rope/sin/Reshape/output_0"

    def is_layer(self, module):
        return module.__class__.__name__.endswith("VisionLayer")

    def make_layer(self, layer_id, layer):
        # Each vision layer is operator_norm --> conv or attention --> ffn_norm --> MLP, with SkipLayerNorm fusing the
        # residual Add + LayerNorm
        skip = not self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(layer_id, layer.operator_norm, skip=skip, simple=True, location="operator")
        if self.layer_types[layer_id] == "conv":
            self.make_spatial_conv(layer_id, layer.conv, self.layernorm_attrs["output_0"])
        else:
            self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, layer.ffn_norm, skip=True, simple=True, location="ffn")

        mlp = layer.feed_forward
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = mlp.w1, mlp.w3, mlp.w2
        self.make_mlp(layer_id, mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # The projector's LayerNorm closes the last residual, with its own epsilon
            self.layernorm_attrs["last_layernorm"] = True
            self.layernorm_attrs["simple"] = False
            self.layernorm_attrs["epsilon"] = self.weights.model.multi_modal_projector.layer_norm.eps

    def make_spatial_conv(self, layer_id, conv, root_input):
        # Make the following subgraph, a gated depthwise convolution over the patch grid:
        #
        #   root_input [1,N,C] --> in_proj --> Split --> b, c, x
        #   Mul(b, x) --> Transpose [1,C,N] --> Reshape [1,C,h,w] --> Conv --> Reshape [1,C,N] --> Transpose [1,N,C]
        #     --> Mul(c, .) --> out_proj
        basename = f"/model/layers.{layer_id}/conv"
        hidden, dtype = self.hidden_size, self.io_dtype
        hidden_shape = self.make_hidden_state_shape()
        channels_shape = ["batch_size", hidden, "sequence_length"]
        grid_shape = ["batch_size", hidden, "rows", "cols"]

        in_proj_name = self.make_matmul(conv.in_proj, f"{basename}/in_proj/MatMul", root_input)
        split_name = f"{basename}/Split"
        gate, scale, values = (f"{split_name}/output_{i}" for i in range(3))
        split_inputs = [f"{in_proj_name}/output_0", f"/model/constants/INT64/{[hidden, hidden, hidden]}"]
        self.make_split(split_name, split_inputs, [gate, scale, values], [dtype] * 3, [hidden_shape] * 3, axis=-1)
        gated_name = f"{basename}/Mul_1"
        self.make_mul(gated_name, [gate, values], dtype, hidden_shape)

        channels_name = f"{basename}/Transpose_1"
        self.make_transpose(channels_name, f"{gated_name}/output_0", dtype, channels_shape, [0, 2, 1])
        grid_name = f"{basename}/grid/Reshape"
        grid_inputs = [f"{channels_name}/output_0", "/model/grid/conv_shape/Concat/output_0"]
        self.make_reshape(grid_name, grid_inputs, dtype, grid_shape)

        weight = f"model.layers.{layer_id}.conv.spatial_conv.weight"
        self.make_initializer(conv.spatial_conv.weight, weight, to=dtype)
        kernel = list(conv.spatial_conv.weight.shape[-2:])
        conv_name = f"{basename}/spatial_conv/Conv"
        self.make_conv(
            conv_name,
            [f"{grid_name}/output_0", weight],
            dtype,
            grid_shape,
            group=hidden,
            kernel_shape=kernel,
            pads=[size // 2 for size in kernel] * 2,
        )

        flat_name = f"{basename}/flat/Reshape"
        flat_inputs = [f"{conv_name}/output_0", "/model/constants/INT64/[0, 0, -1]"]
        self.make_reshape(flat_name, flat_inputs, dtype, channels_shape)
        hidden_name = f"{basename}/Transpose_2"
        self.make_transpose(hidden_name, f"{flat_name}/output_0", dtype, hidden_shape, [0, 2, 1])
        scaled_name = f"{basename}/Mul_2"
        self.make_mul(scaled_name, [scale, f"{hidden_name}/output_0"], dtype, hidden_shape)
        out_proj_name = self.make_matmul(conv.out_proj, f"{basename}/out_proj/MatMul", f"{scaled_name}/output_0")
        self.layernorm_attrs["skip_input"] = f"{out_proj_name}/output_0"

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Alias attribute names for compatibility with the base class
        attention.o_proj = attention.out_proj
        attention.q_norm = attention.q_layernorm
        attention.k_norm = attention.k_layernorm
        super().make_attention(layer_id, attention, root_input, **kwargs)

    def make_repeat_kv(self, layer_id, root_input, past_kv, present_kv, **kwargs):
        # Make the following subgraph, which gives query head i key or value head i // repeats. The base method's
        # subgraph is built around a KV cache, which this model does not have:
        #
        #   root_input [1,N,K*H] --> Reshape [1,N,K,1,H] --> Expand [1,N,K,repeats,H] --> Reshape [1,N,Q*H]
        projection = "k_proj" if root_input == self.attention_attrs["k_path"] else "v_proj"
        basename = f"/model/layers.{layer_id}/attn/{projection}/repeat_kv"
        repeats = self.num_attn_heads // self.num_kv_heads
        heads = ["batch_size", "sequence_length", self.num_kv_heads]
        reshape_1_name = f"{basename}/Reshape_1"
        reshape_1_inputs = [root_input, f"/model/constants/INT64/[0, 0, {self.num_kv_heads}, 1, {self.head_size}]"]
        self.make_reshape(reshape_1_name, reshape_1_inputs, self.io_dtype, [*heads, 1, self.head_size])
        expand_name = f"{basename}/Expand"
        expand_inputs = [f"{reshape_1_name}/output_0", f"/model/constants/INT64/[1, 1, 1, {repeats}, 1]"]
        self.make_expand(expand_name, expand_inputs, self.io_dtype, [*heads, repeats, self.head_size])
        reshape_2_name = f"{basename}/Reshape_2"
        reshape_2_inputs = [f"{expand_name}/output_0", f"/model/constants/INT64/[0, 0, {self.q_size}]"]
        self.make_reshape(
            reshape_2_name, reshape_2_inputs, self.io_dtype, self.make_hidden_state_shape(last_dim=self.q_size)
        )
        return f"{reshape_2_name}/output_0"

    def has_final_norm(self, module, model):
        return module is model.model.multi_modal_projector.layer_norm

    def make_postprocessing_nodes(self):
        # Make the following subgraph, which merges each f x f block of raster patches into one token, in the order
        # of the checkpoint's image features:
        #
        #   layer_norm [1,N,C] --> Reshape [h/f,f,w/f,f,C] --> Transpose [h/f,w/f,f,f,C] --> Reshape [T,f*f*C]
        #     --> linear_1 --> Gelu --> linear_2 --> image_features [T,H]
        basename = "/model/projector"
        i64, f, hidden, dtype = ir.DataType.INT64, self.downsample_factor, self.hidden_size, self.io_dtype
        token_dim = f * f * hidden

        shape_name = f"{basename}/blocks/Concat"
        shape_inputs = ["/model/grid/block_rows/Div/output_0", f"/model/constants/INT64/[{f}, -1, {f}, {hidden}]"]
        self.make_concat(shape_name, shape_inputs, i64, [5])
        blocks_name = f"{basename}/blocks/Reshape"
        blocks_inputs = [self.layernorm_attrs["output_0"], f"{shape_name}/output_0"]
        self.make_reshape(blocks_name, blocks_inputs, dtype, ["block_rows", f, "block_cols", f, hidden])
        transpose_name = f"{basename}/Transpose"
        transposed = ["block_rows", "block_cols", f, f, hidden]
        self.make_transpose(transpose_name, f"{blocks_name}/output_0", dtype, transposed, [0, 2, 1, 3, 4])
        tokens_name = f"{basename}/tokens/Reshape"
        tokens_inputs = [f"{transpose_name}/output_0", f"/model/constants/INT64/[-1, {token_dim}]"]
        self.make_reshape(tokens_name, tokens_inputs, dtype, ["num_image_tokens", token_dim])
        self.make_projector_mlp(f"{tokens_name}/output_0")


class LFM3VLModel(LFM2VLModel):
    # The lfm3_vl pipeline (exclude_embeds=true): the LFM3 decoder, plus the vision and embedding models
    # that turn images and token ids into its inputs_embeds

    vision_class = LFM3VLVisionEncoder
    decoder_class = LFM3Model
    pipeline_type = "lfm3_vl"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        if extra_options.get("use_paged_attention", False):
            raise NotImplementedError("use_paged_attention=true is not supported for the lfm3_vl pipeline yet.")
        # TODO: allow other EPs once a real LFM3-VL model has run on them.
        if ep not in {"cpu", "cuda", "webgpu"}:
            raise NotImplementedError(f"The lfm3_vl pipeline runs on cpu, cuda and webgpu for now, not {ep}.")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The longest side smart_resize can give, in patches. It refuses aspect ratios above 200 and keeps the area
        # within max_pixels, but rounds sides to the nearest multiple of the factor without scaling down, so the
        # long side can exceed sqrt(200 * max_pixels) by up to 100.5 factors.
        patch_size, merge_size = self.image_processor["patch_size"], self.image_processor["merge_size"]
        longest_side = math.isqrt(self.image_processor["size"]["longest_edge"] * 200)
        self.vision.rope_attrs["cache_length"] = (longest_side // (patch_size * merge_size) + 101) * merge_size

    def make_vision_dtype(self, io_dtype, ep):
        # WebGPU's MatMul and Gemm keep fp16 partial sums in fp16: on an M3 Ultra the 1st percentile cosine to fp32
        # PyTorch is 0.970 in fp16 and 1.0 in fp32, so the vision model stays fp32 and the embedding model casts.
        return ir.DataType.FLOAT if ep == "webgpu" else super().make_vision_dtype(io_dtype, ep)

    def read_image_processor(self, config, cache_dir):
        image_processor, _ = ImageProcessingMixin.get_image_processor_dict(
            self.decoder.model_name_or_path,
            cache_dir=cache_dir,
            token=self.decoder.hf_token,
            revision=self.decoder.revision,
        )
        vision_config = config.vision_config
        expected = {
            "resample": 3,  # Bicubic
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_convert_rgb": True,
            "patch_size": vision_config.patch_size,
            "merge_size": vision_config.spatial_merge_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
        }
        unsupported = {
            key: image_processor.get(key) for key, value in expected.items() if image_processor.get(key) != value
        }
        if unsupported:
            raise ValueError(f"Unsupported image processor settings {unsupported}: expected {expected}")
        # Lfm3VlImageProcessor lets these keys override size, which is the only place the builder reads the limits
        overrides = sorted({"min_pixels", "max_pixels"} & image_processor.keys())
        if overrides:
            raise ValueError(f"Unsupported image processor keys {overrides}: set the pixel limits in size")
        # The runtime's image processor writes each image twice, as a clip of two frames
        if vision_config.temporal_patch_size != 2:
            raise ValueError(f"Unsupported temporal_patch_size {vision_config.temporal_patch_size}: expected 2")
        if config.downsample_factor != vision_config.spatial_merge_size:
            raise ValueError(
                f"downsample_factor {config.downsample_factor} differs from "
                f"spatial_merge_size {vision_config.spatial_merge_size}"
            )
        return image_processor

    def make_processor_config(self, image_processor):
        # The checkpoint's Lfm3VlImageProcessor, for images. Its size holds the limits on an image's pixels, not its
        # edges. PatchImage writes the rows the vision model reads and stacks images of any shape.
        def step(name, kind, **attrs):
            return {"operation": {"name": name, "type": kind, "attrs": attrs}}

        patch_size, merge_size = image_processor["patch_size"], image_processor["merge_size"]
        transforms = [
            step("decode_image", "DecodeImage", color_space="RGB"),
            step(
                "resize",
                "Resize",
                interpolation="CUBIC",
                smart_resize=1,
                min_pixels=image_processor["size"]["shortest_edge"],
                max_pixels=image_processor["size"]["longest_edge"],
                patch_size=patch_size,
                merge_size=merge_size,
            ),
            step("rescale", "Rescale", rescale_factor=image_processor["rescale_factor"]),
            step("normalize", "Normalize", mean=image_processor["image_mean"], std=image_processor["image_std"]),
            step(
                "patch_image",
                "PatchImage",
                patch_size=patch_size,
                temporal_patch_size=image_processor["temporal_patch_size"],
                merge_size=merge_size,
            ),
        ]
        return {"processor": {"name": "lfm3_vl_image_processor", "transforms": transforms}}

    def make_vision_section(self):
        return {
            "patch_size": self.image_processor["patch_size"],
            "spatial_merge_size": self.image_processor["merge_size"],
            "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        }
