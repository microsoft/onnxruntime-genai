# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx_ir as ir
import torch

from .base import Model
from .mistral import MistralModel


class PhiModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def make_layer(self, layer_id, layer):
        # Each Phi decoder layer is defined as:
        # input_layernorm --> attention --> MLP --> residual_add
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        residual_add_name = f"/model/layers.{layer_id}/residual_add/Add"
        residual_add_inputs = [self.layernorm_attrs["skip_input"], self.mlp_attrs["output_0"]]
        self.make_add(
            residual_add_name,
            residual_add_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

        # Assign output 0 of residual Add as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_add_name}/output_0"


class Phi3MiniModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MiniLongRoPEModel(Phi3MiniModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()

        # Set position_ids_name based on whether position_ids is available as an input
        if "position_ids" in self.input_names:
            position_ids_result = self.make_position_ids_reformatting()
            self.position_ids_name = (
                f"{position_ids_result}/output_0" if position_ids_result != "position_ids" else "position_ids"
            )
        else:
            # When position_ids is not an input (use_rope_in_attn is True),
            # position_ids won't be used since rotary embeddings are handled in GQA
            self.position_ids_name = None

    def make_position_ids_reformatting(self):
        if self.ep not in self.eps_without_if_support:
            position_ids_input_to_rotemb = super().make_position_ids_reformatting()
            return position_ids_input_to_rotemb

        # Make Long RoPE scaling subgraph to adjust position_ids for sequences beyond original context length
        # For WebGPU: use int32 for all ops due to limited int64 support, then cast back to int64
        # For other EPs: use native int64 throughout
        #
        # WebGPU graph:                              Other EPs graph:
        #   position_ids                               position_ids
        #        |                                          |
        #   Cast (int32)                               ReduceMax (int64)
        #        |                                          |
        #   ReduceMax (int32)                          GreaterOrEqual
        #        |                                          |
        #   GreaterOrEqual                             Cast (int64)
        #        |                                          |
        #   Cast (int32)                               Mul (int64)
        #        |                                          |
        #   Mul (int32)                                Add (int64)
        #        |
        #   Add (int32)
        #        |
        #   Cast (int64)

        basename = "/model/pos_ids_reformat"
        proto_dtype = self.input_types["position_ids"]

        # For WebGPU, use int32 for computation due to limited int64 ops support
        is_webgpu = self.extra_options.get("enable_webgpu_graph", False)
        compute_dtype = ir.DataType.INT32 if is_webgpu else proto_dtype
        compute_str_dtype = self.to_str_dtype(compute_dtype)

        # Cast position_ids to int32 for WebGPU
        input_tensor = "position_ids"
        if is_webgpu:
            cast_input_name = f"{basename}/Cast_input"
            self.make_cast(
                cast_input_name, input_tensor, dtype=ir.DataType.INT32, shape=["batch_size", "sequence_length"]
            )
            input_tensor = f"{cast_input_name}/output_0"

        reduce_max_name = f"{basename}/ReduceMax"
        reduce_max_inputs = [input_tensor]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=compute_dtype, shape=[1])
        greater_or_equal_name = f"{basename}/GreaterOrEqual"
        greater_or_equal_inputs = [
            f"{reduce_max_name}/output_0",
            f"/model/constants/{compute_str_dtype}/{self.original_context_length}",
        ]
        self.make_greater_or_equal(greater_or_equal_name, greater_or_equal_inputs, shape=[])
        cast_name = f"{basename}/Cast"
        self.make_cast(cast_name, f"{greater_or_equal_name}/output_0", dtype=compute_dtype, shape=None)
        mul_name = f"{basename}/Mul"
        mul_inputs = [f"{cast_name}/output_0", f"/model/constants/{compute_str_dtype}/{self.original_context_length}"]
        self.make_mul(mul_name, mul_inputs, dtype=compute_dtype, shape=None)
        add_1_name = f"{basename}/Add_1"
        add_1_inputs = [f"{mul_name}/output_0", input_tensor]
        self.make_add(add_1_name, add_1_inputs, dtype=compute_dtype, shape=["batch_size", "sequence_length"])

        # Cast back to int64 for WebGPU to maintain compatibility
        result_name = add_1_name
        if is_webgpu:
            cast_output_name = f"{basename}/Cast_output"
            self.make_cast(
                cast_output_name,
                f"{add_1_name}/output_0",
                dtype=ir.DataType.INT64,
                shape=["batch_size", "sequence_length"],
            )
            result_name = cast_output_name

        return result_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        if self.position_ids_name is not None:
            super().make_attention(layer_id, attention, root_input, position_ids=self.position_ids_name, **kwargs)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)


class Phi3SmallModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.embed_attrs["scale"] = config.mup_embedding_multiplier
        self.rope_attrs["t_dtype"] = torch.float32
        self.lm_head_attrs["scale"] = 1 / config.mup_width_multiplier

        self.calculate_block_mask()
        self.dense_attention_every_n_layers = config.dense_attention_every_n_layers
        if config.mup_use_scaling:
            self.attention_attrs["scale"] = config.mup_attn_multiplier / self.head_size

        self.clamp_limit = config.gegelu_limit

    def calculate_cdiv(self, a, b):
        return -(a // -b)

    def calculate_block_mask(self):
        # Initialize parameters for calculating block dense mask
        n_heads = self.num_attn_heads
        q_len = self.context_length
        N_CTX = self.context_length
        BLOCK = self.attention_attrs["block_sparse"]["sparse_block_size"]
        local_blocks = self.attention_attrs["block_sparse"]["local_blocks"]
        vert_stride = self.attention_attrs["block_sparse"]["vert_stride"]
        homo_head = self.attention_attrs["block_sparse"]["homo_head"]

        N_BLOCK = self.calculate_cdiv(N_CTX, BLOCK)
        if homo_head:
            q_pos = torch.arange(N_BLOCK)[:, None]
            k_pos = torch.arange(N_BLOCK)[None]
            mask_vert_strided = (torch.arange(N_BLOCK) + 1) % vert_stride == 0
            block_mask_dense = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[-N_BLOCK_Q:].to_sparse_csr()

            crows = block_mask_dense_output.crow_indices()
            cols = block_mask_dense_output.col_indices()

            crows = crows[None].expand(n_heads, crows.shape[0])
            cols = cols[None].expand(n_heads, cols.shape[0])
        else:
            q_pos = torch.arange(N_BLOCK)[None, :, None]
            k_pos = torch.arange(N_BLOCK)[None, None]
            head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
            mask_vert_strided = [
                (torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(n_heads)
            ]
            mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
            block_mask_dense = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[:, -N_BLOCK_Q:]

            # Dense to crow_col
            pad = -1
            dim = block_mask_dense_output.dim()
            assert dim in (2, 3)
            if dim == 2:
                block_mask_dense_output = block_mask_dense_output[None]
            block_mask_dense_output = [xi.to_sparse_csr() for xi in block_mask_dense_output]
            crows = torch.vstack([xi.crow_indices() for xi in block_mask_dense_output])
            cols = [xi.col_indices() for xi in block_mask_dense_output]
            max_cols = max(len(xi) for xi in cols)
            cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
            cols = torch.vstack(cols)
            if dim == 2:
                crows = crows[0]
                cols = cols[0]

        # Create tensors for row indices and col indices
        crows_name = "block_row_indices"
        self.make_initializer(crows, crows_name, to=ir.DataType.INT32)
        self.mask_attrs["block_row_indices"] = crows_name

        cols_name = "block_col_indices"
        self.make_initializer(cols, cols_name, to=ir.DataType.INT32)
        self.mask_attrs["block_col_indices"] = cols_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        dense_attention_op = self.attention_attrs["op_type"]
        sparse_attention_op = "SparseAttention"

        # Use dense attention every n layers and use sparse attention otherwise
        if (self.layer_id + 1) % self.dense_attention_every_n_layers != 0:
            # Use sparse attention
            self.attention_attrs["op_type"] = sparse_attention_op

        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        qkv_weight = attention.query_key_value.weight.T.view(
            self.hidden_size, self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size
        )
        qkv_bias = attention.query_key_value.bias.view(
            self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size
        )

        attention.q_proj = torch.nn.Linear(in_features=q_size, out_features=q_size)
        attention.q_proj.weight = torch.nn.Parameter(
            qkv_weight[:, :, :-2].reshape(q_size, q_size).T, requires_grad=False
        )
        attention.q_proj.bias = (
            None
            if attention.query_key_value.bias is None
            else torch.nn.Parameter(qkv_bias[:, :-2].flatten(), requires_grad=False)
        )

        attention.k_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.k_proj.weight = torch.nn.Parameter(
            qkv_weight[:, :, [-2]].reshape(q_size, kv_size).T, requires_grad=False
        )
        attention.k_proj.bias = (
            None
            if attention.query_key_value.bias is None
            else torch.nn.Parameter(qkv_bias[:, [-2]].flatten(), requires_grad=False)
        )

        attention.v_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.v_proj.weight = torch.nn.Parameter(
            qkv_weight[:, :, [-1]].reshape(q_size, kv_size).T, requires_grad=False
        )
        attention.v_proj.bias = (
            None
            if attention.query_key_value.bias is None
            else torch.nn.Parameter(qkv_bias[:, [-1]].flatten(), requires_grad=False)
        )

        del qkv_weight
        del qkv_bias
        del attention.query_key_value

        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.attention_attrs["op_type"] = dense_attention_op

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #           root_input
        #               |
        #          UpProjMatMul
        #               |
        #           UpProjAdd
        #          /          \
        #         /            \
        #        /              \
        #      Slice             Slice
        #    (even idx)        (odd idx)
        #    /   |   \         /   |   \
        #  Cast  |    |      Cast  |    |
        #   |    |    |       |    |    |
        # IsInf  |   Clip   IsInf  |   Clip
        #   |    |    |       |    |    |
        #    \   |   /         \   |   /
        #     \  |  /           \  |  /
        #      Where             Where
        #        |                 |
        #    QuickGelu            Add
        #        |                 |
        #        +--------+--------+
        #                 |
        #                Mul
        #                 |
        #           DownProjMatMul
        #                 |
        #            DownProjAdd

        # Make input MatMul and Add nodes
        up_matmul_name = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        self.make_matmul(mlp.up_proj, up_matmul_name, root_input)
        up_add_name = f"/model/layers.{layer_id}/mlp/up_proj/Add"
        self.make_add_bias(mlp.up_proj.bias, up_add_name, f"{up_matmul_name}/output_0")

        # Left path
        slice_1_name = f"/model/layers.{layer_id}/mlp/gelu/Slice"
        slice_1_inputs = [
            f"{up_add_name}/output_0",
            "/model/constants/INT64/[0]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[-1]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(
            slice_1_name,
            slice_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        cast_1_name = f"/model/layers.{layer_id}/mlp/gelu/Cast"
        self.make_cast(
            cast_1_name,
            f"{slice_1_name}/output_0",
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        isinf_1_name = f"/model/layers.{layer_id}/mlp/gelu/IsInf"
        self.make_isinf(
            isinf_1_name, f"{cast_1_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size]
        )
        clip_1_name = f"/model/layers.{layer_id}/mlp/gelu/Clip"
        clip_1_inputs = [
            f"{slice_1_name}/output_0",
            "",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}",
        ]
        self.make_clip(
            clip_1_name, clip_1_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size]
        )
        where_1_name = f"/model/layers.{layer_id}/mlp/gelu/Where"
        where_1_inputs = [f"{isinf_1_name}/output_0", f"{slice_1_name}/output_0", f"{clip_1_name}/output_0"]
        self.make_where(
            where_1_name,
            where_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        # Make activation
        act_fn_name = self.make_activation(layer_id, root_input=f"{where_1_name}/output_0")

        # Right path
        slice_2_name = f"/model/layers.{layer_id}/mlp/linear/Slice"
        slice_2_inputs = [
            f"{up_add_name}/output_0",
            "/model/constants/INT64/[1]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[-1]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(
            slice_2_name,
            slice_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        cast_2_name = f"/model/layers.{layer_id}/mlp/linear/Cast"
        self.make_cast(
            cast_2_name,
            f"{slice_2_name}/output_0",
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        isinf_2_name = f"/model/layers.{layer_id}/mlp/linear/IsInf"
        self.make_isinf(
            isinf_2_name, f"{cast_2_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size]
        )
        clip_2_name = f"/model/layers.{layer_id}/mlp/linear/Clip"
        clip_2_inputs = [
            f"{slice_2_name}/output_0",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/-{self.clamp_limit}",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}",
        ]
        self.make_clip(
            clip_2_name, clip_2_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size]
        )
        where_2_name = f"/model/layers.{layer_id}/mlp/linear/Where"
        where_2_inputs = [f"{isinf_2_name}/output_0", f"{slice_2_name}/output_0", f"{clip_2_name}/output_0"]
        self.make_where(
            where_2_name,
            where_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        add_name = f"/model/layers.{layer_id}/mlp/linear/Add"
        add_inputs = [f"{where_2_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1"]
        self.make_add(
            add_name, add_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size]
        )

        # Make Mul node after activation
        mul_name = f"/model/layers.{layer_id}/mlp/Mul"
        mul_inputs = [f"{act_fn_name}/output_0", f"{add_name}/output_0"]
        self.make_mul(
            mul_name, mul_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size]
        )

        # Make output MatMul and Add nodes
        down_matmul_name = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        self.make_matmul(mlp.down_proj, down_matmul_name, f"{mul_name}/output_0")
        down_add_name = f"/model/layers.{layer_id}/mlp/down_proj/Add"
        self.make_add_bias(mlp.down_proj.bias, down_add_name, f"{down_matmul_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_add_name}/output_0"


class Phi3SmallLongRoPEModel(Phi3SmallModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()


class Phi3VModel(Phi3MiniLongRoPEModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MoELongRoPEModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        assert io_dtype == ir.DataType.FLOAT16, "This model only supports float16 io type."
        self.layernorm_attrs["simple"] = False
        self.moe_attrs["use_sparse_mixer"] = True
        self.make_rotary_embedding_multi_cache()

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MoE
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_block_sparse_moe(layer_id, layer.block_sparse_moe, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True


class Phi4MMModel(Phi3VModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.matmul_attrs["use_lora"] = True
        self.attention_attrs["use_packed_matmul"] = False

    def make_layer(self, layer_id, layer):
        layer.self_attn.qkv_proj.lora_A.default = layer.self_attn.qkv_proj.lora_A.vision
        layer.self_attn.qkv_proj.lora_B.default = layer.self_attn.qkv_proj.lora_B.vision
        layer.self_attn.qkv_proj.scaling["default"] = layer.self_attn.qkv_proj.scaling["vision"]
        layer.self_attn.o_proj.lora_A.default = layer.self_attn.o_proj.lora_A.vision
        layer.self_attn.o_proj.lora_B.default = layer.self_attn.o_proj.lora_B.vision
        layer.self_attn.o_proj.scaling["default"] = layer.self_attn.o_proj.scaling["vision"]

        layer.mlp.gate_up_proj.lora_A.default = layer.mlp.gate_up_proj.lora_A.vision
        layer.mlp.gate_up_proj.lora_B.default = layer.mlp.gate_up_proj.lora_B.vision
        layer.mlp.gate_up_proj.scaling["default"] = layer.mlp.gate_up_proj.scaling["vision"]
        layer.mlp.down_proj.lora_A.default = layer.mlp.down_proj.lora_A.vision
        layer.mlp.down_proj.lora_B.default = layer.mlp.down_proj.lora_B.vision
        layer.mlp.down_proj.scaling["default"] = layer.mlp.down_proj.scaling["vision"]

        super().make_layer(layer_id, layer)
