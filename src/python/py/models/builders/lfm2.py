# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# --------------------------------------------------------------------------
import onnx_ir as ir

from .base import Model


class LFM2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config.hidden_act = "silu"
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # LFM2-specific attributes
        self.layernorm_attrs["epsilon"] = config.norm_eps

        self.make_intermediate_size_init(config)

        self.conv_L_cache = config.conv_L_cache

    def make_intermediate_size_init(self, config):
        # Calculate the dynamic intermediate_size for the MLP.
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of
                )
        self.intermediate_size = intermediate_size

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)

    def make_inputs_and_outputs(self):
        conv_cache_shape = ["batch_size", self.hidden_size, self.conv_L_cache - 1]
        self.input_shapes["past.conv"] = conv_cache_shape
        self.output_shapes["present.conv"] = conv_cache_shape
        super().make_inputs_and_outputs()

    def make_past_key_subgraph(self, basename):
        # Find the first attention layer index (may not be layer 0)
        layer_index = self.layer_types.index("full_attention")
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, f"past_key_values.{layer_index}.key", shape=[4])
        gather_name = f"{basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/INT64/2"]
        self.make_gather(gather_name, gather_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        return gather_name

    def make_short_conv(self, layer_id, conv_module, root_input):
        basename = f"/model/layers.{layer_id}/conv"

        # 1. Input projection: project input to 3 * hidden_size
        in_proj_name = f"{basename}/in_proj/MatMul"
        in_proj_name = self.make_matmul(conv_module.in_proj, in_proj_name, root_input)

        # Transpose from (B, S, 3*H) to (B, 3*H, S)
        transpose_1_name = f"{basename}/Transpose_1"
        self.make_transpose(
            transpose_1_name, f"{in_proj_name}/output_0", self.io_dtype,
            shape=["batch_size", 3 * self.hidden_size, "sequence_length"], perm=[0, 2, 1],
        )

        # Split into 3 equal parts along dim 1: b, c, x
        split_tensor_name = f"/model/constants/INT64/{[self.hidden_size, self.hidden_size, self.hidden_size]}"
        split_name = f"{basename}/Split"
        b_out = f"{split_name}/output_0"
        c_out = f"{split_name}/output_1"
        x_out = f"{split_name}/output_2"
        split_shape = ["batch_size", self.hidden_size, "sequence_length"]
        self.make_split(
            split_name,
            inputs=[f"{transpose_1_name}/output_0", split_tensor_name],
            outputs=[b_out, c_out, x_out],
            dtypes=[self.io_dtype] * 3,
            shapes=[split_shape] * 3,
            axis=1,
        )

        # Element-wise multiply: bx = b * x
        mul_1_name = f"{basename}/Mul_1"
        self.make_mul(mul_1_name, [b_out, x_out], self.io_dtype, shape=["batch_size", self.hidden_size, "sequence_length"])

        # 2. Stateful depthwise convolution
        conv_weight_name = f"model.layers.{layer_id}.conv.conv.weight"
        self.make_initializer(conv_module.conv.weight, conv_weight_name, to=self.io_dtype)

        conv_bias_name = ""
        if conv_module.conv.bias is not None:
            conv_bias_name = f"model.layers.{layer_id}.conv.conv.bias"
            self.make_initializer(conv_module.conv.bias, conv_bias_name, to=self.io_dtype)

        conv_op_name = f"{basename}/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op_name,
            root_input=f"{mul_1_name}/output_0",
            weight=conv_weight_name,
            bias=conv_bias_name,
            past_conv_state=self.input_names["past.conv"][layer_id],
            present_conv_state=self.output_names["present.conv"][layer_id],
            activation="none",
            channels=self.hidden_size,
        )

        # Element-wise multiply: result = c * conv_out
        mul_2_name = f"{basename}/Mul_2"
        self.make_mul(
            mul_2_name,
            [c_out, f"{conv_op_name}/output_0"],
            self.io_dtype,
            shape=["batch_size", self.hidden_size, "sequence_length"],
        )

        # 3. Output processing: transpose back and project
        transpose_2_name = f"{basename}/Transpose_2"
        self.make_transpose(
            transpose_2_name, f"{mul_2_name}/output_0", self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size], perm=[0, 2, 1],
        )

        out_proj_name = f"{basename}/out_proj/MatMul"
        out_proj_name = self.make_matmul(conv_module.out_proj, out_proj_name, f"{transpose_2_name}/output_0")
        return f"{out_proj_name}/output_0"

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Alias attribute names for compatibility with the base class
        attention.o_proj = attention.out_proj
        attention.q_norm = attention.q_layernorm
        attention.k_norm = attention.k_layernorm
        super().make_attention(layer_id, attention, root_input, **kwargs)

    def make_layer(self, layer_id, layer):
        # Each LFM2 decoder layer is defined as:
        # operator_norm --> attention/conv --> ffn_norm --> MLP
        # with SkipLayerNorm fusing the residual Add + LayerNorm.
        self.make_layernorm(
            layer_id,
            layer.operator_norm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="operator",
        )

        # Operator block: Attention or Conv depending on layer type
        if self.layer_types[layer_id] == "full_attention":
            self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        else:  # 'conv'
            conv_output = self.make_short_conv(layer_id, layer.conv, self.layernorm_attrs["output_0"])
            self.layernorm_attrs["skip_input"] = conv_output

        self.make_layernorm(
            layer_id,
            layer.ffn_norm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="ffn",
        )

        self.make_feed_forward(layer_id, layer, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def get_feed_forward_module(self, layer):
        # Hugging Face names the module `feed_forward`; the quantized-checkpoint IR exposes it as `mlp`.
        feed_forward = getattr(layer, "feed_forward", None)
        return layer.mlp if feed_forward is None else feed_forward

    def make_feed_forward(self, layer_id, layer, root_input):
        mlp = self.get_feed_forward_module(layer)
        if hasattr(mlp, "w1"):
            # Alias Hugging Face's MLP attribute names for compatibility with the base class
            mlp.gate_proj = mlp.w1
            mlp.up_proj = mlp.w3
            mlp.down_proj = mlp.w2
        self.make_mlp(layer_id, mlp, root_input=root_input)

    def update_genai_config(self, genai_config):
        decoder = genai_config["model"]["decoder"]
        decoder["layer_types"] = self.layer_types
        decoder["conv_cache_size"] = self.conv_L_cache - 1


class LFM2MoEModel(LFM2Model):
    """LFM2-MoE builder (LFM2-8B-A1B, LFM2.5-8B-A1B, LFM2-24B-A2B).

    Same hybrid conv/attention stack as LFM2. The first ``num_dense_layers`` layers keep the dense
    SwiGLU MLP; every later layer routes each token to ``num_experts_per_tok`` of ``num_experts``
    experts. The router (``Lfm2MoeSparseMoeBlock.route_tokens_to_experts`` in transformers) is
    sigmoid based with an auxiliary-loss-free load-balancing bias: experts are *selected* by the
    top-k of ``sigmoid(logits) + expert_bias`` but *mixed* with
    ``sigmoid_i / (sum_selected(sigmoid_j) + 1e-6)``.

    The fused MoE/QMoE op only knows softmax-over-top-k routing, so the selection is done in the
    graph and the op is fed ``log(sigmoid(logits))`` at the selected experts and a large negative
    sentinel everywhere else. Its softmax over the surviving top-k entries then equals
    ``sigmoid_i / sum_selected(sigmoid_j)``. The remaining per-token factor
    ``sum_selected / (sum_selected + 1e-6)`` is applied to the op output, together with
    ``routed_scaling_factor``, so tokens whose selected scores are all tiny (or flushed to zero)
    keep the HF output magnitude.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        if not config.norm_topk_prob:
            raise NotImplementedError(
                "LFM2-MoE export requires norm_topk_prob=True: the fused MoE op always normalizes the "
                "routing weights over the selected experts."
            )

        self.moe_intermediate_size = config.moe_intermediate_size
        self.moe_attrs["num_dense_layers"] = config.num_dense_layers
        self.moe_attrs["use_expert_bias"] = config.use_expert_bias
        self.moe_attrs["routed_scaling_factor"] = config.routed_scaling_factor

        # Router score for the experts that were not selected. Any finite value far below every
        # plausible log-sigmoid works: it must fit in fp16 and exp(sentinel - max) must underflow to 0.
        self.moe_attrs["router_sentinel"] = -10000.0
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True

    def make_intermediate_size_init(self, config):
        # Lfm2MoeConfig has no block_auto_adjust_ff_dim: the dense layers use intermediate_size as is.
        self.intermediate_size = config.intermediate_size

    def make_feed_forward(self, layer_id, layer, root_input):
        if layer_id < self.moe_attrs["num_dense_layers"]:
            super().make_feed_forward(layer_id, layer, root_input)
        else:
            self.make_moe(layer_id, self.get_feed_forward_module(layer), root_input)

    def make_moe(self, layer_id, moe, root_input):
        self.make_moe_preprocessing(layer_id, moe, root_input)
        router_probs, output_scale = self.make_moe_router(layer_id, moe, root_input)
        self.make_moe_subgraph(layer_id, moe, root_input, router_probs, output_scale)

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        # Keep the router in floating point so int4 rounding cannot flip an expert choice.
        moe.gate.exclude_from_quantization = True
        self.make_interleaved_swiglu_moe_preprocessing(layer_id, moe)

    def make_moe_router(self, layer_id, moe, root_input):
        """Emit the in-graph expert selection.

        Returns ``(router_probs, output_scale)``: the masked router scores fed to the MoE/QMoE op and
        the per-token factor that the op output is multiplied by.

        root_input --> MatMul --> Reshape --> Cast(fp32) --> sigmoid --+--> Add(expert_bias) --> TopK
                                                                       |                          |
                                                                       +--> GatherElements <------+
                                                                       |     |        |           |
                                                                       |     |   ReduceSum --> ... --> output_scale
                                                                       |     |
                                                                       |     Clip --> Log
                                                                       |               |
                                                                       +--> Shape --> ConstantOfShape(sentinel)
                                                                                            |
                                                     Cast(io_dtype) <-- ScatterElements <---+
        """
        scores_name = self.make_moe_router_scores(layer_id, moe, root_input)
        indices_name = self.make_moe_router_selection(layer_id, moe, scores_name)
        selected_name = self.make_moe_router_selected_scores(layer_id, scores_name, indices_name)
        output_scale = self.make_moe_router_output_scale(layer_id, root_input, selected_name)
        router_probs = self.make_moe_router_mask(layer_id, scores_name, indices_name, selected_name)
        return router_probs, output_scale

    def make_moe_router_scores(self, layer_id, moe, root_input):
        """Emit the per-expert router scores (`sigmoid(logits)`) in fp32 and return their name."""
        basename = f"/model/layers.{layer_id}/moe/router"
        logits_shape = self.make_moe_router_shape()

        matmul_name = self.make_matmul(moe.gate, f"{basename}/MatMul", root_input)
        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name,
            [f"{matmul_name}/output_0", f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}"],
            dtype=self.io_dtype,
            shape=logits_shape,
        )
        logits_name = f"{reshape_name}/output_0"
        if self.io_dtype != ir.DataType.FLOAT:
            # The scores drive a discrete choice and are logged below, so keep them in fp32.
            cast_name = f"{basename}/Cast"
            self.make_cast(cast_name, logits_name, ir.DataType.FLOAT, shape=logits_shape)
            logits_name = f"{cast_name}/output_0"

        # sigmoid(x) = 1 / (1 + exp(-x)), spelled out. ONNX Runtime's CPU Sigmoid kernel carries an absolute
        # error of about 6e-8 (it is evaluated as 1 - sigmoid(-x)), so it returns 5.96e-8 for x = -16
        # instead of 1.125e-7 and 0 below x = -17. The scores are logged and summed below, where that
        # is a large relative error; this form matches torch.sigmoid to fp32 rounding.
        sigmoid_basename = f"{basename}/sigmoid"
        self.make_neg(f"{sigmoid_basename}/Neg", logits_name, ir.DataType.FLOAT, shape=logits_shape)
        self.make_exp(
            f"{sigmoid_basename}/Exp", f"{sigmoid_basename}/Neg/output_0", ir.DataType.FLOAT, shape=logits_shape
        )
        self.make_add(
            f"{sigmoid_basename}/Add",
            [f"{sigmoid_basename}/Exp/output_0", f"/model/constants/FLOAT/{1.0}"],
            dtype=ir.DataType.FLOAT,
            shape=logits_shape,
        )
        reciprocal_name = f"{sigmoid_basename}/Reciprocal"
        self.make_reciprocal(reciprocal_name, f"{sigmoid_basename}/Add/output_0", ir.DataType.FLOAT, shape=logits_shape)
        return f"{reciprocal_name}/output_0"

    def make_moe_router_selection(self, layer_id, moe, scores_name):
        """Emit the top-k over the bias-corrected scores and return the selected expert indices."""
        basename = f"/model/layers.{layer_id}/moe/router"
        top_k = self.moe_attrs["top_k"]
        logits_shape = self.make_moe_router_shape()

        selection_name = scores_name
        if self.moe_attrs["use_expert_bias"]:
            expert_bias_name = f"model.layers.{layer_id}.moe.expert_bias"
            self.make_initializer(moe.expert_bias, expert_bias_name, to=ir.DataType.FLOAT)
            add_name = f"{basename}/Add"
            self.make_add(add_name, [scores_name, expert_bias_name], dtype=ir.DataType.FLOAT, shape=logits_shape)
            selection_name = f"{add_name}/output_0"

        topk_name = f"{basename}/TopK"
        self.make_topk(
            topk_name,
            [selection_name, f"/model/constants/INT64/[{top_k}]"],
            dtype=ir.DataType.FLOAT,
            shape=self.make_moe_router_shape(last_dim=top_k),
        )
        return f"{topk_name}/output_1"

    def make_moe_router_selected_scores(self, layer_id, scores_name, indices_name):
        """Emit the unbiased scores of the selected experts (`[rows, top_k]`) and return their name."""
        gather_name = f"/model/layers.{layer_id}/moe/router/GatherElements"
        self.make_gather_elements(
            gather_name,
            [scores_name, indices_name],
            dtype=ir.DataType.FLOAT,
            shape=self.make_moe_router_shape(last_dim=self.moe_attrs["top_k"]),
            axis=1,
        )
        return f"{gather_name}/output_0"

    def make_moe_router_output_scale(self, layer_id, root_input, selected_name):
        """Emit the per-token factor applied to the MoE op output and return its name.

        HF mixes the selected experts with `sigmoid_i / (sum_selected + 1e-6)` while the op normalizes
        to `sigmoid_i / sum_selected`, so the op output is scaled by `sum_selected / (sum_selected + 1e-6)`.
        The sum is taken before the clamp in `make_moe_router_mask`, so scores that flushed to zero give
        a zero factor, matching HF exactly. `routed_scaling_factor` is folded in here as well.
        """
        basename = f"/model/layers.{layer_id}/moe/router/scale"
        rows_shape = self.make_moe_router_shape(last_dim=1)

        sum_name = f"{basename}/ReduceSum"
        self.make_reduce_sum(
            sum_name,
            [selected_name, "/model/constants/INT64/[-1]"],
            dtype=ir.DataType.FLOAT,
            shape=rows_shape,
            keepdims=True,
        )
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [f"{sum_name}/output_0", f"/model/constants/FLOAT/{1e-6}"],
            dtype=ir.DataType.FLOAT,
            shape=rows_shape,
        )
        div_name = f"{basename}/Div"
        self.make_div(
            div_name, [f"{sum_name}/output_0", f"{add_name}/output_0"], dtype=ir.DataType.FLOAT, shape=rows_shape
        )
        scale_name = f"{div_name}/output_0"
        routed_scaling_factor = self.moe_attrs["routed_scaling_factor"]
        if routed_scaling_factor != 1.0:
            mul_name = f"{basename}/Mul"
            self.make_mul(
                mul_name,
                [scale_name, f"/model/constants/FLOAT/{routed_scaling_factor}"],
                dtype=ir.DataType.FLOAT,
                shape=rows_shape,
            )
            scale_name = f"{mul_name}/output_0"

        # Reshape [rows, 1] to the MoE output's leading dims + [1] so it broadcasts over hidden_size.
        output_shape = self.make_hidden_state_shape(last_dim=1)
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, root_input, shape=[len(output_shape)])
        slice_name = f"{basename}/Slice"
        self.make_slice(
            slice_name,
            [f"{shape_name}/output_0", "/model/constants/INT64/[0]", "/model/constants/INT64/[-1]"],
            dtype=ir.DataType.INT64,
            shape=[len(output_shape) - 1],
        )
        concat_name = f"{basename}/Concat"
        self.make_concat(
            concat_name,
            [f"{slice_name}/output_0", "/model/constants/INT64/[1]"],
            dtype=ir.DataType.INT64,
            shape=[len(output_shape)],
            axis=0,
        )
        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name, [scale_name, f"{concat_name}/output_0"], dtype=ir.DataType.FLOAT, shape=output_shape
        )
        scale_name = f"{reshape_name}/output_0"

        if self.io_dtype != ir.DataType.FLOAT:
            cast_name = f"{basename}/Cast"
            self.make_cast(cast_name, scale_name, self.io_dtype, shape=output_shape)
            scale_name = f"{cast_name}/output_0"
        return scale_name

    def make_moe_router_mask(self, layer_id, scores_name, indices_name, selected_name):
        """Emit `log(selected scores)` at the selected experts and a sentinel elsewhere, in `io_dtype`.

        The MoE/QMoE op takes the softmax of the top-k of this tensor, which equals
        `sigmoid_i / sum_selected(sigmoid_j)`; `make_moe_router_output_scale` supplies the rest of HF's
        `sigmoid_i / (sum_selected + 1e-6)` mixing weight.
        """
        basename = f"/model/layers.{layer_id}/moe/router"
        logits_shape = self.make_moe_router_shape()
        selected_shape = self.make_moe_router_shape(last_dim=self.moe_attrs["top_k"])

        # A sigmoid that flushed to 0 would give log = -inf, below the sentinel, and let the op's own
        # top-k pick a different expert than the graph selected. Clamp so log stays far above it.
        clip_name = f"{basename}/Clip"
        self.make_clip(
            clip_name,
            [selected_name, f"/model/constants/FLOAT/{1e-30}", ""],
            dtype=ir.DataType.FLOAT,
            shape=selected_shape,
        )
        log_name = f"{basename}/Log"
        self.make_log(log_name, f"{clip_name}/output_0", ir.DataType.FLOAT, shape=selected_shape)

        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, scores_name, shape=[2])
        sentinel_name = f"{basename}/ConstantOfShape"
        self.make_constant_of_shape(
            sentinel_name,
            f"{shape_name}/output_0",
            value=ir.tensor([self.moe_attrs["router_sentinel"]], dtype=ir.DataType.FLOAT),
            dtype=ir.DataType.FLOAT,
            shape=logits_shape,
        )
        scatter_name = f"{basename}/ScatterElements"
        self.make_scatter_elements(
            scatter_name,
            [f"{sentinel_name}/output_0", indices_name, f"{log_name}/output_0"],
            dtype=ir.DataType.FLOAT,
            shape=logits_shape,
            axis=1,
        )
        router_probs_name = f"{scatter_name}/output_0"

        if self.io_dtype != ir.DataType.FLOAT:
            cast_name = f"{basename}/Cast_1"
            self.make_cast(cast_name, router_probs_name, self.io_dtype, shape=logits_shape)
            router_probs_name = f"{cast_name}/output_0"
        return router_probs_name

    def make_moe_subgraph(self, layer_id, moe, root_input, router_probs=None, output_scale=None):
        # `make_moe` always passes the pair returned by `make_moe_router`; the defaults only keep the
        # base-class signature.
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        names = self.make_moe_expert_names(layer_id)
        gate_up_proj_zero_points, down_proj_zero_points = self.moe_attrs.get("zero_point_names", {}).get(
            layer_id, ("", "")
        )
        use_zero_points = bool(gate_up_proj_zero_points and down_proj_zero_points) and self.ep != "trt-rtx"
        gate_up_proj_global_scales, down_proj_global_scales = self.moe_attrs.get("global_scale_names", {}).get(
            layer_id, ("", "")
        )

        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=router_probs,
            weight1=names["gate_up_weight"],
            scales1=names["gate_up_scales"] if op_type == "QMoE" else "",
            bias1=names["gate_up_bias"],
            weight2=names["down_weight"],
            scales2=names["down_scales"] if op_type == "QMoE" else "",
            bias2=names["down_bias"],
            zero_points1=gate_up_proj_zero_points if use_zero_points else "",
            zero_points2=down_proj_zero_points if use_zero_points else "",
            global_scales1=gate_up_proj_global_scales,
            global_scales2=down_proj_global_scales,
        )

        # Apply the routing-mass correction (and routed_scaling_factor) from the router.
        mul_name = f"{basename}/Mul"
        self.make_mul(
            mul_name, [f"{moe_name}/output_0", output_scale], dtype=self.io_dtype, shape=self.make_hidden_state_shape()
        )

        # Assign the MoE output as the residual input of the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{mul_name}/output_0"
