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
    top-k of ``sigmoid(logits) + expert_bias`` but *mixed* with the unbiased ``sigmoid(logits)``,
    renormalized over the selected experts.

    The fused MoE/QMoE op only knows softmax-over-top-k routing, so the selection is done in the
    graph and the op is fed ``log(sigmoid(logits))`` at the selected experts and a large negative
    sentinel everywhere else. Its softmax over the surviving top-k entries then equals
    ``sigmoid_i / sum_selected(sigmoid_j)``, which is the HF routing weight.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        if not config.norm_topk_prob:
            raise NotImplementedError(
                "LFM2-MoE export requires norm_topk_prob=True: the fused MoE op always normalizes the "
                "routing weights over the selected experts."
            )

        self.num_dense_layers = config.num_dense_layers
        self.moe_intermediate_size = config.moe_intermediate_size
        self.use_expert_bias = config.use_expert_bias
        self.routed_scaling_factor = config.routed_scaling_factor

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
        if layer_id < self.num_dense_layers:
            super().make_feed_forward(layer_id, layer, root_input)
        else:
            self.make_moe(layer_id, self.get_feed_forward_module(layer), root_input)

    def make_moe(self, layer_id, moe, root_input):
        self.make_moe_preprocessing(layer_id, moe, root_input)
        router_probs = self.make_moe_router(layer_id, moe, root_input)
        self.make_moe_subgraph(layer_id, moe, root_input, router_probs)

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        # Keep the router in floating point so int4 rounding cannot flip an expert choice.
        moe.gate.exclude_from_quantization = True
        self.make_interleaved_swiglu_moe_preprocessing(layer_id, moe)

    def make_moe_router(self, layer_id, moe, root_input):
        """Emit the in-graph expert selection and return the name of the masked router scores.

        root_input --> MatMul --> Reshape --> Cast(fp32) --> Sigmoid --+--> Add(expert_bias) --> TopK
                                                                       |                          |
                                                                       +--> GatherElements <------+
                                                                       |          |               |
                                                                       |     Clip --> Log         |
                                                                       |               |          |
                                                                       +--> Shape --> ConstantOfShape(sentinel)
                                                                                            |
                                                     Cast(io_dtype) <-- ScatterElements <---+
        """
        scores_name = self.make_moe_router_scores(layer_id, moe, root_input)
        indices_name = self.make_moe_router_selection(layer_id, moe, scores_name)
        return self.make_moe_router_mask(layer_id, scores_name, indices_name)

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

        sigmoid_name = f"{basename}/Sigmoid"
        self.make_sigmoid(sigmoid_name, logits_name, ir.DataType.FLOAT, shape=logits_shape)
        return f"{sigmoid_name}/output_0"

    def make_moe_router_selection(self, layer_id, moe, scores_name):
        """Emit the top-k over the bias-corrected scores and return the selected expert indices."""
        basename = f"/model/layers.{layer_id}/moe/router"
        top_k = self.moe_attrs["top_k"]
        logits_shape = self.make_moe_router_shape()

        selection_name = scores_name
        if self.use_expert_bias:
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

    def make_moe_router_mask(self, layer_id, scores_name, indices_name):
        """Emit `log(scores)` at the selected experts and a sentinel elsewhere, in `io_dtype`.

        The MoE/QMoE op takes the softmax of the top-k of this tensor, which then equals the
        model's `sigmoid_i / sum_selected(sigmoid_j)` mixing weights. Hugging Face divides by
        `sum + 1e-6` instead; the op normalizes without that epsilon, a known parity gap that is
        far below the io_dtype rounding of the scores themselves.
        """
        basename = f"/model/layers.{layer_id}/moe/router"
        logits_shape = self.make_moe_router_shape()
        selected_shape = self.make_moe_router_shape(last_dim=self.moe_attrs["top_k"])

        # Gather the selected scores first so Log runs over top_k entries instead of num_experts.
        gather_name = f"{basename}/GatherElements"
        self.make_gather_elements(
            gather_name, [scores_name, indices_name], dtype=ir.DataType.FLOAT, shape=selected_shape, axis=1
        )
        # A sigmoid that flushed to 0 would give log = -inf, below the sentinel, and let the op's own
        # top-k pick a different expert than the graph selected. Clamp so log stays far above it.
        clip_name = f"{basename}/Clip"
        self.make_clip(
            clip_name,
            [f"{gather_name}/output_0", f"/model/constants/FLOAT/{1e-30}", ""],
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

    def make_moe_router_shape(self, last_dim=None):
        return ["batch_size * sequence_length", self.moe_attrs["num_experts"] if last_dim is None else last_dim]

    def make_moe_subgraph(self, layer_id, moe, root_input, router_probs=None):
        if router_probs is None:
            raise ValueError("LFM2-MoE needs the masked router scores returned by make_moe_router.")
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
        output_name = f"{moe_name}/output_0"

        if self.routed_scaling_factor != 1.0:
            mul_name = f"{basename}/Mul"
            self.make_mul(
                mul_name,
                [output_name, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.routed_scaling_factor}"],
                dtype=self.io_dtype,
                shape=self.make_hidden_state_shape(),
            )
            output_name = f"{mul_name}/output_0"

        # Assign the MoE output as the residual input of the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = output_name
