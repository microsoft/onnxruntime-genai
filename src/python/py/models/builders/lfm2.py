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

        self.conv_L_cache = config.conv_L_cache

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

        # Alias MLP attribute names for compatibility with the base class
        layer.mlp = layer.feed_forward
        layer.mlp.gate_proj = layer.mlp.w1
        layer.mlp.up_proj = layer.mlp.w3
        layer.mlp.down_proj = layer.mlp.w2
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def update_genai_config(self, genai_config):
        decoder = genai_config["model"]["decoder"]
        decoder["layer_types"] = self.layer_types
        decoder["conv_cache_size"] = self.conv_L_cache - 1
