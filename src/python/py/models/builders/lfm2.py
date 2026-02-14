# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
import onnx_ir as ir

from .base import Model


class Lfm2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config.hidden_act = "silu"
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # LFM2-specific attributes
        self.layernorm_attrs["epsilon"] = config.norm_eps
        self.rope_attrs["theta"] = config.rope_parameters["rope_theta"]

        # Calculate the dynamic intermediate_size for the MLP.
        intermediate_size = self.intermediate_size
        if getattr(config, "block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * intermediate_size / 3)
            if getattr(config, "block_ffn_dim_multiplier", None) is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
            multiple_of = getattr(config, "block_multiple_of", 1)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.intermediate_size = intermediate_size

        # Hybrid attention/convolution architecture
        self.layer_types = config.layer_types
        self.conv_L_cache = config.conv_L_cache

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_inputs_and_outputs(self):
        # Add model-specific inputs to list of model inputs
        inputs = self.model.graph.inputs
        for name in self.input_names:
            dtype = self.input_types[name]
            shape = self.input_shapes[name]
            inputs.append(self.make_value(name, dtype=dtype, shape=shape))

        # Add model-specific outputs to list of model outputs
        outputs = self.model.graph.outputs
        for name in self.output_names:
            dtype = self.output_types[name]
            shape = self.output_shapes[name]
            outputs.append(self.make_value(name, dtype=dtype, shape=shape))

        # Add per-layer cache inputs and outputs
        for i in range(self.num_layers):
            if self.layer_types[i] == "full_attention":
                # Add KV cache inputs
                key_in_name = f"past_key_values.{i}.key"
                key_in_shape = self.make_key_value_cache_shape(i, self.input_shapes["past_key_values.key"])
                inputs.append(self.make_value(key_in_name, dtype=self.input_types["past_key_values.key"], shape=key_in_shape))

                value_in_name = f"past_key_values.{i}.value"
                value_in_shape = self.make_key_value_cache_shape(i, self.input_shapes["past_key_values.value"])
                inputs.append(self.make_value(value_in_name, dtype=self.input_types["past_key_values.value"], shape=value_in_shape))

                # Add KV cache outputs
                key_out_name = f"present.{i}.key"
                key_out_shape = self.make_key_value_cache_shape(i, self.output_shapes["present.key"])
                outputs.append(self.make_value(key_out_name, dtype=self.output_types["present.key"], shape=key_out_shape))

                value_out_name = f"present.{i}.value"
                value_out_shape = self.make_key_value_cache_shape(i, self.output_shapes["present.value"])
                outputs.append(self.make_value(value_out_name, dtype=self.output_types["present.value"], shape=value_out_shape))

            elif self.layer_types[i] == "conv":
                conv_cache_shape = ["batch_size", self.hidden_size, self.conv_L_cache]

                # Add conv cache input
                past_conv_name = f"past_conv.{i}"
                inputs.append(self.make_value(past_conv_name, dtype=self.io_dtype, shape=conv_cache_shape))

                # Add conv cache output
                present_conv_name = f"present_conv.{i}"
                outputs.append(self.make_value(present_conv_name, dtype=self.io_dtype, shape=conv_cache_shape))

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
        self.make_node(
            "Split",
            inputs=[f"{transpose_1_name}/output_0", split_tensor_name],
            outputs=[b_out, c_out, x_out],
            name=split_name,
            axis=1,
        )
        for out_val in [b_out, c_out, x_out]:
            self.make_value(out_val, self.io_dtype, shape=["batch_size", self.hidden_size, "sequence_length"])

        # Element-wise multiply: bx = b * x
        mul_1_name = f"{basename}/Mul_1"
        self.make_mul(mul_1_name, [b_out, x_out], self.io_dtype, shape=["batch_size", self.hidden_size, "sequence_length"])

        # 2. Stateful convolution: concatenate with past conv cache
        past_conv_name = f"past_conv.{layer_id}"
        conv_input_name = f"{basename}/Conv_Input"
        self.make_concat(
            conv_input_name, [past_conv_name, f"{mul_1_name}/output_0"], self.io_dtype,
            shape=["batch_size", self.hidden_size, "past_plus_current_seq_len"], axis=2,
        )
        conv_input = f"{conv_input_name}/output_0"

        # Depthwise convolution (group=hidden_size)
        conv_op_name = f"{basename}/Conv"
        conv_weight_name = f"model.layers.{layer_id}.conv.conv.weight"
        self.make_initializer(conv_module.conv.weight, conv_weight_name, to=self.io_dtype)
        conv_inputs = [conv_input, conv_weight_name]
        if conv_module.conv.bias is not None:
            conv_bias_name = f"model.layers.{layer_id}.conv.conv.bias"
            self.make_initializer(conv_module.conv.bias, conv_bias_name, to=self.io_dtype)
            conv_inputs.append(conv_bias_name)

        conv_out_full = f"{conv_op_name}/output_0"
        self.make_node(
            "Conv", inputs=conv_inputs, outputs=[conv_out_full], name=conv_op_name,
            kernel_shape=[self.conv_L_cache], pads=[0, 0], group=self.hidden_size,
        )

        # Slice the conv output to keep only the last seq_len elements (obtained from root_input shape)
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, root_input, shape=[3])
        seq_len_gather_name = f"{basename}/Gather_SeqLen"
        self.make_gather(
            seq_len_gather_name, [f"{shape_name}/output_0", "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64, shape=[], axis=0,
        )

        neg_seq_len_name = f"{basename}/Neg_Seq_Len"
        self.make_mul(
            neg_seq_len_name, [f"{seq_len_gather_name}/output_0", "/model/constants/INT64/-1"],
            ir.DataType.INT64, shape=[],
        )

        unsqueeze_starts_name = f"{basename}/Unsqueeze_starts"
        self.make_unsqueeze(
            unsqueeze_starts_name, [f"{neg_seq_len_name}/output_0", "/model/constants/INT64/[0]"],
            ir.DataType.INT64, shape=[1],
        )

        slice_name = f"{basename}/Slice_Conv_Output"
        conv_out = f"{slice_name}/output_0"
        self.make_node(
            "Slice",
            inputs=[
                conv_out_full,
                f"{unsqueeze_starts_name}/output_0",
                f"/model/constants/INT64/[{np.iinfo(np.int64).max}]",
                "/model/constants/INT64/[2]",
            ],
            outputs=[conv_out],
            name=slice_name,
        )
        self.make_value(conv_out, self.io_dtype, shape=["batch_size", self.hidden_size, "sequence_length"])

        # Element-wise multiply: result = c * conv_out
        mul_2_name = f"{basename}/Mul_2"
        self.make_mul(mul_2_name, [c_out, conv_out], self.io_dtype, shape=["batch_size", self.hidden_size, "sequence_length"])

        # 3. Cache update: slice the conv input to keep last conv_L_cache elements
        present_conv_name = f"present_conv.{layer_id}"
        slice_cache_name = f"{basename}/Slice_Cache"
        self.make_node(
            "Slice",
            inputs=[
                conv_input,
                f"/model/constants/INT64/[-{self.conv_L_cache}]",
                f"/model/constants/INT64/[{np.iinfo(np.int64).max}]",
                "/model/constants/INT64/[2]",
            ],
            outputs=[present_conv_name],
            name=slice_cache_name,
        )
        self.make_value(present_conv_name, self.io_dtype, shape=["batch_size", self.hidden_size, self.conv_L_cache])

        # 4. Output processing: transpose back and project
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
