# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir


class Standard:
    """Reusable expansions containing only standard ONNX operators."""

    def make_gated_add(self, name, root_input, scaled_input, gate, shape):
        # scaled_input * gate -> Add(root_input)
        mul_name = f"{name}/Mul"
        self.make_mul(mul_name, [scaled_input, gate], self.io_dtype, shape=shape)
        self.make_add(name, [root_input, f"{mul_name}/output_0"], self.io_dtype, shape=shape)

    def make_linear_attention_gate(self, name, a, dt_bias, decay_scale, b, shape):
        # a -> Cast(FP32) -> Add(dt_bias) -> Softplus -> Mul(decay_scale) -> Cast
        # b -> Sigmoid
        self.make_cast(f"{name}/a/Cast", a, ir.DataType.FLOAT, shape)
        self.make_add(f"{name}/Add", [f"{name}/a/Cast/output_0", dt_bias], ir.DataType.FLOAT, shape)
        self.make_softplus(f"{name}/Softplus", f"{name}/Add/output_0", ir.DataType.FLOAT, shape)
        self.make_mul(f"{name}/Mul", [f"{name}/Softplus/output_0", decay_scale], ir.DataType.FLOAT, shape)
        self.make_cast(name, f"{name}/Mul/output_0", self.io_dtype, shape)
        self.make_node("Sigmoid", [b], [f"{name}/output_1"], name=f"{name}/Sigmoid")
        self.make_value(f"{name}/output_1", self.io_dtype, shape)

    def make_gated_rms_norm(self, name, root_input, scale, gate, shape, epsilon=1e-5):
        # root_input -> Reshape(heads) -> RMSNorm -> Flatten -> Cast(FP32) --+
        # gate -> Cast(FP32) -> SiLU ------------------------------------+-> Mul -> Cast
        head_size = int(self.values[scale].shape[0])
        grouped_shape = [*shape[:-1], shape[-1] // head_size, head_size]
        # An inferred -1 is ambiguous when batch or sequence length is zero.
        reshape = self.make_expansion_constant(
            name=f"{name}/group_shape", value=[*([0] * (len(shape) - 1)), *grouped_shape[-2:]]
        )
        self.make_reshape(f"{name}/Reshape", [root_input, reshape], self.io_dtype, grouped_shape)
        normalized = f"{name}/SimplifiedLayerNormalization/output_0"
        self.make_simplified_layer_norm(
            f"{name}/SimplifiedLayerNormalization",
            f"{name}/Reshape/output_0",
            scale,
            normalized,
            self.io_dtype,
            grouped_shape,
            epsilon=epsilon,
        )
        restore = self.make_expansion_constant(f"{name}/flat_shape", [*([0] * (len(shape) - 1)), shape[-1]])
        self.make_reshape(f"{name}/Flatten", [normalized, restore], self.io_dtype, shape)
        self.make_cast(f"{name}/norm/Cast", f"{name}/Flatten/output_0", ir.DataType.FLOAT, shape)
        self.make_cast(f"{name}/gate/Cast", gate, ir.DataType.FLOAT, shape)
        gate = f"{name}/gate/Cast/output_0"
        self.make_sigmoid(f"{name}/Sigmoid", gate, ir.DataType.FLOAT, shape)
        self.make_mul(f"{name}/SiLU", [gate, f"{name}/Sigmoid/output_0"], ir.DataType.FLOAT, shape)
        self.make_mul(f"{name}/Mul", [f"{name}/norm/Cast/output_0", f"{name}/SiLU/output_0"], ir.DataType.FLOAT, shape)
        self.make_cast(name, f"{name}/Mul/output_0", self.io_dtype, shape)

    def make_mrotary_embedding(self, name, root_input, output, **kwargs):
        # position_ids -> Gather(T/H/W) -> select cos/sin cache columns --+
        # root_input -> Reshape(heads) -> rotate pairs <-----------------+
        #                            +-> unrotated tail -> Concat -> Flatten
        if self.use_paged_attention:
            raise ValueError("TRT-RTX MRoPE expansion requires non-paged attention")
        dtype = kwargs["dtype"]
        num_heads = kwargs["num_heads"]
        rotary_dim = self.rope_attrs["rotary_embedding_dim"] or self.head_size
        owners = self.get_mrope_owners(rotary_dim)
        leading = list(self.values[root_input].shape)[:-1]
        positions = self.make_mrope_positions(name, kwargs["position_ids"], leading)
        axes = self.make_expansion_constant(f"{name}/head_axis", [2])
        cos, sin = [
            self.make_mrope_cache(name, kind, kwargs[f"{kind}_cache_name"], positions, owners, axes, dtype, leading)
            for kind in ("cos", "sin")
        ]
        reshape = self.make_expansion_constant(f"{name}/head_shape", [0, 0, num_heads, self.head_size])
        self.make_reshape(f"{name}/Reshape", [root_input, reshape], dtype, [*leading, num_heads, self.head_size])
        heads = f"{name}/Reshape/output_0"
        rotated = self.make_mrope_rotation(name, heads, cos, sin, dtype, [*leading, num_heads, rotary_dim])
        self.make_mrope_output(name, heads, rotated, output, dtype, [*leading, num_heads, rotary_dim])

    def make_layernorm_subgraph(self, name, **kwargs):
        # This method can be used to create multiple LayerNorm operations
        op_type = kwargs.pop("op_type")
        inputs = kwargs.pop("inputs")
        outputs = kwargs.pop("outputs")
        skip = kwargs.pop("skip")
        new_io_dtype = kwargs.pop("new_io_dtype")

        if op_type == "LayerNormalization":
            # Create LayerNorm op
            self.make_layernorm_op(name, op_type, inputs, outputs, skip, new_io_dtype, **kwargs)

        elif op_type == "SkipLayerNormalization":
            # Create subgraph to calculate SkipLayerNorm
            self.make_skip_layer_norm(
                name,
                root_input=inputs[0],
                skip_input=inputs[1],
                weight_name=inputs[2],
                bias_name=inputs[3],
                output_0=outputs[0],
                output_3=outputs[3] if len(outputs) > 3 else None,
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

        elif op_type == "SimplifiedLayerNormalization":
            # Create subgraph to calculate RMSNorm
            self.make_simplified_layer_norm(
                name,
                root_input=inputs[0],
                weight_name=inputs[1],
                output_0=outputs[0],
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

        elif op_type == "SkipSimplifiedLayerNormalization":
            # Create subgraph to calculate SkipRMSNorm
            self.make_skip_simplified_layer_norm(
                name,
                root_input=inputs[0],
                skip_input=inputs[1],
                weight_name=inputs[2],
                output_0=outputs[0],
                output_3=outputs[3] if len(outputs) > 3 else None,
                io_dtype=new_io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )

    def make_attention_mask_reformatting_for_gqa(self):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in GroupQueryAttention
        #
        # Key difference vs make_attention_mask_standard_reformatting_for_gqa:
        # - Standard mode: total_seq_len is calculated from Shape op (always runs on CPU)
        # - Graph capture mode: No Shape ops inserted to ensure all ops run on GPU (no CPU ops)
        #
        #          attention_mask
        #               |
        #         Cast to int32
        #               |
        #           ReduceSum (keepdims=0)
        #              /    \
        #             /      \
        #           Sub    ReduceMax
        #            |        |
        #       seqlens_k  total_seq_len
        #         (1D)       (int)
        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Calculate ReduceSum from attention_mask
        cast_1_name = f"{attn_mask_basename}/Cast"
        self.make_cast(
            cast_1_name,
            self.input_names["attention_mask"],
            dtype=ir.DataType.INT32,
            shape=["batch_size", "total_sequence_length"],
        )
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = [f"{cast_1_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        # Left branch: Calculate seqlens_k = ReduceSum - 1
        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT32/[1]"]
        self.make_sub(sub_name, sub_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        # Right branch: ReduceMax to get maximum int value for total_seq_len
        reduce_max_name = f"{attn_mask_basename}/ReduceMax"
        reduce_max_inputs = [f"{reduce_sum_name}/output_0"]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=ir.DataType.INT32, shape=[])

        self.mask_attrs["seqlens_k"] = sub_name
        self.mask_attrs["total_seq_len"] = reduce_max_name
