# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
import onnx_ir as ir


class TRT_RTX:
    """
    TRT-RTX specific subgraph expansions
    """

    def make_expansion_constant(self, name, value, dtype=np.int64):
        # Shape inference needs these small constants in the graph, not in external weight files.
        tensor = ir.tensor(np.asarray(value, dtype=dtype), name=name)
        self.make_node("Constant", [], [name], name=f"{name}/Constant", value=tensor)
        self.make_value(name, tensor.dtype, tensor.shape)
        return name

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
        self.make_node(
            "SimplifiedLayerNormalization",
            [f"{name}/Reshape/output_0", scale],
            [normalized],
            name=f"{name}/SimplifiedLayerNormalization",
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
        )
        self.make_value(normalized, self.io_dtype, grouped_shape)
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
        dtype = kwargs["dtype"]
        num_heads = kwargs["num_heads"]
        rotary_dim = self.rope_attrs["rotary_embedding_dim"] or self.head_size
        owners = self.get_mrope_owners(rotary_dim)
        leading = list(self.values[root_input].shape)[:-1]
        positions = self.make_mrope_positions(name, kwargs["position_ids"], leading)
        axes = self.make_expansion_constant(f"{name}/head_axis", [len(leading)])
        cos, sin = [
            self.make_mrope_cache(name, kind, kwargs[f"{kind}_cache_name"], positions, owners, axes, dtype, leading)
            for kind in ("cos", "sin")
        ]
        reshape = self.make_expansion_constant(f"{name}/head_shape", [*([0] * len(leading)), num_heads, self.head_size])
        self.make_reshape(f"{name}/Reshape", [root_input, reshape], dtype, [*leading, num_heads, self.head_size])
        heads = f"{name}/Reshape/output_0"
        rotated = self.make_mrope_rotation(name, heads, cos, sin, dtype, [*leading, num_heads, rotary_dim])
        self.make_mrope_output(name, heads, rotated, output, dtype, [*leading, num_heads, rotary_dim])

    def get_mrope_owners(self, rotary_dim):
        """Map each rotary cache column to its temporal, height, or width position stream."""
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > self.head_size:
            raise ValueError("TRT-RTX MRoPE requires a positive, even rotary dimension no greater than the head size")
        half = rotary_dim // 2
        sections = self.rope_attrs["mrope_section"]
        layout = self.rope_attrs["mrope_layout"]
        if (
            layout not in (0, 1)
            or len(sections) != 3
            or any(section < 0 for section in sections)
            or sum(sections) != half
        ):
            raise ValueError(
                "TRT-RTX MRoPE requires sectioned/interleaved layout and three non-negative sections "
                "summing to half the rotary dimension"
            )
        if layout == 0:
            return np.repeat(np.arange(3), sections)
        # Match ORT MRotaryEmbedding / Qwen apply_interleaved_mrope: T is the
        # default; H/W replace every third slot up to their section bounds.
        # This is not round-robin exhaustion: [1, 1, 2] maps to [T, H, W, T].
        owners = np.zeros(half, dtype=np.int64)
        for dim in (1, 2):
            owners[dim : min(3 * sections[dim], half) : 3] = dim
        return owners

    def make_mrope_positions(self, name, position_ids, shape):
        # position_ids[3, B, S] -> Gather(0), Gather(1), Gather(2)
        positions = []
        for dim in range(3):
            index = self.make_expansion_constant(f"{name}/position_index_{dim}", dim)
            self.make_gather(f"{name}/positions_{dim}", [position_ids, index], ir.DataType.INT64, shape, axis=0)
            positions.append(f"{name}/positions_{dim}/output_0")
        return positions

    def make_mrope_cache(self, name, kind, cache, positions, owners, axes, dtype, leading):
        # cache -> Gather(T/H/W positions) -> Where(H) -> Where(W) -> Unsqueeze(head axis)
        cache_shape = [*leading, len(owners)]
        streams = []
        for dim, position in enumerate(positions):
            self.make_gather(f"{name}/{kind}_{dim}", [cache, position], dtype, cache_shape, axis=0)
            streams.append(f"{name}/{kind}_{dim}/output_0")
        selected = streams[0]
        for dim in (1, 2):
            mask = self.make_expansion_constant(f"{name}/{kind}_mask_{dim}", owners == dim, np.bool_)
            self.make_where(f"{name}/{kind}_select_{dim}", [mask, streams[dim], selected], dtype, cache_shape)
            selected = f"{name}/{kind}_select_{dim}/output_0"
        self.make_unsqueeze(f"{name}/{kind}/Unsqueeze", [selected, axes], dtype, [*leading, 1, len(owners)])
        return f"{name}/{kind}/Unsqueeze/output_0"

    def make_mrope_rotation(self, name, heads, cos, sin, dtype, shape):
        # heads -> Gather(x1/x2) -> [x1*cos - x2*sin, x2*cos + x1*sin] -> Concat
        #                                                              -> optional pair interleaving
        rotary_dim = shape[-1]
        half = rotary_dim // 2
        pair_shape = [*shape[:-1], half]
        interleaved = self.rope_attrs["interleaved"]
        first = np.arange(0, rotary_dim, 2) if interleaved else np.arange(half)
        second = first + 1 if interleaved else first + half
        parts = []
        for label, indices in (("first", first), ("second", second)):
            index = self.make_expansion_constant(f"{name}/{label}_indices", indices)
            self.make_gather(f"{name}/{label}", [heads, index], dtype, pair_shape, axis=-1)
            parts.append(f"{name}/{label}/output_0")
        x1, x2 = parts
        rotated = []
        for label, left, right, make_op in (("first", x1, x2, self.make_sub), ("second", x2, x1, self.make_add)):
            self.make_mul(f"{name}/{label}/cos", [left, cos], dtype, pair_shape)
            self.make_mul(f"{name}/{label}/sin", [right, sin], dtype, pair_shape)
            make_op(
                f"{name}/{label}/rotate",
                [f"{name}/{label}/cos/output_0", f"{name}/{label}/sin/output_0"],
                dtype,
                pair_shape,
            )
            rotated.append(f"{name}/{label}/rotate/output_0")
        self.make_concat(f"{name}/Concat", rotated, dtype, shape, axis=-1)
        merged = f"{name}/Concat/output_0"
        if interleaved:
            order = self.make_expansion_constant(
                f"{name}/interleave_indices", np.stack([np.arange(half), np.arange(half) + half], axis=1).ravel()
            )
            self.make_gather(f"{name}/Interleave", [merged, order], dtype, shape, axis=-1)
            merged = f"{name}/Interleave/output_0"
        return merged

    def make_mrope_output(self, name, heads, rotated, output, dtype, shape):
        # rotated + Gather(unrotated tail) -> Concat -> Reshape[B, S, N*H]
        rotary_dim = shape[-1]
        if rotary_dim < self.head_size:
            tail_index = self.make_expansion_constant(f"{name}/tail_indices", np.arange(rotary_dim, self.head_size))
            self.make_gather(
                f"{name}/Tail", [heads, tail_index], dtype, [*shape[:-1], self.head_size - rotary_dim], axis=-1
            )
            self.make_concat(
                f"{name}/ConcatTail", [rotated, f"{name}/Tail/output_0"], dtype, [*shape[:-1], self.head_size], axis=-1
            )
            rotated = f"{name}/ConcatTail/output_0"
        flat_shape = self.make_expansion_constant(
            f"{name}/flat_shape", [*([0] * (len(shape) - 2)), shape[-2] * self.head_size]
        )
        self.make_node("Reshape", [rotated, flat_shape], [output], name=name)
        self.make_value(output, dtype, [*shape[:-2], shape[-2] * self.head_size])

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

    def make_skip_simplified_layer_norm(
        self, basename, root_input, skip_input, weight_name, output_0, output_3, io_dtype, shape
    ):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      SimplifiedLayerNorm----> output (0)
        make_add_name = f"{basename}/Add"
        output_3 = f"{make_add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        make_simplified_layer_norm_name = f"{basename}/skip_simplified_layer_norm"
        self.make_simplified_layer_norm(
            make_simplified_layer_norm_name, output_3, weight_name, output_0, io_dtype, shape=shape
        )

    def make_skip_layer_norm(
        self, basename, root_input, skip_input, weight_name, bias_name, output_0, output_3, io_dtype, shape
    ):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      LayerNormalization-----> output (0)
        make_add_name = f"{basename}/Add"
        output_3 = f"{make_add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        make_layer_norm_name = f"{basename}/LayerNormalization"
        inputs = [output_3, weight_name, bias_name]

        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        kwargs.update({"axis": -1, "stash_type": 1})

        self.make_node("LayerNormalization", inputs=inputs, outputs=[output_0], name=make_layer_norm_name, **kwargs)
        self.make_value(output_0, io_dtype, shape=shape)

    # This expansion contrib-op can be updated / deprecated in the future.
    def make_simplified_layer_norm(self, basename, root_input, weight_name, output_0, io_dtype, shape):
        #                            Cast (float32) - most calc happens in higher precision
        #                              |
        #                      +-------+-------+
        #                      |               |
        #                     Pow              |
        #                      |               |
        #                  ReduceMean          |
        #                      |               |
        #                     Add              |
        #                      |               |
        #                    Sqrt              |
        #                      |               |
        #                     Div              |
        #                      |               |
        #                      +-------+-------+
        #                              |
        #                             Mul
        #                              |
        #                            Cast_1 (io_dtype - float16)
        #                              |
        #                            Mul_1

        make_cast_name = f"{basename}/Cast"
        self.make_cast(make_cast_name, root_input, ir.DataType.FLOAT, shape=shape)

        make_pow_name = f"{basename}/Pow"
        make_pow_inputs = [f"{make_cast_name}/output_0", "/model/constants/FLOAT/2"]

        self.make_node(
            "Pow", inputs=make_pow_inputs, outputs=[f"{make_pow_name}/output_0"], name=make_pow_name, domain=""
        )
        self.make_value(f"{make_pow_name}/output_0", ir.DataType.FLOAT, shape=shape)

        make_reducemean_name = f"{basename}/ReduceMean"
        make_reducemean_inputs = [f"{make_pow_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_reduce_mean(
            make_reducemean_name, make_reducemean_inputs, ir.DataType.FLOAT, keepdims=True, shape=shape
        )

        make_add_name = f"{basename}/Add"
        make_add_inputs = [
            f"{make_reducemean_name}/output_0",
            f"/model/constants/FLOAT/{self.layernorm_attrs['epsilon']}",
        ]
        self.make_add(make_add_name, make_add_inputs, ir.DataType.FLOAT, shape=shape)

        make_sqrt_name = f"{basename}/Sqrt"
        make_sqrt_inputs = [f"{make_add_name}/output_0"]
        self.make_sqrt(make_sqrt_name, make_sqrt_inputs, ir.DataType.FLOAT, shape=shape)

        make_div_name = f"{basename}/Div"
        make_div_inputs = ["/model/constants/FLOAT/1", f"{make_sqrt_name}/output_0"]
        self.make_div(make_div_name, make_div_inputs, ir.DataType.FLOAT, shape=shape)

        make_mul_name = f"{basename}/Mul"
        make_mul_inputs = [f"{make_div_name}/output_0", f"{make_cast_name}/output_0"]
        self.make_mul(make_mul_name, make_mul_inputs, ir.DataType.FLOAT, shape=shape)

        make_cast_1_name = f"{basename}/Cast_1"
        self.make_cast(make_cast_1_name, f"{make_mul_name}/output_0", dtype=io_dtype, shape=shape)

        make_mul_1_name = f"{basename}/Mul_1"
        make_mul_1_inputs = [f"{make_cast_1_name}/output_0", weight_name]

        self.make_node("Mul", inputs=make_mul_1_inputs, outputs=[output_0], name=make_mul_1_name)
        self.make_value(output_0, dtype=io_dtype, shape=shape)

    def make_causal_conv_with_state(self, name, **kwargs):
        inputs = [
            kwargs["root_input"],
            kwargs["weight"],
            kwargs["bias"],
            kwargs["past_conv_state"],
        ]
        output = f"{name}/output_0"

        attributes = {
            "ndim": kwargs.get("ndim", 1),
            "activation": kwargs.get("activation", "silu"),
        }
        if self.context_length_attrs["state_window"]:
            attributes["state_window"] = self.context_length_attrs["state_window"]

        self.make_node(
            "CausalConvWithState",
            inputs=inputs,
            outputs=[output, kwargs["present_conv_state"]],
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, shape=["batch_size", kwargs["channels"], "sequence_length"])

    def make_linear_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"],
            kwargs["k_path"],
            kwargs["v_path"],
            kwargs["past_recurrent_state"],
            kwargs["decay"],
            kwargs["beta"],
        ]
        output = f"{name}/output_0"

        attributes = {
            "q_num_heads": kwargs["q_num_heads"],
            "kv_num_heads": kwargs["kv_num_heads"],
            "update_rule": kwargs.get("update_rule", "gated_delta"),
            "scale": kwargs.get("scale", 1.0),
        }
        if self.context_length_attrs["state_window"]:
            attributes["state_window"] = self.context_length_attrs["state_window"]

        self.make_node(
            "LinearAttention",
            inputs=inputs,
            outputs=[output, kwargs["present_recurrent_state"]],
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, shape=["batch_size", "sequence_length", self.linear_value_dim])
