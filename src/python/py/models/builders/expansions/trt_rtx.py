# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
import onnx_ir as ir


def _emit(model, name, op_type, inputs, dtype, shape, **attributes):
    output = f"{name}/output_0"
    model.make_node(op_type, inputs, [output], name=name, **attributes)
    model.make_value(output, dtype, shape)
    return output


def _constant(model, name, value, dtype=np.int64):
    # Shape inference needs these small constants in the graph, not in external weight files.
    tensor = ir.tensor(np.asarray(value, dtype=dtype), name=name)
    model.make_node("Constant", [], [name], name=f"{name}/Constant", value=tensor)
    model.make_value(name, tensor.dtype, tensor.shape)
    return name


class TRT_RTX:
    """
    TRT-RTX specific subgraph expansions
    """

    def make_gated_add(self, name, root_input, scaled_input, gate, shape):
        mul_name = f"{name}/Mul"
        self.make_mul(mul_name, [scaled_input, gate], self.io_dtype, shape=shape)
        self.make_add(name, [root_input, f"{mul_name}/output_0"], self.io_dtype, shape=shape)

    def make_linear_attention_gate(self, name, a, dt_bias, decay_scale, b, shape):
        # Preserve the legacy float32 decay calculation before casting for LinearAttention.
        a_fp32 = _emit(self, f"{name}/a/Cast", "Cast", [a], ir.DataType.FLOAT, shape, to=ir.DataType.FLOAT)
        biased = _emit(self, f"{name}/Add", "Add", [a_fp32, dt_bias], ir.DataType.FLOAT, shape)
        softplus = _emit(self, f"{name}/Softplus", "Softplus", [biased], ir.DataType.FLOAT, shape)
        decay = _emit(self, f"{name}/Mul", "Mul", [softplus, decay_scale], ir.DataType.FLOAT, shape)
        self.make_node("Cast", [decay], [f"{name}/output_0"], name=name, to=self.io_dtype)
        self.make_node("Sigmoid", [b], [f"{name}/output_1"], name=f"{name}/Sigmoid")
        self.make_value(f"{name}/output_0", self.io_dtype, shape)
        self.make_value(f"{name}/output_1", self.io_dtype, shape)

    def make_gated_rms_norm(self, name, root_input, scale, gate, shape, epsilon=1e-5):
        # Normalize each head independently, then apply the gate in float32 as in the legacy path.
        head_size = int(self.values[scale].shape[0])
        grouped_shape = [*shape[:-1], shape[-1] // head_size, head_size]
        reshape = _constant(self, f"{name}/group_shape", [*([0] * (len(shape) - 1)), -1, head_size])
        grouped = _emit(self, f"{name}/Reshape", "Reshape", [root_input, reshape], self.io_dtype, grouped_shape)
        normalized = _emit(
            self,
            f"{name}/SimplifiedLayerNormalization",
            "SimplifiedLayerNormalization",
            [grouped, scale],
            self.io_dtype,
            grouped_shape,
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
        )
        restore = _constant(self, f"{name}/flat_shape", [*([0] * (len(shape) - 1)), shape[-1]])
        normalized = _emit(self, f"{name}/Flatten", "Reshape", [normalized, restore], self.io_dtype, shape)
        normalized = _emit(
            self, f"{name}/norm/Cast", "Cast", [normalized], ir.DataType.FLOAT, shape, to=ir.DataType.FLOAT
        )
        gate = _emit(self, f"{name}/gate/Cast", "Cast", [gate], ir.DataType.FLOAT, shape, to=ir.DataType.FLOAT)
        sigmoid = _emit(self, f"{name}/Sigmoid", "Sigmoid", [gate], ir.DataType.FLOAT, shape)
        silu = _emit(self, f"{name}/SiLU", "Mul", [gate, sigmoid], ir.DataType.FLOAT, shape)
        gated = _emit(self, f"{name}/Mul", "Mul", [normalized, silu], ir.DataType.FLOAT, shape)
        self.make_node("Cast", [gated], [f"{name}/output_0"], name=name, to=self.io_dtype)
        self.make_value(f"{name}/output_0", self.io_dtype, shape)

    def make_mrotary_embedding(self, name, root_input, output, **kwargs):
        # Select T/H/W cache columns explicitly, then apply the legacy rotary arithmetic.
        # Keep the rank-3 position-ID ABI; selecting only T would lose multimodal positions.
        if self.use_paged_attention:
            raise ValueError("TRT-RTX MRoPE expansion requires non-paged attention")
        dtype = kwargs["dtype"]
        num_heads = kwargs["num_heads"]
        head_size = self.head_size
        rotary_dim = self.rope_attrs["rotary_embedding_dim"] or head_size
        half = rotary_dim // 2
        sections = self.rope_attrs["mrope_section"]
        layout = self.rope_attrs["mrope_layout"]
        if layout not in (0, 1) or len(sections) != 3 or sum(sections) != half:
            raise ValueError(
                "TRT-RTX MRoPE requires sectioned/interleaved layout and sections summing to half the rotary dimension"
            )
        owners = np.zeros(half, dtype=np.int64)
        if layout == 0:
            owners = np.repeat(np.arange(3), sections)
        else:
            for dim in (1, 2):
                owners[dim : min(3 * sections[dim], half) : 3] = dim

        leading = list(self.values[root_input].shape)[:-1]
        cache_shape = [*leading, half]
        position_streams = []
        for dim in range(3):
            index = _constant(self, f"{name}/position_index_{dim}", dim)
            position_streams.append(
                _emit(
                    self,
                    f"{name}/positions_{dim}",
                    "Gather",
                    [kwargs["position_ids"], index],
                    ir.DataType.INT64,
                    leading,
                    axis=0,
                )
            )
        axes = _constant(self, f"{name}/head_axis", [2])
        caches = []
        for kind in ("cos", "sin"):
            streams = [
                _emit(
                    self,
                    f"{name}/{kind}_{dim}",
                    "Gather",
                    [kwargs[f"{kind}_cache_name"], positions],
                    dtype,
                    cache_shape,
                    axis=0,
                )
                for dim, positions in enumerate(position_streams)
            ]
            selected = streams[0]
            for dim in (1, 2):
                mask = _constant(self, f"{name}/{kind}_mask_{dim}", owners == dim, np.bool_)
                selected = _emit(
                    self, f"{name}/{kind}_select_{dim}", "Where", [mask, streams[dim], selected], dtype, cache_shape
                )
            caches.append(
                _emit(self, f"{name}/{kind}/Unsqueeze", "Unsqueeze", [selected, axes], dtype, [*leading, 1, half])
            )

        reshape = _constant(self, f"{name}/head_shape", [0, 0, num_heads, head_size])
        heads = _emit(
            self, f"{name}/Reshape", "Reshape", [root_input, reshape], dtype, [*leading, num_heads, head_size]
        )
        interleaved = self.rope_attrs["interleaved"]
        first = np.arange(0, rotary_dim, 2) if interleaved else np.arange(half)
        second = first + 1 if interleaved else first + half
        parts = []
        for label, indices in (("first", first), ("second", second)):
            index = _constant(self, f"{name}/{label}_indices", indices)
            parts.append(
                _emit(self, f"{name}/{label}", "Gather", [heads, index], dtype, [*leading, num_heads, half], axis=-1)
            )
        x1, x2 = parts
        cos, sin = caches
        rotated = []
        for label, left, right, op in (("first", x1, x2, "Sub"), ("second", x2, x1, "Add")):
            a = _emit(self, f"{name}/{label}/cos", "Mul", [left, cos], dtype, [*leading, num_heads, half])
            b = _emit(self, f"{name}/{label}/sin", "Mul", [right, sin], dtype, [*leading, num_heads, half])
            rotated.append(_emit(self, f"{name}/{label}/rotate", op, [a, b], dtype, [*leading, num_heads, half]))
        merged = _emit(self, f"{name}/Concat", "Concat", rotated, dtype, [*leading, num_heads, rotary_dim], axis=-1)
        if interleaved:
            order = _constant(
                self, f"{name}/interleave_indices", np.stack([np.arange(half), np.arange(half) + half], axis=1).ravel()
            )
            merged = _emit(
                self, f"{name}/Interleave", "Gather", [merged, order], dtype, [*leading, num_heads, rotary_dim], axis=-1
            )
        if rotary_dim < head_size:
            tail_index = _constant(self, f"{name}/tail_indices", np.arange(rotary_dim, head_size))
            tail = _emit(
                self,
                f"{name}/Tail",
                "Gather",
                [heads, tail_index],
                dtype,
                [*leading, num_heads, head_size - rotary_dim],
                axis=-1,
            )
            merged = _emit(
                self, f"{name}/ConcatTail", "Concat", [merged, tail], dtype, [*leading, num_heads, head_size], axis=-1
            )
        flat_shape = _constant(self, f"{name}/flat_shape", [0, 0, num_heads * head_size])
        self.make_node("Reshape", [merged, flat_shape], [output], name=name)
        self.make_value(output, dtype, [*leading, num_heads * head_size])

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
