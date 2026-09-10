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

    def make_gated_rms_norm(self, name, root_input, scale, gate, shape, epsilon=1e-5):
        output = f"{name}/output_0"
        norm_shape = shape
        norm_input = root_input
        norm_output = f"{name}/SimplifiedLayerNormalization/output_0"
        hidden_dim = shape[-1]

        if hidden_dim == self.linear_value_dim:
            norm_shape = [*shape[:-1], self.linear_num_value_heads, self.linear_value_head_dim]
            reshape_name = f"{name}/input_flat/Reshape"
            self.make_reshape(
                reshape_name,
                [root_input, f"/model/constants/INT64/[0, 0, {self.linear_num_value_heads}, {self.linear_value_head_dim}]"],
                self.io_dtype,
                norm_shape,
            )
            norm_input = f"{reshape_name}/output_0"

        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[norm_input, scale],
            outputs=[norm_output],
            name=f"{name}/SimplifiedLayerNormalization",
            epsilon=epsilon,
            axis=-1,
            stash_type=1,
        )
        self.make_value(norm_output, self.io_dtype, shape=norm_shape)

        if norm_shape != shape:
            reshape_name = f"{name}/norm_unflat/Reshape"
            self.make_reshape(
                reshape_name,
                [norm_output, f"/model/constants/INT64/[0, 0, {hidden_dim}]"],
                self.io_dtype,
                shape,
            )
            norm_output = f"{reshape_name}/output_0"

        gate_cast = f"{name}/gate_cast/Cast/output_0"
        self.make_cast(f"{name}/gate_cast/Cast", gate, ir.DataType.FLOAT, shape)
        gate_sigmoid = f"{name}/gate_sigmoid/Sigmoid/output_0"
        self.make_sigmoid(f"{name}/gate_sigmoid/Sigmoid", gate_cast, ir.DataType.FLOAT, shape)
        gate_silu = f"{name}/gate_silu/Mul/output_0"
        self.make_mul(f"{name}/gate_silu/Mul", [gate_cast, gate_sigmoid], ir.DataType.FLOAT, shape)

        norm_cast = f"{name}/norm_cast/Cast/output_0"
        self.make_cast(f"{name}/norm_cast/Cast", norm_output, ir.DataType.FLOAT, shape)
        gated = f"{name}/gated/Mul/output_0"
        self.make_mul(f"{name}/gated/Mul", [norm_cast, gate_silu], ir.DataType.FLOAT, shape)
        self.make_node("Cast", inputs=[gated], outputs=[output], name=f"{name}/output/Cast", to=self.io_dtype)
        self.make_value(output, self.io_dtype, shape=shape)

    def make_mrotary_embedding(self, name, root_input, output, **kwargs):
        num_heads = kwargs.pop("num_heads")
        position_ids = kwargs.pop("position_ids")
        cos_cache_name = kwargs.pop("cos_cache_name")
        sin_cache_name = kwargs.pop("sin_cache_name")
        dtype = kwargs.pop("dtype")
        rotary_embedding_dim = self.rope_attrs["rotary_embedding_dim"]
        rotary_dim = rotary_embedding_dim or self.head_size
        rotary_half_dim = rotary_dim // 2
        cache_shape = ["batch_size", "sequence_length", rotary_half_dim]

        def make_axis_cache(cache_name, cache_kind, axis):
            gather_position_name = f"{name}/{cache_kind}/position_ids_dim{axis}/Gather"
            self.make_gather(
                gather_position_name,
                [position_ids, f"/model/constants/INT64/[{axis}]"],
                ir.DataType.INT64,
                [1, "batch_size", "sequence_length"],
                axis=0,
            )
            squeeze_position_name = f"{name}/{cache_kind}/position_ids_dim{axis}/Squeeze"
            self.make_squeeze(
                squeeze_position_name,
                [f"{gather_position_name}/output_0", "/model/constants/INT64/[0]"],
                ir.DataType.INT64,
                ["batch_size", "sequence_length"],
            )

            gather_cache_name = f"{name}/{cache_kind}/dim{axis}/Gather"
            self.make_gather(
                gather_cache_name,
                [cache_name, f"{squeeze_position_name}/output_0"],
                dtype,
                cache_shape,
                axis=0,
            )
            return f"{gather_cache_name}/output_0"

        def make_mixed_cache(cache_name, cache_kind):
            axis_caches = [make_axis_cache(cache_name, cache_kind, axis) for axis in range(3)]
            if self.rope_attrs["mrope_layout"] == 0:
                sections = self.rope_attrs["mrope_section"]
                if sum(sections) != rotary_half_dim:
                    raise ValueError("Chunked MRoPE sections must sum to half the rotary embedding dimension.")
                chunks = []
                start = 0
                for axis, section in enumerate(sections):
                    slice_name = f"{name}/{cache_kind}/dim{axis}/Slice"
                    self.make_slice(
                        slice_name,
                        [
                            axis_caches[axis],
                            f"/model/constants/INT64/[{start}]",
                            f"/model/constants/INT64/[{start + section}]",
                            "/model/constants/INT64/[-1]",
                        ],
                        dtype,
                        ["batch_size", "sequence_length", section],
                    )
                    chunks.append(f"{slice_name}/output_0")
                    start += section
                concat_name = f"{name}/{cache_kind}/Concat"
                self.make_concat(concat_name, chunks, dtype, cache_shape, axis=-1)
                mixed_cache = f"{concat_name}/output_0"
            elif self.rope_attrs["mrope_layout"] == 1:
                sections = self.rope_attrs["mrope_section"]
                h_mask = np.zeros(rotary_half_dim, dtype=np.bool_)
                w_mask = np.zeros(rotary_half_dim, dtype=np.bool_)
                for idx in range(1, sections[1] * 3, 3):
                    if idx < rotary_half_dim:
                        h_mask[idx] = True
                for idx in range(2, sections[2] * 3, 3):
                    if idx < rotary_half_dim:
                        w_mask[idx] = True

                h_mask_name = f"{name}/{cache_kind}/h_mask"
                w_mask_name = f"{name}/{cache_kind}/w_mask"
                self.make_initializer(h_mask, h_mask_name)
                self.make_initializer(w_mask, w_mask_name)
                where_h_name = f"{name}/{cache_kind}/h/Where"
                self.make_where(where_h_name, [h_mask_name, axis_caches[1], axis_caches[0]], dtype, cache_shape)
                where_w_name = f"{name}/{cache_kind}/w/Where"
                self.make_where(where_w_name, [w_mask_name, axis_caches[2], f"{where_h_name}/output_0"], dtype, cache_shape)
                mixed_cache = f"{where_w_name}/output_0"
            else:
                raise ValueError(f"Unsupported MRoPE layout: {self.rope_attrs['mrope_layout']}")

            flat_name = f"{name}/{cache_kind}_flat/Reshape"
            self.make_reshape(
                flat_name,
                [mixed_cache, f"/model/constants/INT64/[-1, {rotary_half_dim}]"],
                dtype,
                ["total_token_count", rotary_half_dim],
            )
            return f"{flat_name}/output_0"

        flat_cos = make_mixed_cache(cos_cache_name, "cos")
        flat_sin = make_mixed_cache(sin_cache_name, "sin")

        shape_name = f"{name}/position_ids/Shape"
        self.make_shape(shape_name, position_ids, [3])
        batch_seq_shape_name = f"{name}/position_ids/batch_seq/Slice"
        self.make_slice(
            batch_seq_shape_name,
            [f"{shape_name}/output_0", "/model/constants/INT64/[1]", "/model/constants/INT64/[3]", "/model/constants/INT64/[0]"],
            ir.DataType.INT64,
            [2],
        )
        batch_name = f"{name}/position_ids/batch/Gather"
        self.make_gather(batch_name, [f"{shape_name}/output_0", "/model/constants/INT64/1"], ir.DataType.INT64, [], axis=0)
        sequence_name = f"{name}/position_ids/sequence/Gather"
        self.make_gather(sequence_name, [f"{shape_name}/output_0", "/model/constants/INT64/2"], ir.DataType.INT64, [], axis=0)
        total_name = f"{name}/position_ids/total/Mul"
        self.make_mul(total_name, [f"{batch_name}/output_0", f"{sequence_name}/output_0"], ir.DataType.INT64, [])
        range_name = f"{name}/position_ids/Range"
        self.make_range(range_name, ["/model/constants/INT64/0", f"{total_name}/output_0", "/model/constants/INT64/1"], ir.DataType.INT64, ["total_token_count"])
        flat_position_name = f"{name}/position_ids/Reshape"
        self.make_reshape(flat_position_name, [f"{range_name}/output_0", f"{batch_seq_shape_name}/output_0"], ir.DataType.INT64, ["batch_size", "sequence_length"])

        input_shape = ["batch_size", "sequence_length", num_heads, self.head_size]
        rotary_shape = ["batch_size", num_heads, "sequence_length", self.head_size]
        input_reshape_name = f"{name}/input/Reshape"
        self.make_reshape(
            input_reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"],
            dtype,
            input_shape,
        )
        input_transpose_name = f"{name}/input/Transpose"
        self.make_transpose(input_transpose_name, f"{input_reshape_name}/output_0", dtype, rotary_shape, perm=[0, 2, 1, 3])

        rotary_name = f"{name}/RotaryEmbedding"
        rotary_output = f"{rotary_name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [f"{input_transpose_name}/output_0", f"{flat_position_name}/output_0", flat_cos, flat_sin],
            [rotary_output],
            name=rotary_name,
            domain="com.microsoft",
            interleaved=self.rope_attrs["interleaved"],
            rotary_embedding_dim=rotary_embedding_dim,
            num_heads=num_heads,
        )
        self.make_value(rotary_output, dtype, shape=rotary_shape)

        output_transpose_name = f"{name}/output/Transpose"
        self.make_transpose(output_transpose_name, rotary_output, dtype, input_shape, perm=[0, 2, 1, 3])
        output_shape = ["batch_size", "sequence_length", self.head_size * num_heads]
        self.make_node(
            "Reshape",
            inputs=[f"{output_transpose_name}/output_0", f"/model/constants/INT64/[0, 0, {self.head_size * num_heads}]"],
            outputs=[output],
            name=f"{name}/output/Reshape",
        )
        self.make_value(output, dtype, shape=output_shape)

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

    def make_linear_attention_gate(self, name, a, dt_bias, decay_scale, b, shape):
        decay = f"{name}/output_0"
        beta = f"{name}/output_1"

        add = f"{name}/Add/output_0"
        self.make_add(f"{name}/Add", [a, dt_bias], self.io_dtype, shape)
        softplus = f"{name}/Softplus/output_0"
        self.make_softplus(f"{name}/Softplus", add, self.io_dtype, shape)

        self.make_node("Mul", inputs=[softplus, decay_scale], outputs=[decay], name=f"{name}/Mul", domain="")
        self.make_value(decay, self.io_dtype, shape=shape)
        self.make_node("Sigmoid", inputs=[b], outputs=[beta], name=f"{name}/Sigmoid", domain="")
        self.make_value(beta, self.io_dtype, shape=shape)
