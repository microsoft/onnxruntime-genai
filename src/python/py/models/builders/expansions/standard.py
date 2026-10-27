# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import numpy as np
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
        reshape = f"/model/constants/INT64/{[*([0] * (len(shape) - 1)), *grouped_shape[-2:]]}"
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
        restore = f"/model/constants/INT64/{[*([0] * (len(shape) - 1)), shape[-1]]}"
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
        axes = "/model/constants/INT64/[2]"
        cos, sin = [
            self.make_mrope_cache(name, kind, kwargs[f"{kind}_cache_name"], positions, owners, axes, dtype, leading)
            for kind in ("cos", "sin")
        ]
        reshape = f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"
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
                shape=self.values[inputs[0]].shape,
            )

        elif op_type == "SimplifiedLayerNormalization":
            # Create subgraph to calculate RMSNorm
            self.make_simplified_layer_norm(
                name,
                root_input=inputs[0],
                weight_name=inputs[1],
                output_0=outputs[0],
                io_dtype=new_io_dtype,
                shape=self.values[inputs[0]].shape,
                epsilon=kwargs.get("epsilon"),
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
                shape=self.values[inputs[0]].shape,
            )

    def make_skip_simplified_layer_norm(
        self, basename, root_input, skip_input, weight_name, output_0, output_3, io_dtype, shape
    ):
        add_name = f"{basename}/Add"
        output_3 = f"{add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", [root_input, skip_input], [output_3], name=add_name)
        self.make_value(output_3, io_dtype, shape)
        self.make_simplified_layer_norm(
            f"{basename}/skip_simplified_layer_norm",
            output_3,
            weight_name,
            output_0,
            io_dtype,
            shape,
        )

    def make_skip_layer_norm(
        self, basename, root_input, skip_input, weight_name, bias_name, output_0, output_3, io_dtype, shape
    ):
        add_name = f"{basename}/Add"
        output_3 = f"{add_name}/output_0" if output_3 is None else output_3
        self.make_node("Add", [root_input, skip_input], [output_3], name=add_name)
        self.make_value(output_3, io_dtype, shape)
        self.make_layernorm_op(
            f"{basename}/LayerNormalization",
            "LayerNormalization",
            [output_3, weight_name, bias_name],
            [output_0],
            False,
            io_dtype,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output_0, io_dtype, shape)

    def make_simplified_layer_norm(self, basename, root_input, weight_name, output_0, io_dtype, shape, epsilon=None):
        # Match SimplifiedLayerNormalization: accumulate in FP32, cast before scaling.
        self.make_cast(f"{basename}/Cast", root_input, ir.DataType.FLOAT, shape)
        self.make_node(
            "Pow",
            [f"{basename}/Cast/output_0", "/model/constants/FLOAT/2"],
            [f"{basename}/Pow/output_0"],
            name=f"{basename}/Pow",
        )
        self.make_value(f"{basename}/Pow/output_0", ir.DataType.FLOAT, shape)
        reduced_shape = [*shape[:-1], 1]
        self.make_reduce_mean(
            f"{basename}/ReduceMean",
            [f"{basename}/Pow/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            reduced_shape,
            keepdims=1,
        )
        epsilon = self.layernorm_attrs["epsilon"] if epsilon is None else epsilon
        self.make_add(
            f"{basename}/Add",
            [f"{basename}/ReduceMean/output_0", f"/model/constants/FLOAT/{epsilon}"],
            ir.DataType.FLOAT,
            reduced_shape,
        )
        self.make_sqrt(f"{basename}/Sqrt", [f"{basename}/Add/output_0"], ir.DataType.FLOAT, reduced_shape)
        self.make_div(
            f"{basename}/Div",
            ["/model/constants/FLOAT/1", f"{basename}/Sqrt/output_0"],
            ir.DataType.FLOAT,
            reduced_shape,
        )
        self.make_mul(
            f"{basename}/Mul", [f"{basename}/Div/output_0", f"{basename}/Cast/output_0"], ir.DataType.FLOAT, shape
        )
        self.make_cast(f"{basename}/Cast_1", f"{basename}/Mul/output_0", io_dtype, shape)
        self.make_node("Mul", [f"{basename}/Cast_1/output_0", weight_name], [output_0], name=f"{basename}/Mul_1")
        self.make_value(output_0, io_dtype, shape)

    def make_mrope_positions(self, name, position_ids, shape):
        positions = []
        for dim in range(3):
            index = f"{name}/position_index_{dim}"
            self.make_initializer(np.array(dim, dtype=np.int64), index)
            self.make_gather(f"{name}/positions_{dim}", [position_ids, index], ir.DataType.INT64, shape, axis=0)
            positions.append(f"{name}/positions_{dim}/output_0")
        return positions

    def make_mrope_cache(self, name, kind, cache, positions, owners, axes, dtype, leading):
        cache_shape = [*leading, len(owners)]
        streams = []
        for dim, position in enumerate(positions):
            self.make_gather(f"{name}/{kind}_{dim}", [cache, position], dtype, cache_shape, axis=0)
            streams.append(f"{name}/{kind}_{dim}/output_0")
        selected = streams[0]
        for dim in (1, 2):
            mask = f"{name}/{kind}_mask_{dim}"
            self.make_initializer(owners == dim, mask)
            self.make_where(f"{name}/{kind}_select_{dim}", [mask, streams[dim], selected], dtype, cache_shape)
            selected = f"{name}/{kind}_select_{dim}/output_0"
        self.make_unsqueeze(f"{name}/{kind}/Unsqueeze", [selected, axes], dtype, [*leading, 1, len(owners)])
        return f"{name}/{kind}/Unsqueeze/output_0"

    def make_mrope_rotation(self, name, heads, cos, sin, dtype, shape):
        rotary_dim = shape[-1]
        half = rotary_dim // 2
        pair_shape = [*shape[:-1], half]
        interleaved = self.rope_attrs["interleaved"]
        first = np.arange(0, rotary_dim, 2) if interleaved else np.arange(half)
        second = first + 1 if interleaved else first + half
        parts = []
        for label, indices in (("first", first), ("second", second)):
            index = f"{name}/{label}_indices"
            self.make_initializer(indices, index)
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
            order = f"{name}/interleave_indices"
            self.make_initializer(np.stack([np.arange(half), np.arange(half) + half], axis=1).ravel(), order)
            self.make_gather(f"{name}/Interleave", [merged, order], dtype, shape, axis=-1)
            merged = f"{name}/Interleave/output_0"
        return merged

    def make_mrope_output(self, name, heads, rotated, output, dtype, shape):
        rotary_dim = shape[-1]
        if rotary_dim < self.head_size:
            tail_index = f"{name}/tail_indices"
            self.make_initializer(np.arange(rotary_dim, self.head_size), tail_index)
            self.make_gather(
                f"{name}/Tail", [heads, tail_index], dtype, [*shape[:-1], self.head_size - rotary_dim], axis=-1
            )
            self.make_concat(
                f"{name}/ConcatTail", [rotated, f"{name}/Tail/output_0"], dtype, [*shape[:-1], self.head_size], axis=-1
            )
            rotated = f"{name}/ConcatTail/output_0"
        flat_shape = f"/model/constants/INT64/[0, 0, {shape[-2] * self.head_size}]"
        self.make_node("Reshape", [rotated, flat_shape], [output], name=name)
        self.make_value(output, dtype, [*shape[:-2], shape[-2] * self.head_size])

    def make_branchwise_rms_norm(self, name, root_input, norm, hidden_size):
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        branches = self.hyper_connection_attrs["num_branches"]
        grouped_shape = [*token_shape, branches, hidden_size]
        grouped_dims = [-1, branches, hidden_size] if self.use_paged_attention else [0, 0, branches, hidden_size]
        self.make_reshape(
            f"{name}/Reshape", [root_input, f"/model/constants/INT64/{grouped_dims}"], self.io_dtype, grouped_shape
        )
        norm_scale_name = f"{name[1:].replace('/', '.')}.norm_scale"
        self.make_initializer(norm.weight.new_ones(hidden_size), norm_scale_name, to=self.io_dtype)
        normalized = f"{name}/SimplifiedLayerNormalization/output_0"
        self.make_simplified_layer_norm(
            f"{name}/SimplifiedLayerNormalization",
            f"{name}/Reshape/output_0",
            norm_scale_name,
            normalized,
            self.io_dtype,
            grouped_shape,
            epsilon=self.layernorm_attrs["epsilon"],
        )
        flat_shape = [*token_shape, branches * hidden_size]
        flat_dims = [-1, branches * hidden_size] if self.use_paged_attention else [0, 0, branches * hidden_size]
        self.make_reshape(
            f"{name}/Flatten", [normalized, f"/model/constants/INT64/{flat_dims}"], self.io_dtype, flat_shape
        )
        scale_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(norm.weight + 1.0, scale_name, to=self.io_dtype)
        self.make_mul(name, [f"{name}/Flatten/output_0", scale_name], self.io_dtype, flat_shape)
        return f"{name}/output_0"

    def make_scaled_silu(self, name, root_input, shape):
        branches = self.hyper_connection_attrs["num_branches"]
        self.make_div(
            f"{name}/Div",
            [root_input, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{branches}"],
            self.io_dtype,
            shape,
        )
        self.make_sigmoid(f"{name}/Sigmoid", f"{name}/Div/output_0", self.io_dtype, shape)
        self.make_mul(name, [f"{name}/Div/output_0", f"{name}/Sigmoid/output_0"], self.io_dtype, shape)
        return f"{name}/output_0"

    def make_hyper_connection_pre_mix(self, name, streams, pre_mix, token_shape):
        branches = self.hyper_connection_attrs["num_branches"]
        grouped_shape = [*token_shape, branches, self.hidden_size]
        grouped_dims = (
            [-1, branches, self.hidden_size] if self.use_paged_attention else [0, 0, branches, self.hidden_size]
        )
        self.make_reshape(
            f"{name}/streams/Reshape", [streams, f"/model/constants/INT64/{grouped_dims}"], self.io_dtype, grouped_shape
        )
        self.make_reshape(
            f"{name}/pre_mix/Reshape", [pre_mix, f"/model/constants/INT64/{grouped_dims}"], self.io_dtype, grouped_shape
        )
        self.make_mul(
            f"{name}/Mul",
            [f"{name}/streams/Reshape/output_0", f"{name}/pre_mix/Reshape/output_0"],
            self.io_dtype,
            grouped_shape,
        )
        self.make_reduce_mean(
            name,
            [f"{name}/Mul/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, self.hidden_size],
            keepdims=0,
        )
        return f"{name}/output_0"

    def make_hyper_connection_post_mix(self, name, streams, block_output, post_mix, token_shape, output_name=None):
        branches = self.hyper_connection_attrs["num_branches"]
        self.make_unsqueeze(
            f"{name}/output/Unsqueeze",
            [block_output, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, self.hidden_size],
        )
        self.make_unsqueeze(
            f"{name}/weight/Unsqueeze",
            [post_mix, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [*token_shape, branches, 1],
        )
        self.make_mul(
            f"{name}/Mul",
            [f"{name}/output/Unsqueeze/output_0", f"{name}/weight/Unsqueeze/output_0"],
            self.io_dtype,
            [*token_shape, branches, self.hidden_size],
        )
        flat_shape = [*token_shape, branches * self.hidden_size]
        flat_dims = (
            [-1, branches * self.hidden_size] if self.use_paged_attention else [0, 0, branches * self.hidden_size]
        )
        self.make_reshape(
            f"{name}/Reshape",
            [f"{name}/Mul/output_0", f"/model/constants/INT64/{flat_dims}"],
            self.io_dtype,
            flat_shape,
        )
        output = f"{name}/output_0" if output_name is None else output_name
        self.make_node("Add", [streams, f"{name}/Reshape/output_0"], [output], name=name)
        self.make_value(output, self.io_dtype, flat_shape)
        return output

    def make_l2_normalize(self, basename, root_input):
        grouped_shape = ["batch_size", "sequence_length", self.linear_num_key_heads, self.linear_key_head_dim]
        grouped_dims = [0, 0, self.linear_num_key_heads, self.linear_key_head_dim]
        self.make_reshape(
            f"{basename}/flat/Reshape",
            [root_input, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        self.make_lp_normalization(
            f"{basename}/LpNormalization",
            f"{basename}/flat/Reshape/output_0",
            self.io_dtype,
            grouped_shape,
            axis=-1,
            p=2,
        )
        total_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.make_reshape(
            f"{basename}/unflat/Reshape",
            [f"{basename}/LpNormalization/output_0", f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return f"{basename}/unflat/Reshape/output_0"
