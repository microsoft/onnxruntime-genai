# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx_ir as ir


class Qwen38:
    """Standard ONNX subgraphs retained around Qwen3.8 contrib-op replacements."""

    def make_branchwise_rms_norm_expansion(self, name, root_input, norm, hidden_size):
        """Apply RMS normalization independently to each hyper-connection branch."""
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        grouped_shape = [*token_shape, self.hc_count, hidden_size]
        grouped_dims = [-1, self.hc_count, hidden_size] if self.use_paged_attention else [0, 0, self.hc_count, hidden_size]
        reshape_name = f"{name}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        norm_scale_name = f"{name[1:].replace('/', '.')}.norm_scale"
        self.make_initializer(norm.weight.new_ones(hidden_size), norm_scale_name, to=self.io_dtype)
        normalized_name = f"{name}/SimplifiedLayerNormalization"
        normalized = f"{normalized_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[f"{reshape_name}/output_0", norm_scale_name],
            outputs=[normalized],
            name=normalized_name,
            axis=-1,
            epsilon=self.layernorm_attrs["epsilon"],
            stash_type=1,
        )
        self.make_value(normalized, self.io_dtype, grouped_shape)
        flatten_name = f"{name}/Flatten"
        flat_shape = [*token_shape, self.hc_count * hidden_size]
        flat_dims = (
            [-1, self.hc_count * hidden_size]
            if self.use_paged_attention
            else [0, 0, self.hc_count * hidden_size]
        )
        self.make_reshape(
            flatten_name,
            [normalized, f"/model/constants/INT64/{flat_dims}"],
            self.io_dtype,
            flat_shape,
        )
        scale_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(norm.weight + 1.0, scale_name, to=self.io_dtype)
        scale_mul_name = f"{name}/Scale"
        self.make_mul(scale_mul_name, [f"{flatten_name}/output_0", scale_name], self.io_dtype, flat_shape)
        return f"{scale_mul_name}/output_0"

    def make_scaled_silu_expansion(self, name, root_input, shape, alpha):
        """Expand ScaledSiLU into standard ONNX operators."""
        divide_name = f"{name}/Div"
        self.make_div(
            divide_name,
            [root_input, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{1 / alpha:g}"],
            self.io_dtype,
            shape,
        )
        sigmoid_name = f"{name}/Sigmoid"
        self.make_sigmoid(sigmoid_name, f"{divide_name}/output_0", self.io_dtype, shape)
        self.make_mul(
            name,
            [f"{divide_name}/output_0", f"{sigmoid_name}/output_0"],
            self.io_dtype,
            shape,
        )
        return f"{name}/output_0"

    def make_hyper_connection_pre_mix_expansion(self, name, streams, pre_mix, token_shape, hidden_size):
        """Expand HyperConnectionPreMix into reshape, multiply, and reduction nodes."""
        grouped_shape = [*token_shape, self.hc_count, hidden_size]
        grouped_dims = (
            [-1, self.hc_count, hidden_size]
            if self.use_paged_attention
            else [0, 0, self.hc_count, hidden_size]
        )
        stream_reshape = f"{name}/streams/Reshape"
        self.make_reshape(
            stream_reshape,
            [streams, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        gate_reshape = f"{name}/pre_mix/Reshape"
        self.make_reshape(
            gate_reshape,
            [pre_mix, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        weighted = f"{name}/Mul"
        self.make_mul(
            weighted,
            [f"{stream_reshape}/output_0", f"{gate_reshape}/output_0"],
            self.io_dtype,
            grouped_shape,
        )
        self.make_reduce_mean(
            name,
            [f"{weighted}/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, hidden_size],
        )
        return f"{name}/output_0"

    def make_hyper_connection_post_mix_expansion(
        self, name, streams, block_output, post_mix, token_shape, hidden_size
    ):
        """Expand identity-stream HyperConnectionPostMix into standard ONNX operators."""
        output_unsqueeze = f"{name}/output/Unsqueeze"
        self.make_unsqueeze(
            output_unsqueeze,
            [block_output, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, hidden_size],
        )
        weight_unsqueeze = f"{name}/weight/Unsqueeze"
        self.make_unsqueeze(
            weight_unsqueeze,
            [post_mix, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [*token_shape, self.hc_count, 1],
        )
        weighted = f"{name}/Mul"
        self.make_mul(
            weighted,
            [f"{output_unsqueeze}/output_0", f"{weight_unsqueeze}/output_0"],
            self.io_dtype,
            [*token_shape, self.hc_count, hidden_size],
        )
        flatten = f"{name}/Reshape"
        flat_dims = [-1, self.hc_count * hidden_size] if self.use_paged_attention else [0, 0, self.hc_count * hidden_size]
        self.make_reshape(
            flatten,
            [f"{weighted}/output_0", f"/model/constants/INT64/{flat_dims}"],
            self.io_dtype,
            [*token_shape, self.hc_count * hidden_size],
        )
        self.make_add(
            name,
            [streams, f"{flatten}/output_0"],
            self.io_dtype,
            [*token_shape, self.hc_count * hidden_size],
        )
        return f"{name}/output_0"

    def make_qsa_visibility_mask(self, layer_id, root_input):
        """Build the dense causal visibility mask consumed by SparseAttentionIndexer."""
        basename = f"/model/layers.{layer_id}/attn/indexer/visibility"
        attention_shape = f"{basename}/attention_mask/Shape"
        self.make_shape(attention_shape, self.input_names["attention_mask"], [2])
        total_length = f"{basename}/total_length/Gather"
        self.make_gather(
            total_length,
            [f"{attention_shape}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
            axis=0,
        )

        input_shape = f"{basename}/input/Shape"
        self.make_shape(input_shape, root_input, [3])
        sequence_length = f"{basename}/sequence_length/Gather"
        self.make_gather(
            sequence_length,
            [f"{input_shape}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
            axis=0,
        )
        past_length = f"{basename}/past_length/Sub"
        self.make_sub(
            past_length,
            [f"{total_length}/output_0", f"{sequence_length}/output_0"],
            ir.DataType.INT64,
            [],
        )

        key_positions = f"{basename}/key_positions/Range"
        self.make_range(
            key_positions,
            ["/model/constants/INT64/0", f"{total_length}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["total_sequence_length"],
        )
        query_limits = f"{basename}/query_limits/Range"
        self.make_range(
            query_limits,
            [f"{past_length}/output_0", f"{total_length}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["sequence_length"],
        )
        query_limits_inc = f"{basename}/query_limits/Add"
        self.make_add(
            query_limits_inc,
            [f"{query_limits}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["sequence_length"],
        )
        keys_2d = f"{basename}/key_positions/Unsqueeze"
        self.make_unsqueeze(
            keys_2d,
            [f"{key_positions}/output_0", "/model/constants/INT64/[0]"],
            ir.DataType.INT64,
            [1, "total_sequence_length"],
        )
        limits_2d = f"{basename}/query_limits/Unsqueeze"
        self.make_unsqueeze(
            limits_2d,
            [f"{query_limits_inc}/output_0", "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            ["sequence_length", 1],
        )
        causal = f"{basename}/causal/Less"
        self.make_less(causal, [f"{keys_2d}/output_0", f"{limits_2d}/output_0"])
        causal_4d = f"{basename}/causal/Unsqueeze"
        self.make_unsqueeze(
            causal_4d,
            [f"{causal}/output_0", "/model/constants/INT64/[0, 1]"],
            ir.DataType.BOOL,
            [1, 1, "sequence_length", "total_sequence_length"],
        )

        padding = f"{basename}/padding/Greater"
        self.make_greater(
            padding,
            [self.input_names["attention_mask"], "/model/constants/INT64/0"],
            ["batch_size", "total_sequence_length"],
        )
        padding_4d = f"{basename}/padding/Unsqueeze"
        self.make_unsqueeze(
            padding_4d,
            [f"{padding}/output_0", "/model/constants/INT64/[1, 2]"],
            ir.DataType.BOOL,
            ["batch_size", 1, 1, "total_sequence_length"],
        )
        visible = f"{basename}/And"
        self.make_and(
            visible,
            [f"{causal_4d}/output_0", f"{padding_4d}/output_0"],
            ["batch_size", 1, "sequence_length", "total_sequence_length"],
        )
        return f"{visible}/output_0"

    def make_selected_counts(self, layer_id, selected_indices, capacity, packed):
        """Expand the indexer's -1 padding convention into executor counts."""
        basename = f"/model/layers.{layer_id}/attn/indexer/selected"
        valid = f"{basename}/GreaterOrEqual"
        selected_shape = ["num_tokens", capacity] if packed else ["batch_size", "sequence_length", capacity]
        self.make_greater_or_equal(
            valid,
            [selected_indices, "/model/constants/INT32/0"],
            selected_shape,
        )
        valid_int = f"{basename}/Where"
        self.make_where(
            valid_int,
            [f"{valid}/output_0", "/model/constants/INT32/1", "/model/constants/INT32/0"],
            ir.DataType.INT32,
            selected_shape,
        )
        counts = f"{basename}/ReduceSum"
        count_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        self.make_reduce_sum(
            counts,
            [f"{valid_int}/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.INT32,
            count_shape,
        )
        return f"{counts}/output_0"
