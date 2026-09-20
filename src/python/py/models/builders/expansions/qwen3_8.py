# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math

import onnx_ir as ir


class Qwen38:
    """Standard ONNX subgraphs retained around Qwen3.8 contrib-op replacements."""

    def make_gated_delta_net_expansion(self, layer_id, attention, conv_out_3d, b_name, a_name):
        """Expand dense GatedDeltaNet into gate preprocessing plus LinearAttention."""
        q_path, k_path, v_path, decay, beta = self.make_gated_delta_net_gates_expansion(
            layer_id, attention, conv_out_3d, b_name, a_name
        )
        name = f"/model/layers.{layer_id}/linear_attn/LinearAttention"
        self.make_linear_attention(
            name,
            q_path=q_path,
            k_path=k_path,
            v_path=v_path,
            past_recurrent_state=self.input_names["past.recurrent"][layer_id],
            present_recurrent_state=self.output_names["present.recurrent"][layer_id],
            decay=decay,
            beta=beta,
            q_num_heads=self.linear_num_key_heads,
            kv_num_heads=self.linear_num_value_heads,
            update_rule="gated_delta",
            scale=1.0,
        )
        return f"{name}/output_0"

    def make_gated_delta_net_gates_expansion(self, layer_id, attention, conv_out_3d, b_name, a_name):
        """Expand Qwen gate arithmetic and Q/K normalization around LinearAttention."""
        basename = f"/model/layers.{layer_id}/linear_attn"
        split_name = f"{basename}/split_qkv/Split"
        q_path = f"{split_name}/output_0"
        k_path = f"{split_name}/output_1"
        v_path = f"{split_name}/output_2"
        self.make_split(
            split_name,
            inputs=[
                conv_out_3d,
                f"/model/constants/INT64/[{self.linear_key_dim}, {self.linear_key_dim}, {self.linear_value_dim}]",
            ],
            outputs=[q_path, k_path, v_path],
            dtypes=[self.io_dtype] * 3,
            shapes=[
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_value_dim],
            ],
            axis=-1,
        )

        q_path = self.make_l2_normalize_expansion(f"{basename}/q_l2norm", q_path)
        k_path = self.make_l2_normalize_expansion(f"{basename}/k_l2norm", k_path)
        scale = 1.0 / math.sqrt(self.linear_key_head_dim)
        q_scaled = f"{basename}/q_scaled/Mul"
        self.make_mul(
            q_scaled,
            [q_path, f"/model/constants/{self.io_dtype}/{scale}"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_key_dim],
        )

        dt_bias = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(attention.dt_bias, dt_bias, to=ir.DataType.FLOAT)
        decay_scale = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        self.make_initializer((-attention.A_log.data.exp()).detach(), decay_scale, to=ir.DataType.FLOAT)
        gate_name = f"{basename}/LinearAttentionGate"
        self.make_linear_attention_gate(
            gate_name,
            a=f"{a_name}/output_0",
            dt_bias=dt_bias,
            decay_scale=decay_scale,
            b=f"{b_name}/output_0",
            shape=["batch_size", "sequence_length", self.linear_num_value_heads],
        )
        return (
            f"{q_scaled}/output_0",
            k_path,
            v_path,
            f"{gate_name}/output_0",
            f"{gate_name}/output_1",
        )

    def make_l2_normalize_expansion(self, basename, root_input):
        """Expand per-head L2 normalization without runtime Shape operators."""
        total_dim = self.linear_num_key_heads * self.linear_key_head_dim
        grouped_shape = [
            "batch_size",
            "sequence_length",
            self.linear_num_key_heads,
            self.linear_key_head_dim,
        ]
        flat_name = f"{basename}/flat/Reshape"
        self.make_reshape(
            flat_name,
            [root_input, f"/model/constants/INT64/[0, 0, {self.linear_num_key_heads}, {self.linear_key_head_dim}]"],
            self.io_dtype,
            grouped_shape,
        )
        norm_name = f"{basename}/LpNormalization"
        self.make_lp_normalization(
            norm_name,
            f"{flat_name}/output_0",
            self.io_dtype,
            grouped_shape,
            axis=-1,
            p=2,
        )
        unflat_name = f"{basename}/unflat/Reshape"
        self.make_reshape(
            unflat_name,
            [f"{norm_name}/output_0", f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return f"{unflat_name}/output_0"

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
