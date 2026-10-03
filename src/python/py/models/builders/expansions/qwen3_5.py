import math

import onnx_ir as ir


class Qwen35:
    """Shared Qwen3.5 attention gate and normalization expansions."""

    def split_attention_query_gate(self, layer_id):
        """Split the doubled per-head query projection into query and output gate."""

        q_size = self.num_attn_heads * self.head_size
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        q_gate_reshape = [0, self.num_attn_heads, self.head_size * 2]
        q_reshape = [0, q_size]
        if not self.use_paged_attention:
            q_gate_reshape.insert(1, 0)
            q_reshape.insert(1, 0)

        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [self.attention_attrs["q_path"], f"/model/constants/INT64/{q_gate_reshape}"],
            self.io_dtype,
            [*token_shape, self.num_attn_heads, self.head_size * 2],
        )

        split_name = f"/model/layers.{layer_id}/attn/q_gate/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        q_gate_shape = [*token_shape, self.num_attn_heads, self.head_size]
        self.make_split(
            split_name,
            inputs=[rs_qg_output, f"/model/constants/INT64/[{self.head_size}, {self.head_size}]"],
            outputs=[q_4d_output, gate_4d_output],
            dtypes=[self.io_dtype, self.io_dtype],
            shapes=[q_gate_shape, q_gate_shape],
            axis=-1,
        )

        rs_q_name = f"/model/layers.{layer_id}/attn/q_proj/Reshape"
        self.make_reshape(
            rs_q_name,
            [q_4d_output, f"/model/constants/INT64/{q_reshape}"],
            self.io_dtype,
            [*token_shape, q_size],
        )

        rs_g_name = f"/model/layers.{layer_id}/attn/gate/Reshape"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/{q_reshape}"],
            self.io_dtype,
            [*token_shape, q_size],
        )

        self.attention_attrs["q_path"] = f"{rs_q_name}/output_0"
        self.attention_attrs["gate_path"] = f"{rs_g_name}/output_0"


    def make_attention_output_proj(self, layer_id, attention, root_input, **kwargs):
        """Apply Qwen3.5's attention output gate before the shared output projection."""
        q_size = self.num_attn_heads * self.head_size
        output_shape = self.make_hidden_state_shape(last_dim=q_size)
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(
            sigmoid_name,
            self.attention_attrs["gate_path"],
            self.io_dtype,
            output_shape,
        )

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(
            gated_name,
            [self.attention_attrs["o_path"], f"{sigmoid_name}/output_0"],
            self.io_dtype,
            output_shape,
        )
        self.attention_attrs["o_path"] = f"{gated_name}/output_0"

        super().make_attention_output_proj(layer_id, attention, root_input, **kwargs)


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

