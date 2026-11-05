# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from .standard import Standard


class TRT_RTX:  # noqa: N801
    make_gated_add = Standard.make_gated_add
    make_linear_attention_gate = Standard.make_linear_attention_gate
    make_gated_rms_norm = Standard.make_gated_rms_norm
    make_mrotary_embedding = Standard.make_mrotary_embedding
    make_layernorm_subgraph = Standard.make_layernorm_subgraph
    make_skip_simplified_layer_norm = Standard.make_skip_simplified_layer_norm
    make_skip_layer_norm = Standard.make_skip_layer_norm
    make_simplified_layer_norm = Standard.make_simplified_layer_norm
    make_mrope_positions = Standard.make_mrope_positions
    make_mrope_cache = Standard.make_mrope_cache
    make_mrope_rotation = Standard.make_mrope_rotation
    make_mrope_output = Standard.make_mrope_output
    make_branchwise_rms_norm = Standard.make_branchwise_rms_norm
    make_scaled_silu = Standard.make_scaled_silu
    make_hyper_connection_pre_mix = Standard.make_hyper_connection_pre_mix
    make_hyper_connection_post_mix = Standard.make_hyper_connection_post_mix

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
