# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir
import torch


class Qwen38Engram:
    """Paged Engram expansion and layout-independent graph primitives."""

    def make_ple(self, layer_id, ple, root_input):
        basename = f"/model/layers.{layer_id}/ple"
        ngram_op_type = "VarlenNGramHashMapping"
        ngram_name = f"{basename}/{ngram_op_type}"
        ngram_ids = f"{ngram_name}/output_0"
        embedding = ple.ple_embedding
        token_shape = ["num_tokens"]
        state_update_capacity = getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0)
        flatten_dims = [-1, self.ple_attrs["embed_dim"]]
        eos_token_id = self.get_engram_pad_id(embedding.eos_token_id)
        multipliers = f"model.layers.{layer_id}.ple.layer_multipliers"
        vocab_sizes = f"model.layers.{layer_id}.ple.head_vocab_sizes"
        offsets = f"model.layers.{layer_id}.ple.head_offsets"
        eos = f"model.layers.{layer_id}.ple.eos_token_id"
        self.make_initializer(embedding.layer_multipliers, multipliers)
        self.make_initializer(embedding.ngram_heads_vocab_sizes, vocab_sizes)
        self.make_initializer(embedding.ngram_heads_offsets, offsets)
        self.make_initializer(torch.tensor(eos_token_id, dtype=torch.int64), eos)
        ngram_inputs = [self.input_names["input_ids"], multipliers, vocab_sizes]
        ngram_inputs.append(self.input_names["cumulative_sequence_lengths"])
        ngram_inputs.extend([self.input_names["past.ple_tokens"][layer_id], offsets, eos])
        if state_update_capacity:
            ngram_inputs.extend(["", "", self.input_names["state_update.capture_count"]])
        ngram_outputs = [ngram_ids, self.output_names["present.ple_tokens"][layer_id]]
        if state_update_capacity:
            ngram_outputs.extend(["", self.output_names["state_update.ple_tokens"][layer_id]])
        self.make_node(
            ngram_op_type,
            inputs=ngram_inputs,
            outputs=ngram_outputs,
            name=ngram_name,
            domain="com.microsoft",
            max_ngram_size=self.ple_attrs["ngram_size"],
            n_head_per_ngram=self.ple_attrs["heads_per_ngram"],
            pad_id=eos_token_id,
            reset_on_eos=1,
            **{"state_update_capacity": state_update_capacity} if state_update_capacity else {},
        )
        ngram_heads = (self.ple_attrs["ngram_size"] - 1) * self.ple_attrs["heads_per_ngram"]
        self.make_value(ngram_ids, ir.DataType.INT64, ["num_tokens", ngram_heads])
        if self.ple_attrs["external"]:
            self.make_value(ngram_outputs[1], ir.DataType.INT64, ["batch_size", self.ple_attrs["ngram_size"] - 1])
        if self.ple_attrs["external"]:
            engram_embeddings = self.input_names["engram_embeddings"]
        else:
            engram_embeddings = self.make_ple_embedding(
                basename, embedding, ngram_ids, ngram_heads, token_shape, flatten_dims
            )
        grouped_dims = [-1, self.hyper_connection_attrs["num_branches"], self.hidden_size]
        gate_inputs, grouped_shape, key_scale, query_scale, conv_scale = self.make_ple_projections(
            layer_id, ple, root_input, engram_embeddings, token_shape, grouped_dims
        )
        gate_name = f"{basename}/EngramGate"
        gated_value = f"{gate_name}/output_0"
        gated_value_normed = f"{gate_name}/output_1"
        gate_shape = [1, *grouped_shape]
        for index, value in enumerate(gate_inputs):
            name = f"{gate_name}/input_{index}/Unsqueeze"
            self.make_unsqueeze(
                name,
                [value, "/model/constants/INT64/[0]"],
                self.io_dtype,
                [1, *grouped_shape] if index < 2 else [1, *token_shape, self.hidden_size],
            )
            gate_inputs[index] = f"{name}/output_0"
        self.make_ple_gate(
            gate_name, gated_value, gated_value_normed, gate_inputs, key_scale, query_scale, conv_scale, gate_shape
        )
        ple_shape, gated_value_flat, gated_value_normed_flat, conv_weight = self.make_ple_conv_inputs(
            layer_id, ple, gate_name, gated_value, gated_value_normed, token_shape, flatten_dims
        )
        conv_op_type = "VarlenCausalConvWithState"
        conv_name = f"{basename}/{conv_op_type}"
        conv_output = f"{conv_name}/output_0"
        self.make_varlen_causal_conv_with_state(
            conv_name,
            root_input=f"{gated_value_normed_flat}/output_0",
            weight=conv_weight,
            cumulative_sequence_length=self.input_names["cumulative_sequence_lengths"],
            bias="",
            past_conv_state=self.input_names["past.ple_conv"][layer_id],
            present_conv_state=self.output_names["present.ple_conv"][layer_id],
            output_shape=ple_shape,
            present_conv_shape=[
                "batch_size",
                self.hyper_connection_attrs["hidden_size"],
                self.ple_attrs["conv_dilation"] * (self.ple_attrs["conv_kernel_size"] - 1),
            ],
            dilation=self.ple_attrs["conv_dilation"],
            **{
                "state_update_capacity": state_update_capacity,
                "state_update_capture_count": self.input_names["state_update.capture_count"],
                "state_update_active": self.input_names["state_update.active"],
                "state_update_value": self.output_names["state_update.ple_conv_value"][layer_id],
                "state_update_value_shape": self.output_shapes["state_update.ple_conv_value"],
            }
            if state_update_capacity
            else {},
        )
        add_name = f"{basename}/Add"
        self.make_add(add_name, [f"{gated_value_flat}/output_0", conv_output], self.io_dtype, ple_shape)
        return f"{add_name}/output_0"
