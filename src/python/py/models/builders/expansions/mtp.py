# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License


class Qwen38MTP:
    """Qwen3.8 MTP normalization and projection expansions."""

    def make_offset_rmsnorm(self, name, root_input, weight_tensor):
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight_tensor + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=self.make_hidden_state_shape())
        return output

    def make_mtp_input_projection(self):
        basename = "/model/mtp"
        embed_weight = "model.embed_tokens.weight"
        self.make_initializer(self.mtp_weights.embedding.weight, embed_weight, to=self.io_dtype)
        embed_gather = f"{basename}/embed_tokens/Gather"
        self.make_node(
            "Gather",
            inputs=[embed_weight, self.input_names["input_ids"]],
            outputs=[f"{embed_gather}/output_0"],
            name=embed_gather,
        )
        self.make_value(f"{embed_gather}/output_0", self.io_dtype, self.make_hidden_state_shape())

        embedding_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_embedding",
            f"{embed_gather}/output_0",
            self.mtp_weights.pre_fc_norm_embedding.weight,
        )
        hidden_norm = self.make_branchwise_rms_norm(
            f"{basename}/pre_fc_norm_hidden",
            self.input_names["hidden_states"],
            self.mtp_weights.pre_fc_norm_hidden,
            self.hidden_size,
        )
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        grouped_shape = [*token_shape, self.hyper_connection_attrs["num_branches"], self.hidden_size]
        grouped_dims = (
            [-1, self.hyper_connection_attrs["num_branches"], self.hidden_size]
            if self.use_paged_attention
            else [0, 0, self.hyper_connection_attrs["num_branches"], self.hidden_size]
        )
        hidden_grouped = f"{basename}/hidden/Reshape"
        self.make_reshape(
            hidden_grouped,
            [hidden_norm, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        hidden_proj = self.make_matmul(
            self.mtp_weights.fc_hidden,
            f"{basename}/fc_hidden/MatMul",
            f"{hidden_grouped}/output_0",
            output_shape=grouped_shape,
        )
        embedding_proj = self.make_matmul(
            self.mtp_weights.fc_embedding,
            f"{basename}/fc_embedding/MatMul",
            embedding_norm,
        )
        embedding_grouped = f"{basename}/fc_embedding/Unsqueeze"
        self.make_unsqueeze(
            embedding_grouped,
            [f"{embedding_proj}/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, self.hidden_size],
        )
        fused = f"{basename}/input_fusion/Add"
        self.make_add(
            fused,
            [f"{hidden_proj}/output_0", f"{embedding_grouped}/output_0"],
            self.io_dtype,
            grouped_shape,
        )
        flatten_dims = (
            [-1, self.hyper_connection_attrs["hidden_size"]]
            if self.use_paged_attention else [0, 0, self.hyper_connection_attrs["hidden_size"]]
        )
        flattened = f"{basename}/input_fusion/Reshape"
        self.make_reshape(
            flattened,
            [f"{fused}/output_0", f"/model/constants/INT64/{flatten_dims}"],
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"]),
        )
        return f"{flattened}/output_0"


class Qwen35MTP:
    """Qwen3.5 MTP normalization, embedding, and projection expansions."""

    def make_offset_rmsnorm(self, name, root_input, weight_tensor):
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight_tensor + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=self.make_hidden_state_shape())
        return output

    def make_mtp_input_projection(self):
        basename = "/model/mtp"

        embed_output = self.make_mtp_embedding(basename)
        self.make_value(embed_output, self.io_dtype, shape=self.make_hidden_state_shape())

        embedding_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_embedding", embed_output, self.mtp_weights.pre_fc_norm_embedding.weight
        )
        hidden_states_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_hidden",
            self.input_names["hidden_states"],
            self.mtp_weights.pre_fc_norm_hidden.weight,
        )

        concat_name = f"{basename}/fc/Concat"
        self.make_concat(
            concat_name,
            [embedding_norm, hidden_states_norm],
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=2 * self.hidden_size),
            axis=-1,
        )

        fc_name = self.make_matmul(self.mtp_weights.fc, f"{basename}/fc/MatMul", f"{concat_name}/output_0")
        return f"{fc_name}/output_0"

    def make_mtp_embedding(self, basename):
        return self.make_embedding_lookup(
            self.mtp_weights.embedding.weight,
            f"{basename}/embed_tokens",
            self.mtp_weights.lm_head,
        )
