# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy
import os

import onnx_ir as ir

from .text import Qwen4ExpTextModel


class Qwen4ExpMTPTextModel(Qwen4ExpTextModel):
    """Qwen4-Exp one-layer self-speculative MTP head builder."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options, multimodal_rotary=False):
        config = copy.deepcopy(config)
        config.text_config.num_hidden_layers = 1
        config.text_config.layer_types = ["indexed_attention"]
        config.text_config.ple_layer_ids = []
        config.num_hidden_layers = 1
        config.layer_types = ["indexed_attention"]
        self.mtp_layer_config = copy.deepcopy(config.text_config)

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        extra_options["text_only"] = True
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_sparse_rotary_init(multimodal_rotary)

        self.hyper_connection_attrs["tile_first_hidden_state"] = False
        self.hyper_connection_attrs["emit_pre_final_hidden_states"] = True
        self.include_hidden_states = True
        self.output_names["hidden_states"] = "hidden_states_out"
        self.output_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"])
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"])
        self.make_index_share_init(extra_options)

    def make_index_share_init(self, extra_options):
        if not self.use_paged_attention:
            return
        max_draft_tokens = int(extra_options.get("max_draft_tokens", 7))
        if not 1 <= max_draft_tokens <= 7:
            raise ValueError("IndexShare export requires between one and seven draft tokens.")
        capacity = self.sparse_indexer_attrs["token_budget"] + self.sparse_indexer_attrs["compress_ratio"] - 1
        self.sparse_indexer_attrs["index_share"] = {
            "enabled": True,
            "base_capacity": capacity,
            "max_draft_tokens": max_draft_tokens,
            "indices_output": "present_indices",
            "counts_output": "present_counts",
            "status_output": "indexer_status",
        }
        for name, dtype, shape in (
            ("mode", ir.DataType.INT32, [1]),
            ("projection_rows", ir.DataType.INT64, ["indexer_projection_rows"]),
            ("past_indices", ir.DataType.INT32, ["indexshare_rows", capacity]),
            ("past_counts", ir.DataType.INT32, ["indexshare_rows"]),
            ("base_row_indices", ir.DataType.INT32, ["indexshare_merge_queries"]),
            ("range_starts", ir.DataType.INT32, ["indexshare_merge_queries"]),
            ("range_ends", ir.DataType.INT32, ["indexshare_merge_queries"]),
        ):
            self.input_names[name] = name
            self.input_types[name] = dtype
            self.input_shapes[name] = shape
        output_capacity = capacity + max_draft_tokens - 1
        for name, shape in (
            ("present_indices", ["num_tokens", output_capacity]),
            ("present_counts", ["num_tokens"]),
            ("indexer_status", ["num_tokens"]),
        ):
            self.output_names[name] = name
            self.output_types[name] = ir.DataType.INT32
            self.output_shapes[name] = shape

    def get_final_hyper_connection_mixer(self):
        return self.mtp_weights.hyper_connection_mixer

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True
        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_lm_head(self.mtp_weights.lm_head)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            model_dir = snapshot_download(
                repo_id=model_dir,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        try:
            from loaders.mtp import Qwen4ExpMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.mtp import Qwen4ExpMTPModel  # noqa: PLC0415

        self.mtp_weights = Qwen4ExpMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.mtp_layer_config,
            preserve_quantization=self.quant_config.checkpoint_policy == "preserve",
            load_quantized_model=self.load_weights,
        )

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

        norm_name = f"{basename}/pre_fc_norm_embedding"
        weight_name = f"{norm_name[1:].replace('/', '.')}.weight"
        self.make_initializer(
            self.mtp_weights.pre_fc_norm_embedding.weight + self.layernorm_attrs["add_offset"],
            weight_name, to=self.io_dtype,
        )
        embedding_norm = f"{norm_name}/output_0"
        self.make_layernorm_subgraph(
            norm_name, op_type="SimplifiedLayerNormalization",
            inputs=[f"{embed_gather}/output_0", weight_name], outputs=[embedding_norm],
            skip=False, new_io_dtype=self.io_dtype, epsilon=self.layernorm_attrs["epsilon"], axis=-1, stash_type=1,
        )
        self.make_value(embedding_norm, self.io_dtype, self.make_hidden_state_shape())
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
