# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy

import onnx_ir as ir
from loaders.engram import EngramWeights

from .base import Model
from .expansions.qwen3_8 import Qwen38
from .qwen3_5 import Qwen35MoETextModel


class Qwen4ExpTextModel(Qwen38, Qwen35MoETextModel):
    """Qwen4-Exp decoder builder using external token/vision embeddings."""

    CPU_EMBEDDING_ANNOTATION = "cpu_embedding"
    prepare_engram_embedding = EngramWeights.prepare_engram_embedding
    get_engram_pad_id = EngramWeights.get_engram_pad_id

    def make_ple(self, layer_id, ple, root_input):
        if self.use_paged_attention:
            return super().make_ple(layer_id, ple, root_input)
        from .qwen3_8 import Qwen38Engram  # noqa: PLC0415

        return Qwen38Engram.make_ple(self, layer_id, ple, root_input)

    def resolve_model_loader(self):
        """Load Qwen4-Exp weights independently of the Qwen3.5 multimodal runtime alias."""
        return self.resolve_transformers_class("Qwen4ExpForConditionalGeneration")

    def is_packed_matmul_supported(self):
        return Model.is_packed_matmul_supported(self)

    def make_attention_init(self, config):
        super().make_attention_init(config)
        self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()

    def make_moe_router(self, layer_id, moe, root_input):
        return super().make_moe_router(layer_id, moe, root_input)

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        return super().make_shared_expert(layer_id, shared_expert, shared_expert_gate, root_input)

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = copy.deepcopy(extra_options)
        text_only = extra_options.get("text_only", False)
        extra_options["exclude_embeds"] = not text_only
        extra_options.setdefault("filename", "model.onnx" if text_only else "text.onnx")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if not hasattr(self, "context_length_attrs"):
            self.context_length_attrs = {"state_window": 0, "state_window_dims": []}
        self.use_cpu_embedding_gather = False
        self.model.metadata_props["qwen4_exp.past_indexer_names"] = "past.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.present_indexer_names"] = "present.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.past_ple_token_names"] = "past.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.present_ple_token_names"] = "present.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.past_ple_conv_names"] = "past.%d.ple_conv"
        self.model.metadata_props["qwen4_exp.present_ple_conv_names"] = "present.%d.ple_conv"

        self.hyper_connection_attrs = {
            "num_branches": config.hc_count,
            "hidden_size": config.hc_count * self.hidden_size,
            "tile_first_hidden_state": True,
            "emit_pre_final_hidden_states": False,
        }
        self.ple_attrs = {
            "layer_ids": {layer_id - 1 for layer_id in config.ple_layer_ids},
            "embed_dim": config.ple_embed_dim,
            "conv_kernel_size": config.ple_conv_kernel_size,
            "conv_dilation": config.ngram_size,
            "ngram_size": config.ngram_size,
            "token_pad_id": self.get_engram_pad_id(config.eos_token_id),
            "heads_per_ngram": config.heads_per_ngram,
            "external": extra_options.get("external_engram", False),
        }
        self.rope_attrs["cast"]["use_fp32"] = False
        self.attention_attrs["output_gate_type"] = config.output_gate_type or config.hidden_act
        self.sparse_indexer_attrs = {
            "num_heads": config.indexer_n_heads,
            "kv_heads": config.indexer_kv_heads,
            "head_dim": config.indexer_head_dim,
            "token_budget": config.indexer_budget,
            "compress_ratio": config.indexer_compress_ratio,
            "fixed_cache": not self.use_paged_attention and self.ep == "cuda",
        }
        if extra_options.get("kv_cache_quant_scheme", "none") != "none":
            raise ValueError(
                "Qwen3.8 sparse attention does not support quantized KV caches; use kv_cache_quant_scheme=none."
            )
        self.make_sparse_rotary_init(not text_only)
        if self.ple_attrs["external"]:
            self.input_names["engram_embeddings"] = "engram_embeddings"
            self.input_types["engram_embeddings"] = self.io_dtype
            self.input_shapes["engram_embeddings"] = self.make_hidden_state_shape(last_dim=self.ple_attrs["embed_dim"])

        qsa_layers = {
            layer_id: f"past.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "indexed_attention"
        }
        qsa_outputs = {
            layer_id: f"present.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "indexed_attention"
        }
        self.input_names["past_key_values.key"] = self.make_cache_names(
            ["indexed_attention"], "past_key_values.key"
        )
        self.input_names["past_key_values.value"] = self.make_cache_names(
            ["indexed_attention"], "past_key_values.value"
        )
        self.output_names["present.key"] = self.make_cache_names(["indexed_attention"], "present.key")
        self.output_names["present.value"] = self.make_cache_names(["indexed_attention"], "present.value")
        self.input_names["past.indexer"] = qsa_layers
        self.input_types["past.indexer"] = self.io_dtype
        self.output_names["present.indexer"] = qsa_outputs
        self.output_types["present.indexer"] = self.io_dtype
        if self.use_paged_attention:
            self.sparse_indexer_attrs["state_capacity"] = (
                self.context_length + self.sparse_indexer_attrs["compress_ratio"] - 1
            ) // self.sparse_indexer_attrs["compress_ratio"]
            indexer_state_shape = [
                "batch_size", self.sparse_indexer_attrs["state_capacity"], self.sparse_indexer_attrs["head_dim"]
            ]
            indexer_buffer_shape = [
                "batch_size",
                2 * self.sparse_indexer_attrs["compress_ratio"] - 1,
                self.sparse_indexer_attrs["head_dim"],
            ]
            indexer_lengths_shape = ["batch_size", 2]
            self.input_shapes["past.indexer"] = indexer_state_shape
            self.output_shapes["present.indexer"] = indexer_state_shape
            self.input_names["past.indexer_kv_buffer"] = {
                layer_id: f"past.{layer_id}.indexer_kv_buffer" for layer_id in qsa_layers
            }
            self.input_types["past.indexer_kv_buffer"] = self.io_dtype
            self.input_shapes["past.indexer_kv_buffer"] = indexer_buffer_shape
            self.output_names["present.indexer_kv_buffer"] = {
                layer_id: f"present.{layer_id}.indexer_kv_buffer" for layer_id in qsa_layers
            }
            self.output_types["present.indexer_kv_buffer"] = self.io_dtype
            self.output_shapes["present.indexer_kv_buffer"] = indexer_buffer_shape
            self.input_names["past.indexer_state_lengths"] = {
                layer_id: f"past.{layer_id}.indexer_state_lengths" for layer_id in qsa_layers
            }
            self.input_types["past.indexer_state_lengths"] = ir.DataType.INT32
            self.input_shapes["past.indexer_state_lengths"] = indexer_lengths_shape
            self.output_names["present.indexer_state_lengths"] = {
                layer_id: f"present.{layer_id}.indexer_state_lengths" for layer_id in qsa_layers
            }
            self.output_types["present.indexer_state_lengths"] = ir.DataType.INT32
            self.output_shapes["present.indexer_state_lengths"] = indexer_lengths_shape
            self.input_shapes["attention_metadata"] = [5]
        else:
            self.input_shapes["past.indexer"] = [
                "batch_size", "past_sequence_length", self.sparse_indexer_attrs["head_dim"]
            ]
            self.output_shapes["present.indexer"] = [
                "batch_size", "total_sequence_length", self.sparse_indexer_attrs["head_dim"]
            ]
        if self.sparse_indexer_attrs["fixed_cache"]:
            self.input_names["past_sequence_length"] = "past_sequence_length"
            self.input_types["past_sequence_length"] = ir.DataType.INT32
            self.input_shapes["past_sequence_length"] = [1]

        ple_token_state = {layer_id: f"past.{layer_id}.ple_tokens" for layer_id in self.ple_attrs["layer_ids"]}
        ple_conv_state = {layer_id: f"past.{layer_id}.ple_conv" for layer_id in self.ple_attrs["layer_ids"]}
        present_ple_tokens = {layer_id: f"present.{layer_id}.ple_tokens" for layer_id in self.ple_attrs["layer_ids"]}
        present_ple_conv = {layer_id: f"present.{layer_id}.ple_conv" for layer_id in self.ple_attrs["layer_ids"]}
        self.input_names["past.ple_tokens"] = ple_token_state
        self.input_types["past.ple_tokens"] = ir.DataType.INT64
        self.input_shapes["past.ple_tokens"] = ["batch_size", self.ple_attrs["ngram_size"] - 1]
        self.input_names["past.ple_conv"] = ple_conv_state
        self.input_types["past.ple_conv"] = self.io_dtype
        ple_conv_state_length = self.ple_attrs["conv_dilation"] * (self.ple_attrs["conv_kernel_size"] - 1)
        self.input_shapes["past.ple_conv"] = (
            ["batch_size", self.hyper_connection_attrs["hidden_size"], ple_conv_state_length]
            if self.use_paged_attention
            else [
                *self.context_length_attrs["state_window_dims"],
                "batch_size",
                ple_conv_state_length,
                self.hyper_connection_attrs["hidden_size"],
            ]
        )
        self.output_names["present.ple_tokens"] = present_ple_tokens
        self.output_types["present.ple_tokens"] = ir.DataType.INT64
        self.output_shapes["present.ple_tokens"] = ["batch_size", self.ple_attrs["ngram_size"] - 1]
        self.output_names["present.ple_conv"] = present_ple_conv
        self.output_types["present.ple_conv"] = self.io_dtype
        self.output_shapes["present.ple_conv"] = self.input_shapes["past.ple_conv"]

        state_update_capacity = (
            int(extra_options.get("state_update_capacity", self.context_length_attrs["state_update_capacity"]))
            if self.use_paged_attention else 0
        )
        if state_update_capacity and (self.ple_attrs["layer_ids"] or qsa_layers):
            self.context_length_attrs["state_update_capacity"] = state_update_capacity
            self.input_names["state_update.capture_count"] = "state_update_capture_count"
            self.input_types["state_update.capture_count"] = ir.DataType.INT32
            self.input_shapes["state_update.capture_count"] = ["batch_size"]
            self.input_names["state_update.active"] = "state_update_active"
            self.input_types["state_update.active"] = ir.DataType.INT32
            self.input_shapes["state_update.active"] = [1]
            self.output_names["state_update.ple_tokens"] = {
                layer_id: f"state_update.{layer_id}.ple_tokens" for layer_id in self.ple_attrs["layer_ids"]
            }
            self.output_types["state_update.ple_tokens"] = ir.DataType.INT64
            self.output_shapes["state_update.ple_tokens"] = [
                "batch_size",
                state_update_capacity,
                self.ple_attrs["ngram_size"] - 1,
            ]
            self.output_names["state_update.ple_conv_value"] = {
                layer_id: f"state_update.{layer_id}.ple_conv_value" for layer_id in self.ple_attrs["layer_ids"]
            }
            self.output_types["state_update.ple_conv_value"] = self.io_dtype
            self.output_shapes["state_update.ple_conv_value"] = [
                "batch_size",
                state_update_capacity,
                self.hyper_connection_attrs["hidden_size"],
            ]
            self.output_names["state_update.indexer"] = {
                layer_id: f"state_update.{layer_id}.indexer" for layer_id in qsa_layers
            }
            self.output_types["state_update.indexer"] = self.io_dtype
            self.output_shapes["state_update.indexer"] = [
                "batch_size",
                state_update_capacity,
                self.sparse_indexer_attrs["head_dim"],
            ]

        self.input_names["input_ids"] = "input_ids"
        self.input_types["input_ids"] = ir.DataType.INT64
        self.input_shapes["input_ids"] = (
            ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        )

    def make_sparse_rotary_init(self, multimodal):
        self.sparse_rotary_attrs = {"multimodal": multimodal}
        if multimodal:
            self.input_names["rotary_position_ids"] = "rotary_position_ids"
            self.input_types["rotary_position_ids"] = ir.DataType.INT64
            self.input_shapes["rotary_position_ids"] = [3, "batch_size", "rotary_sequence_length"]

    def update_genai_config(self, genai_config):
        super().update_genai_config(genai_config)
        decoder = genai_config["model"]["decoder"]
        if "rotary_position_ids" in self.input_names:
            decoder["inputs"]["rotary_position_ids"] = self.input_names["rotary_position_ids"]
        decoder["inputs"]["past_ple_token_names"] = "past.%d.ple_tokens"
        decoder["inputs"]["past_ple_conv_names"] = "past.%d.ple_conv"
        decoder["inputs"]["past_indexer_names"] = "past.%d.indexer_key"
        decoder["inputs"]["past_indexer_kv_buffer_names"] = "past.%d.indexer_kv_buffer"
        decoder["inputs"]["past_indexer_state_lengths_names"] = "past.%d.indexer_state_lengths"
        if self.sparse_indexer_attrs["fixed_cache"]:
            decoder["inputs"]["past_sequence_length"] = self.input_names["past_sequence_length"]
        decoder["outputs"]["present_ple_token_names"] = "present.%d.ple_tokens"
        decoder["outputs"]["present_ple_conv_names"] = "present.%d.ple_conv"
        decoder["outputs"]["present_indexer_names"] = "present.%d.indexer_key"
        decoder["outputs"]["present_indexer_kv_buffer_names"] = "present.%d.indexer_kv_buffer"
        decoder["outputs"]["present_indexer_state_lengths_names"] = "present.%d.indexer_state_lengths"
        if getattr(self, "context_length_attrs", {}).get("state_window", 0):
            decoder["outputs"]["state_update_indexer_value_names"] = "state_update.%d.indexer_value"
            decoder["outputs"]["state_update_indexer_row_names"] = "state_update.%d.indexer_row"
        decoder["ple_token_pad_id"] = self.ple_attrs["token_pad_id"]
        decoder["indexer_compress_ratio"] = self.sparse_indexer_attrs["compress_ratio"]
        if self.ep != "cpu" and not self.ple_attrs["external"]:
            session_options = decoder["session_options"]
            session_options["session.layer_assignment_settings"] = f"cpu(={self.CPU_EMBEDDING_ANNOTATION})"

    def make_decoder_state_groups(self, inputs, outputs):
        state_groups = super().make_decoder_state_groups(inputs, outputs)
        if not self.use_paged_attention:
            return state_groups

        qsa_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "indexed_attention"
        ]
        if qsa_layers:
            paged_group = next((group for group in state_groups if group["kind"] == "paged_kv"), None)
            if paged_group is None:
                state_groups.insert(0, self.make_paged_key_value_state_group(qsa_layers))
            else:
                paged_group["layer_ids"] = sorted(set(paged_group["layer_ids"] + qsa_layers))

        if self.ple_attrs["layer_ids"]:
            group = {
                "kind": "fixed_ple",
                "layer_ids": sorted(self.ple_attrs["layer_ids"]),
            }
            if self.context_length_attrs["state_update_capacity"]:
                group["state_update"] = {"capacity": self.context_length_attrs["state_update_capacity"]}
            state_groups.append(group)

        indexer_bindings = {
            "past.indexer",
            "past.indexer_kv_buffer",
            "past.indexer_state_lengths",
        }
        if indexer_bindings.issubset(self.input_names):
            if qsa_layers:
                group = {
                    "kind": "fixed_indexer",
                    "layer_ids": qsa_layers,
                }
                if self.context_length_attrs["state_update_capacity"]:
                    group["state_update"] = {
                        "capacity": self.context_length_attrs["state_update_capacity"],
                        "compress_ratio": self.sparse_indexer_attrs["compress_ratio"],
                    }
                state_groups.append(group)
        return state_groups

    def make_layer(self, layer_id, layer):
        if layer_id == 0 and self.hyper_connection_attrs["tile_first_hidden_state"]:
            tile_name = "/model/hyper_connection/Tile"
            num_branches = self.hyper_connection_attrs["num_branches"]
            tile_repeats = [1, num_branches] if self.use_paged_attention else [1, 1, num_branches]
            self.make_tile(
                tile_name,
                [self.layernorm_attrs["root_input"], f"/model/constants/INT64/{tile_repeats}"],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"]),
            )
            self.layernorm_attrs["root_input"] = f"{tile_name}/output_0"

        hyper_states = self.layernorm_attrs["root_input"]
        if layer_id in self.ple_attrs["layer_ids"]:
            ple_output = self.make_ple(layer_id, layer.ple, hyper_states)
            ple_add = f"/model/layers.{layer_id}/ple/residual/Add"
            self.make_add(
                ple_add,
                [hyper_states, ple_output],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"]),
            )
            hyper_states = f"{ple_add}/output_0"

        mixed, residual, injection = self.make_hyper_connection_mix(
            layer_id, layer.attn_hyper_connection, hyper_states, "attn"
        )
        attention = self.get_attn_module(layer_id, layer)
        if self.layer_types[layer_id] == "linear_attention":
            self.make_qwen_gated_delta_net(layer_id, attention, mixed)
        else:
            self.make_qwen_sparse_attention(layer_id, attention, mixed)
        hyper_states = self.make_hyper_connection_injection(
            layer_id, self.layernorm_attrs["skip_input"], residual, injection, "attn"
        )

        mixed, residual, injection = self.make_hyper_connection_mix(
            layer_id, layer.mlp_hyper_connection, hyper_states, "mlp"
        )
        moe = self.get_moe_module(layer_id, layer)
        self.make_moe_preprocessing(layer_id, moe, mixed)
        self.make_moe_router(layer_id, moe, mixed)
        moe_output = self.make_moe_subgraph(layer_id, moe, mixed)
        hidden_states_output = (
            self.output_names["hidden_states"]
            if layer_id == self.num_layers - 1
            and self.hyper_connection_attrs["emit_pre_final_hidden_states"]
            and (self.include_hidden_states or self.exclude_lm_head)
            else None
        )
        hyper_states = self.make_hyper_connection_injection(
            layer_id, moe_output, residual, injection, "mlp", output_name=hidden_states_output
        )
        self.layernorm_attrs["root_input"] = hyper_states
        self.layernorm_attrs["skip_input"] = hyper_states

        if layer_id == self.num_layers - 1:
            final_output = self.make_hyper_connection_mix(
                self.num_layers,
                self.get_final_hyper_connection_mixer(),
                hyper_states,
                "final",
                combine=False,
            )
            if (self.include_hidden_states or self.exclude_lm_head) and not self.hyper_connection_attrs["emit_pre_final_hidden_states"]:
                self.make_node(
                    "Identity",
                    inputs=[final_output],
                    outputs=[self.output_names["hidden_states"]],
                    name="/model/final_hidden_states/Identity",
                )
                final_output = self.output_names["hidden_states"]
            self.layernorm_attrs["output_0"] = final_output

    def get_final_hyper_connection_mixer(self):
        return self.weights.model.language_model.hyper_connection_mixer
