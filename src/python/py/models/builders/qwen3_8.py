import copy
import json
import os

import onnx_ir as ir
import torch
from builder_config import serialize_genai_config
from huggingface_hub import hf_hub_download, snapshot_download

from .base import Model
from .expansions.qwen3_8 import Qwen38, Qwen38Graph, Qwen38MTP, Qwen38Vision
from .mtp import MTPModel
from .qwen3_5 import Qwen35MoETextModel


class Qwen4ExpTextModel(Qwen38, Qwen35MoETextModel):
    """Qwen4-Exp decoder builder using external token/vision embeddings."""

    CPU_EMBEDDING_ANNOTATION = "cpu_embedding"


    def is_packed_matmul_supported(self):
        return Model.is_packed_matmul_supported(self)

    def make_attention_init(self, config):
        super().make_attention_init(config)
        self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()


    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = copy.deepcopy(extra_options)
        text_only = extra_options.get("text_only", False)
        extra_options["exclude_embeds"] = not text_only
        extra_options.setdefault("filename", "model.onnx" if text_only else "text.onnx")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.merge_shared_input_projections = bool(extra_options.get("merge_shared_input_projections", False))
        self.merged_shared_expert_layers = set()
        if self.merge_shared_input_projections and onnx_dtype not in {
            ir.DataType.FLOAT,
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            raise ValueError("merge_shared_input_projections requires an fp32, fp16 or bf16 export.")
        if not hasattr(self, "context_length_attrs"):
            self.context_length_attrs = {"state_window": 0, "state_window_dims": []}
        self.use_cpu_embedding_gather = False
        if self.use_paged_attention:
            self.input_names.pop("position_ids", None)
        self.model.metadata_props["qwen4_exp.past_indexer_names"] = "past.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.present_indexer_names"] = "present.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.past_ple_token_names"] = "past.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.present_ple_token_names"] = "present.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.past_ple_conv_names"] = "past.%d.ple_conv"
        self.model.metadata_props["qwen4_exp.present_ple_conv_names"] = "present.%d.ple_conv"

        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hc_count * self.hidden_size
        self.ple_layer_ids = {layer_id - 1 for layer_id in config.ple_layer_ids}
        self.ple_embed_dim = config.ple_embed_dim
        self.ple_conv_kernel_size = config.ple_conv_kernel_size
        self.ple_conv_dilation = config.ngram_size
        self.ngram_size = config.ngram_size
        self.ple_token_pad_id = config.eos_token_id
        self.rope_attrs["cast"]["use_fp32"] = False
        self.heads_per_ngram = config.heads_per_ngram
        self.indexer_num_heads = config.indexer_n_heads
        self.indexer_kv_heads = config.indexer_kv_heads
        self.indexer_head_dim = config.indexer_head_dim
        self.indexer_budget = config.indexer_budget
        self.indexer_compress_ratio = config.indexer_compress_ratio
        self.output_gate_type = config.output_gate_type or config.hidden_act
        self.tile_first_hidden_state = True
        self.emit_pre_final_hidden_states = False
        self.external_engram = extra_options.get("external_engram", False)
        if getattr(self, "external_engram", False):
            self.input_names["engram_embeddings"] = "engram_embeddings"
            self.input_types["engram_embeddings"] = self.io_dtype
            self.input_shapes["engram_embeddings"] = self.make_hidden_state_shape(last_dim=self.ple_embed_dim)

        qsa_layers = {
            layer_id: f"past.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        }
        qsa_outputs = {
            layer_id: f"present.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        }
        self.input_names["past_key_values.key"] = self.make_cache_names(
            ["qwen_sparse_attention"], "past_key_values.key"
        )
        self.input_names["past_key_values.value"] = self.make_cache_names(
            ["qwen_sparse_attention"], "past_key_values.value"
        )
        self.output_names["present.key"] = self.make_cache_names(["qwen_sparse_attention"], "present.key")
        self.output_names["present.value"] = self.make_cache_names(["qwen_sparse_attention"], "present.value")
        self.input_names["past.indexer"] = qsa_layers
        self.input_types["past.indexer"] = self.io_dtype
        self.fixed_indexer_cache = not self.use_paged_attention and self.ep == "cuda"
        self.output_names["present.indexer"] = qsa_outputs
        self.output_types["present.indexer"] = self.io_dtype
        if self.use_paged_attention:
            self.indexer_state_capacity = (
                self.context_length + self.indexer_compress_ratio - 1
            ) // self.indexer_compress_ratio
            indexer_state_shape = ["batch_size", self.indexer_state_capacity, self.indexer_head_dim]
            indexer_buffer_shape = [
                "batch_size",
                2 * self.indexer_compress_ratio - 1,
                self.indexer_head_dim,
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
            self.input_shapes["past.indexer"] = ["batch_size", "past_sequence_length", self.indexer_head_dim]
            self.output_shapes["present.indexer"] = ["batch_size", "total_sequence_length", self.indexer_head_dim]
        if self.fixed_indexer_cache:
            self.input_names["past_sequence_length"] = "past_sequence_length"
            self.input_types["past_sequence_length"] = ir.DataType.INT32
            self.input_shapes["past_sequence_length"] = [1]

        ple_token_state = {layer_id: f"past.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        ple_conv_state = {layer_id: f"past.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        present_ple_tokens = {layer_id: f"present.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        present_ple_conv = {layer_id: f"present.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        self.input_names["past.ple_tokens"] = ple_token_state
        self.input_types["past.ple_tokens"] = ir.DataType.INT64
        self.input_shapes["past.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.input_names["past.ple_conv"] = ple_conv_state
        self.input_types["past.ple_conv"] = self.io_dtype
        ple_conv_state_length = self.ple_conv_dilation * (self.ple_conv_kernel_size - 1)
        self.input_shapes["past.ple_conv"] = (
            ["batch_size", self.hc_hidden_size, ple_conv_state_length]
            if self.use_paged_attention
            else [
                *self.context_length_attrs["state_window_dims"],
                "batch_size",
                ple_conv_state_length,
                self.hc_hidden_size,
            ]
        )
        self.output_names["present.ple_tokens"] = present_ple_tokens
        self.output_types["present.ple_tokens"] = ir.DataType.INT64
        self.output_shapes["present.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.output_names["present.ple_conv"] = present_ple_conv
        self.output_types["present.ple_conv"] = self.io_dtype
        self.output_shapes["present.ple_conv"] = self.input_shapes["past.ple_conv"]

        state_update_capacity = int(extra_options.get("state_update_capacity", 0)) if self.use_paged_attention else 0
        if state_update_capacity and (self.ple_layer_ids or qsa_layers):
            self.context_length_attrs["state_update_capacity"] = state_update_capacity
            self.input_names["state_update.capture_count"] = "state_update_capture_count"
            self.input_types["state_update.capture_count"] = ir.DataType.INT32
            self.input_shapes["state_update.capture_count"] = ["batch_size"]
            self.input_names["state_update.active"] = "state_update_active"
            self.input_types["state_update.active"] = ir.DataType.INT32
            self.input_shapes["state_update.active"] = [1]
            self.output_names["state_update.ple_tokens"] = {
                layer_id: f"state_update.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids
            }
            self.output_types["state_update.ple_tokens"] = ir.DataType.INT64
            self.output_shapes["state_update.ple_tokens"] = [
                "batch_size",
                state_update_capacity,
                self.ngram_size - 1,
            ]
            self.output_names["state_update.ple_conv_value"] = {
                layer_id: f"state_update.{layer_id}.ple_conv_value" for layer_id in self.ple_layer_ids
            }
            self.output_types["state_update.ple_conv_value"] = self.io_dtype
            self.output_shapes["state_update.ple_conv_value"] = [
                "batch_size",
                state_update_capacity,
                self.hc_hidden_size,
            ]
            self.output_names["state_update.indexer"] = {
                layer_id: f"state_update.{layer_id}.indexer" for layer_id in qsa_layers
            }
            self.output_types["state_update.indexer"] = self.io_dtype
            self.output_shapes["state_update.indexer"] = [
                "batch_size",
                state_update_capacity,
                self.indexer_head_dim,
            ]

        self.input_names["input_ids"] = "input_ids"
        self.input_types["input_ids"] = ir.DataType.INT64
        self.input_shapes["input_ids"] = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]

    @staticmethod
    def prepare_engram_embedding(table):
        weight = table.weight.detach().cpu()
        scale = getattr(table, "weight_scale", None)
        if weight.dtype == torch.float8_e4m3fn:
            if scale is None:
                scale = torch.ones(1, dtype=torch.float32)
            else:
                scale = torch.as_tensor(scale).detach().cpu()
            if scale.numel() != 1:
                raise ValueError(f"Engram FP8 weight scale must be scalar, got shape {tuple(scale.shape)}.")
            return weight, scale

        if not weight.is_floating_point():
            raise ValueError(f"Engram embedding weight must be floating point, got {weight.dtype}.")
        max_abs = weight.abs().max().float()
        if not torch.isfinite(max_abs):
            raise ValueError("Engram embedding weight contains non-finite values.")
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = max_abs / fp8_max if max_abs > 0 else torch.ones((), dtype=torch.float32)
        quantized_weight = (weight / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        return quantized_weight, scale.reshape(1)

    def update_genai_config(self, genai_config):
        super().update_genai_config(genai_config)
        decoder = genai_config["model"]["decoder"]
        decoder["inputs"]["past_ple_token_names"] = "past.%d.ple_tokens"
        decoder["inputs"]["past_ple_conv_names"] = "past.%d.ple_conv"
        decoder["inputs"]["past_indexer_names"] = "past.%d.indexer_key"
        decoder["inputs"]["past_indexer_kv_buffer_names"] = "past.%d.indexer_kv_buffer"
        decoder["inputs"]["past_indexer_state_lengths_names"] = "past.%d.indexer_state_lengths"
        if getattr(self, "fixed_indexer_cache", False):
            decoder["inputs"]["past_sequence_length"] = self.input_names["past_sequence_length"]
        decoder["outputs"]["present_ple_token_names"] = "present.%d.ple_tokens"
        decoder["outputs"]["present_ple_conv_names"] = "present.%d.ple_conv"
        decoder["outputs"]["present_indexer_names"] = "present.%d.indexer_key"
        decoder["outputs"]["present_indexer_kv_buffer_names"] = "present.%d.indexer_kv_buffer"
        decoder["outputs"]["present_indexer_state_lengths_names"] = "present.%d.indexer_state_lengths"
        if getattr(self, "context_length_attrs", {}).get("state_window", 0):
            decoder["outputs"]["state_update_indexer_value_names"] = (
                "state_update.%d.indexer_value"
            )
            decoder["outputs"]["state_update_indexer_row_names"] = (
                "state_update.%d.indexer_row"
            )
        decoder["ple_token_pad_id"] = self.ple_token_pad_id
        if self.ep != "cpu" and not getattr(self, "external_engram", False):
            session_options = decoder["session_options"]
            session_options["session.layer_assignment_settings"] = (
                f"cpu(={self.CPU_EMBEDDING_ANNOTATION})"
            )

    def make_decoder_state_groups(self, inputs, outputs):
        state_groups = super().make_decoder_state_groups(inputs, outputs)
        if not self.use_paged_attention:
            return state_groups

        qsa_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        ]
        if qsa_layers:
            paged_group = next((group for group in state_groups if group["kind"] == "paged_kv"), None)
            if paged_group is None:
                state_groups.insert(0, self.make_paged_key_value_state_group(qsa_layers))
            else:
                paged_group["layer_ids"] = sorted(set(paged_group["layer_ids"] + qsa_layers))

        if self.ple_layer_ids:
            group = {
                "kind": "fixed_ple",
                "layer_ids": sorted(self.ple_layer_ids),
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
                        "compress_ratio": self.indexer_compress_ratio,
                    }
                state_groups.append(group)
        return state_groups


    def make_layer(self, layer_id, layer):
        if layer_id == 0 and self.tile_first_hidden_state:
            tile_name = "/model/hyper_connection/Tile"
            tile_repeats = [1, self.hc_count] if self.use_paged_attention else [1, 1, self.hc_count]
            self.make_tile(
                tile_name,
                [self.layernorm_attrs["root_input"], f"/model/constants/INT64/{tile_repeats}"],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hc_hidden_size),
            )
            self.layernorm_attrs["root_input"] = f"{tile_name}/output_0"

        hyper_states = self.layernorm_attrs["root_input"]
        if layer_id in self.ple_layer_ids:
            ple_output = self.make_ple(layer_id, layer.ple, hyper_states)
            ple_add = f"/model/layers.{layer_id}/ple/residual/Add"
            self.make_add(
                ple_add,
                [hyper_states, ple_output],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hc_hidden_size),
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
            if layer_id == self.num_layers - 1 and self.emit_pre_final_hidden_states
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
            if (self.include_hidden_states or self.exclude_lm_head) and not self.emit_pre_final_hidden_states:
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


class _Qwen4ExpGraphModel(Qwen38Graph, Model):
    def __init__(self, io_dtype, filename, graph_name):
        self.io_dtype = ir.DataType(io_dtype)
        self.filename = filename
        self.graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22}, name=graph_name)
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

    def save_model(self, output_dir):
        ir.save(
            self.model,
            os.path.join(output_dir, self.filename),
            external_data=f"{self.filename}.data",
            size_threshold_bytes=0,
        )


class Qwen4ExpEmbeddingModel(_Qwen4ExpGraphModel):
    def __init__(self, config, embedding_weight, io_dtype):
        super().__init__(io_dtype, "embedding.onnx", "qwen4_exp_embedding")
        hidden_size = embedding_weight.shape[1]
        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        image_features = self.make_value("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        inputs_embeds = self.make_value(
            "inputs_embeds", self.io_dtype, ["batch_size", "sequence_length", hidden_size]
        )
        self.graph.inputs.extend([input_ids, image_features])
        self.graph.outputs.append(inputs_embeds)

        weight_name = "model.embed_tokens.weight"
        self.make_initializer(embedding_weight, weight_name, to=self.io_dtype)
        image_token = "image_token_id"
        video_token = "video_token_id"
        self.make_initializer(torch.tensor(config.image_token_id, dtype=torch.int64), image_token)
        self.make_initializer(torch.tensor(config.video_token_id, dtype=torch.int64), video_token)
        gathered = "/model/embed_tokens/Gather/output_0"
        self.make_node(
            "Gather", [weight_name, "input_ids"], [gathered], name="/model/embed_tokens/Gather", axis=0
        )
        self.make_value(gathered, self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        image_mask = self.make_binary(
            "Equal",
            "/model/image_mask/Equal",
            ["input_ids", image_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        video_mask = self.make_binary(
            "Equal",
            "/model/video_mask/Equal",
            ["input_ids", video_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        multimodal_mask = self.make_binary(
            "Or",
            "/model/multimodal_mask/Or",
            [image_mask, video_mask],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        indices = "/model/multimodal_indices/NonZero/output_0"
        self.make_node("NonZero", [multimodal_mask], [indices], name="/model/multimodal_indices/NonZero")
        self.make_value(indices, ir.DataType.INT64, [2, "num_image_tokens"])
        scatter_indices = "/model/multimodal_indices/Transpose/output_0"
        self.make_node(
            "Transpose",
            [indices],
            [scatter_indices],
            name="/model/multimodal_indices/Transpose",
            perm=[1, 0],
        )
        self.make_value(scatter_indices, ir.DataType.INT64, ["num_image_tokens", 2])
        self.make_node(
            "ScatterND",
            [gathered, scatter_indices, "image_features"],
            ["inputs_embeds"],
            name="/model/merge_embeddings/ScatterND",
        )


class Qwen4ExpEngramModel(_Qwen4ExpGraphModel):
    def __init__(self, ple, config, io_dtype, scale_dtype=None):
        super().__init__(io_dtype, "engram.onnx", "qwen4_exp_engram")
        self.graph.opset_imports["com.microsoft"] = 1
        embedding = ple.ple_embedding
        ngram_size = config.ngram_size
        heads_per_ngram = config.heads_per_ngram
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_embed_dim = config.ple_embed_dim
        head_dim = ple_embed_dim // ngram_heads
        ple_layer_id = config.ple_layer_ids[0] - 1

        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        past_tokens = self.make_value(
            f"past.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        engram_embeddings = self.make_value(
            "engram_embeddings", self.io_dtype, ["batch_size", "sequence_length", ple_embed_dim]
        )
        present_tokens = self.make_value(
            f"present.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        self.graph.inputs.extend([input_ids, past_tokens])
        self.graph.outputs.extend([engram_embeddings, present_tokens])

        multipliers = "model.ple.layer_multipliers"
        vocab_sizes = "model.ple.head_vocab_sizes"
        offsets = "model.ple.head_offsets"
        eos = "model.ple.eos_token_id"
        self.make_initializer(embedding.layer_multipliers, multipliers)
        self.make_initializer(embedding.ngram_heads_vocab_sizes, vocab_sizes)
        self.make_initializer(embedding.ngram_heads_offsets, offsets)
        self.make_initializer(torch.tensor(embedding.eos_token_id, dtype=torch.int64), eos)

        ngram_ids = "/model/ple/NGramHashMapping/output_0"
        self.make_node(
            "NGramHashMapping",
            inputs=[input_ids.name, multipliers, vocab_sizes, past_tokens.name, offsets, eos],
            outputs=[ngram_ids, present_tokens.name],
            name="/model/ple/NGramHashMapping",
            domain="com.microsoft",
            max_ngram_size=ngram_size,
            n_head_per_ngram=heads_per_ngram,
            pad_id=embedding.eos_token_id,
            reset_on_eos=1,
        )
        self.make_value(
            ngram_ids,
            ir.DataType.INT64,
            ["batch_size", "sequence_length", ngram_heads],
        )

        table_name = "model.ple.ngram_embedding.weight"
        quantized_weight, weight_scale = Qwen4ExpTextModel.prepare_engram_embedding(
            embedding.ngram_embedding
        )
        self.make_initializer(quantized_weight, table_name)
        self.make_initializer(
            weight_scale.reshape(1, 1),
            "model.ple.ngram_embedding.weight_scale",
            to=scale_dtype,
        )
        scale_dtype = self.graph.initializers["model.ple.ngram_embedding.weight_scale"].dtype
        gathered = "/model/ple/ngram_embedding/GatherBlockQuantized/output_0"
        self.make_node(
            "GatherBlockQuantized",
            inputs=[table_name, ngram_ids, "model.ple.ngram_embedding.weight_scale"],
            outputs=[gathered],
            name="/model/ple/ngram_embedding/GatherBlockQuantized",
            domain="com.microsoft",
            metadata_props={"layer_ann": Qwen4ExpTextModel.CPU_EMBEDDING_ANNOTATION},
            gather_axis=0,
            quantize_axis=1,
            block_size=0,
        )
        self.make_value(
            gathered,
            scale_dtype,
            ["batch_size", "sequence_length", ngram_heads, head_dim],
        )
        if scale_dtype != self.io_dtype:
            cast_name = "/model/ple/ngram_embedding/Cast"
            self.make_cast(
                cast_name, gathered, self.io_dtype,
                ["batch_size", "sequence_length", ngram_heads, head_dim],
            )
            gathered = f"{cast_name}/output_0"
        self.make_reshape(
            "/model/ple/ngram_embedding/Reshape",
            [gathered, f"/model/constants/INT64/[0, 0, {ple_embed_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", ple_embed_dim],
            output_name=engram_embeddings.name,
        )

    def save_model(self, output_dir):
        ir.save(
            self.model,
            os.path.join(output_dir, self.filename),
            external_data="engram.onnx.data",
            size_threshold_bytes=0,
        )


class Qwen4ExpVisionModel(Qwen38Vision, _Qwen4ExpGraphModel):
    def __init__(self, config, visual, io_dtype):
        super().__init__(io_dtype, "vision.onnx", "qwen4_exp_vision")
        self.config = config
        self.visual = visual
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.merge_size = config.spatial_merge_size
        patch_size = config.patch_size[0] if isinstance(config.patch_size, (list, tuple)) else config.patch_size
        temporal_size = (
            config.temporal_patch_size[0]
            if isinstance(config.temporal_patch_size, (list, tuple))
            else config.temporal_patch_size
        )
        self.patch_dim = config.in_channels * temporal_size * patch_size * patch_size
        self.make_model()


    def make_model(self):
        pixel_values = self.make_value("pixel_values", self.io_dtype, ["num_patches", self.patch_dim])
        image_grid = self.make_value("image_grid_thw", ir.DataType.INT64, [1, 3])
        image_features = self.make_value("image_features", self.io_dtype, ["num_image_tokens", self.config.out_hidden_size])
        self.graph.inputs.extend([pixel_values, image_grid])
        self.graph.outputs.append(image_features)

        shape = "/vision/pixel_values/Shape/output_0"
        self.make_shape("/vision/pixel_values/Shape", "pixel_values", [2])
        self.make_gather(
            "/vision/pixel_values/num_patches/Gather",
            [shape, "/model/constants/INT64/0"],
            ir.DataType.INT64,
            [],
            axis=0,
        )
        num_patches = "/vision/pixel_values/num_patches/Gather/output_0"
        _, height, width = self.make_grid_values()
        row, col = self.make_patch_positions(num_patches, height, width)

        patch_weight = self.visual.patch_embed.proj.weight.reshape(self.hidden_size, -1).T
        patch_weight_name = "visual.patch_embed.proj.weight"
        patch_bias_name = "visual.patch_embed.proj.bias"
        self.make_initializer(patch_weight, patch_weight_name, to=self.io_dtype)
        self.make_initializer(self.visual.patch_embed.proj.bias, patch_bias_name, to=self.io_dtype)
        patch_matmul = self.make_binary(
            "MatMul",
            "/visual/patch_embed/proj/MatMul",
            ["pixel_values", patch_weight_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/proj/Add",
            [patch_matmul, patch_bias_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        position_embeddings = self.make_position_embeddings(row, col, height, width)
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/add_position/Add",
            [hidden_states, position_embeddings],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        cos, sin = self.make_rotary_embeddings(row, col)

        for layer_id, block in enumerate(self.visual.blocks):
            norm1 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm1", block.norm1, hidden_states, ["num_patches", self.hidden_size]
            )
            attention = self.make_attention(layer_id, block.attn, norm1, cos, sin)
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/attn/residual/Add",
                [hidden_states, attention],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )
            norm2 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm2", block.norm2, hidden_states, ["num_patches", self.hidden_size]
            )
            fc1 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc1",
                block.mlp.linear_fc1,
                norm2,
                ["num_patches", self.config.intermediate_size],
            )
            activated = self.make_gelu(
                f"/visual/blocks/{layer_id}/mlp/Gelu",
                fc1,
                ["num_patches", self.config.intermediate_size],
                approximate="tanh",
            )
            fc2 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc2",
                block.mlp.linear_fc2,
                activated,
                ["num_patches", self.hidden_size],
            )
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/mlp/residual/Add",
                [hidden_states, fc2],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )

        merger = self.visual.merger
        merged_norm = self.make_layer_norm(
            "visual.merger.norm", merger.norm, hidden_states, ["num_patches", self.hidden_size]
        )
        merged_input = "/visual/merger/Reshape/output_0"
        merged_hidden_size = self.hidden_size * self.merge_size * self.merge_size
        self.make_reshape(
            "/visual/merger/Reshape",
            [merged_norm, f"/model/constants/INT64/[-1, {merged_hidden_size}]"],
            self.io_dtype,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_fc1 = self.make_linear(
            "visual.merger.linear_fc1",
            merger.linear_fc1,
            merged_input,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_activated = self.make_gelu(
            "/visual/merger/Gelu", merger_fc1, ["num_image_tokens", merged_hidden_size]
        )
        self.make_linear(
            "visual.merger.linear_fc2",
            merger.linear_fc2,
            merger_activated,
            ["num_image_tokens", self.config.out_hidden_size],
            output="image_features",
        )


class Qwen4ExpModel(MTPModel):
    """Composite builder that emits Qwen4-Exp vision, embedding, and text graphs."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__()
        self.config = config
        self.extra_options = copy.deepcopy(extra_options)
        self.text_only = self.extra_options.get("text_only", False)
        decoder_options = self.make_mtp_init(config, self.extra_options)
        decoder_options["external_engram"] = True
        self.decoder = Qwen4ExpTextModel(
            copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options
        )
        self.decoder.model_type = "qwen4_exp_text" if self.text_only else "qwen3_5"
        self.mtp = None
        if self.mtp_attrs["build"]:
            self.decoder.emit_pre_final_hidden_states = True
            self.decoder.output_shapes["hidden_states"] = self.decoder.make_hidden_state_shape(
                last_dim=self.decoder.hc_hidden_size
            )
            self.make_mtp_model(config, io_dtype, onnx_dtype, ep, cache_dir, decoder_options)
        self.input_path = None

        text_config = config.text_config
        self.bos_token_id = text_config.bos_token_id
        self.eos_token_id = text_config.eos_token_id
        self.pad_token_id = text_config.pad_token_id
        self.vocab_size = self.decoder.vocab_size
        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length
        self.exclude_embeds = self.decoder.exclude_embeds
        self.model_type = self.decoder.model_type

    def make_mtp_init(self, config, extra_options):
        decoder_options = super().make_mtp_init(config, extra_options)
        num_mtp_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0) or 0
        self.mtp_attrs["build"] = num_mtp_layers > 0 and not extra_options.get("exclude_mtp", False)
        self.mtp_attrs["shared_initializer_names"] = {"model.embed_tokens.weight"}
        self.mtp_attrs["shared_initializer_prefixes"] = ("lm_head.MatMul.",)
        if not self.mtp_attrs["build"]:
            return decoder_options
        if num_mtp_layers != 1:
            raise ValueError(f"Qwen4-Exp MTP export requires exactly one MTP layer, got {num_mtp_layers}.")
        incompatible_options = [
            option for option in ("exclude_lm_head",) if extra_options.get(option, False)
        ]
        if extra_options.get("prune_lm_head", False) and not extra_options.get("use_paged_attention", False):
            incompatible_options.append("prune_lm_head")
        if incompatible_options:
            raise ValueError("Qwen4-Exp MTP export cannot be combined with " + ", ".join(incompatible_options) + ".")
        decoder_options["include_hidden_states"] = True
        return decoder_options

    def make_mtp_model(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.mtp_attrs["io_dtype"] = io_dtype
        self.mtp_attrs["onnx_dtype"] = onnx_dtype
        self.mtp_attrs["extra_options"] = copy.deepcopy(extra_options)
        if getattr(self.decoder, "quant_type", None) in {"modelopt", "compressed-tensors", "fp8"} and "mtp_quant_config" not in extra_options:
            mtp_quant_config = copy.deepcopy(self.decoder.quant_config)
            mtp_quant_config.weights.type = "none"
            mtp_quant_config.moe.type = "none"
            self.mtp_attrs["extra_options"]["_quant_config"] = mtp_quant_config
            self.mtp_attrs["onnx_dtype"] = io_dtype
        self.resolve_mtp_model_config(extra_options)
        mtp_options = self.mtp_attrs["extra_options"]
        for option in (
            "use_paged_attention", "prune_lm_head", "paged_block_size", "enable_webgpu_graph",
            "gpu_utilization_factor", "max_batch_size", "state_update_capacity",
            "max_draft_tokens", "indexshare_mtp",
        ):
            if option in extra_options:
                mtp_options[option] = copy.deepcopy(extra_options[option])
        mtp_options["text_only"] = True
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        mtp_options.pop("external_engram", None)
        mtp_options.pop("merge_shared_input_projections", None)
        self.mtp = Qwen4ExpMTPTextModel(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
        )

    def make_model(self, input_path):
        if self.decoder.quant_type == "fp8" and not os.path.isdir(input_path):
            input_path = snapshot_download(
                repo_id=self.decoder.model_name_or_path,
                cache_dir=self.decoder.cache_dir,
                token=self.decoder.hf_token,
                allow_patterns=["*.json", "*.safetensors"],
            )
        self.input_path = input_path
        self.decoder.make_model(input_path)
        if self.mtp is not None:
            print("Building Qwen4-Exp MTP (multi-token prediction) head -> mtp.onnx")
            self.mtp.make_model(input_path)

    def save_model(self, output_dir):
        table_name = "model.ple.ngram_embedding.weight"
        if self.input_path is None:
            raise RuntimeError("make_model must be called before save_model.")
        weights = self.decoder.load_weights(self.input_path)
        language_model = weights.model.language_model
        if len(self.decoder.ple_layer_ids) != 1:
            raise ValueError(
                f"Qwen4-Exp Engram export requires exactly one PLE layer, got {sorted(self.decoder.ple_layer_ids)}."
            )
        ple_layer_id = next(iter(self.decoder.ple_layer_ids))
        engram_model = Qwen4ExpEngramModel(
            language_model.layers[ple_layer_id].ple,
            self.config.text_config,
            self.decoder.io_dtype,
            scale_dtype=ir.DataType.FLOAT16 if self.decoder.io_dtype == ir.DataType.FLOAT16 else ir.DataType.FLOAT,
        )
        engram_model.save_model(output_dir)
        table = ir.load(os.path.join(output_dir, engram_model.filename)).graph.initializers[table_name].const_value
        for component in (self.decoder, self.mtp):
            if component is not None and table_name in getattr(component, "external_data_files", {}):
                component.external_data_tensors = {table_name: table}

        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
            if self.extra_options.get("indexshare_mtp", self.mtp.use_paged_attention):
                self.mtp_attrs["index_share"] = self.export_indexshare_graphs(
                    output_dir, self.mtp.filename, int(self.extra_options.get("indexshare_max_draft_tokens", 7))
                )
        if self.text_only:
            return
        embedding_model = Qwen4ExpEmbeddingModel(
            self.config, language_model.embed_tokens.weight.detach().cpu(), self.decoder.io_dtype
        )
        embedding_model.save_model(output_dir)
        vision_config = self.config.vision_config
        if vision_config.out_hidden_size != self.decoder.hidden_size:
            raise ValueError(
                "Qwen4-Exp vision output size must match the text embedding size: "
                f"{vision_config.out_hidden_size} != {self.decoder.hidden_size}."
            )
        visual = weights.load_visual() if hasattr(weights, "load_visual") else weights.model.visual
        vision_model = Qwen4ExpVisionModel(vision_config, visual, self.decoder.io_dtype)
        vision_model.save_model(output_dir)
        del weights

    def configure_paged_sessions(self, genai_config):
        if self.decoder.ep not in {"cuda", "webgpu"} or not getattr(self.decoder, "use_paged_attention", False):
            return
        model_config = genai_config["model"]
        decoder_options = model_config["decoder"].get("session_options", {})
        ep = self.decoder.ep
        inherited_provider_options = next(
            (provider[ep] for provider in decoder_options.get("provider_options", []) if ep in provider),
            {},
        )
        for section in ("decoder", "mtp"):
            if section not in model_config:
                continue
            options = model_config[section].setdefault("session_options", {})
            providers = options.setdefault("provider_options", [])
            provider_options = next((provider[ep] for provider in providers if ep in provider), None)
            if provider_options is None:
                provider_options = {}
                providers.append({ep: provider_options})
            provider_options.update(copy.deepcopy(inherited_provider_options))
            options["session.use_device_allocator_for_initializers"] = "1"
            if ep != "cuda":
                continue
            options.update({
                "ep.cuda.fpa_intb_gemm": "1",
                "ep.cuda.qmoe_skip_nvfp4_gemv_profiling": "1",
                "ep.cuda.sparse_paged_attention_grouped": "1",
                "ep.cuda.sparse_paged_attention_grouped_decode": "0",
                "ep.cuda.sparse_paged_attention_grouped_decode_splits": "0",
                "ep.cuda.sparse_paged_attention_grouped_tile_size": "8",
                "ep.cuda.sparse_paged_attention_grouped_vectorized": "1",
                "ep.cuda.sparse_paged_attention_warp_reduction": "0",
                "session.use_device_allocator_for_initializers": "1",
            })

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(config.text_config, extra_kwargs, out_dir)
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        model_config = genai_config["model"]
        if (
            self.decoder.ep == "cuda"
            and self.decoder.moe_attrs.get("quant_type") == "nvfp4"
            and self.decoder.moe_attrs.get("weights_prepacked", -1) != 1
        ):
            model_config["decoder"].setdefault("session_options", {})["session.disable_prepacking"] = "1"
        decoder_inputs = model_config["decoder"]["inputs"]
        decoder_inputs["engram_embeddings"] = "engram_embeddings"
        if not self.text_only:
            model_config["type"] = "qwen3_5"
            model_config["image_token_id"] = config.image_token_id
            model_config["vision_start_token_id"] = config.vision_start_token_id
            decoder_inputs["inputs_embeds"] = "inputs_embeds"
            decoder_inputs["input_ids"] = "input_ids"
            model_config["embedding"] = {
                "filename": "embedding.onnx",
                "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            }
            model_config["vision"] = {
                "filename": "vision.onnx",
                "spatial_merge_size": config.vision_config.spatial_merge_size,
                "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
                "outputs": {"image_features": "image_features"},
            }
        ple_layer_id = config.text_config.ple_layer_ids[0] - 1
        engram_session_options = {
            "provider_options": [{"cuda" if self.decoder.ep == "cuda" else "cpu": {}}],
        }
        if self.decoder.ep == "cuda":
            engram_session_options["session.layer_assignment_settings"] = (
                f"cpu(={Qwen4ExpTextModel.CPU_EMBEDDING_ANNOTATION})"
            )
        model_config["engram"] = {
            "filename": "engram.onnx",
            "cache_capacity": 4096,
            "session_options": engram_session_options,
            "inputs": {
                "input_ids": "input_ids",
                "past_ple_token_names": f"past.{ple_layer_id}.ple_tokens",
            },
            "outputs": {
                "embeddings": "engram_embeddings",
                "present_ple_token_names": f"present.{ple_layer_id}.ple_tokens",
            },
        }
        self.configure_paged_sessions(genai_config)
        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
        if self.mtp is not None:
            self.add_mtp_to_genai_config(out_dir)

    def add_mtp_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder_outputs = genai_config["model"]["decoder"].setdefault("outputs", {})
        decoder_outputs["hidden_states"] = "hidden_states"
        genai_config["model"]["mtp"] = {
            "filename": "mtp.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": self.decoder.num_kv_heads,
            "head_size": self.decoder.head_size,
            "main_hidden_states": "hidden_states",
            "inputs": {
                "input_ids": "input_ids",
                "hidden_states": "hidden_states",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
                "past_indexer_names": "past.%d.indexer_key",
                "past_sequence_length": "past_sequence_length",
            },
            "outputs": {
                "logits": "logits",
                "hidden_states": "hidden_states_out",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
                "present_indexer_names": "present.%d.indexer_key",
            },
            "session_options": {
                name: value
                for name, value in genai_config["model"]["decoder"].get("session_options", {}).items()
                if name in ("ep.cuda.fpa_intb_gemm", "session.use_device_allocator_for_initializers")
            },
        }
        if self.mtp.use_paged_attention and self.mtp.prune_lm_head:
            genai_config["model"]["mtp"]["inputs"]["logits_indices"] = "logits_indices"
        self.add_shared_initializers_to_genai_config(genai_config)
        if "index_share" in self.mtp_attrs:
            metadata = self.mtp_attrs["index_share"]
            mtp_config = genai_config["model"]["mtp"]
            mtp_config["base_capacity"] = metadata["base_capacity"]
            mtp_config["inputs"]["past_indices"] = "past_indices"
            mtp_config["inputs"]["past_counts"] = "past_counts"
            mtp_config["outputs"]["present_indices"] = metadata["indices_output"]
            mtp_config["outputs"]["present_counts"] = metadata["counts_output"]
            mtp_config["outputs"]["indexer_status"] = metadata["status_output"]
            genai_config.setdefault("speculative", {}).setdefault("max_draft_tokens", metadata["max_draft_tokens"])
        self.configure_paged_sessions(genai_config)
        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)
        preprocessor_path = (
            os.path.join(model_name_or_path, "preprocessor_config.json")
            if os.path.isdir(model_name_or_path)
            else hf_hub_download(
                model_name_or_path,
                "preprocessor_config.json",
                cache_dir=extra_kwargs.get("cache_dir"),
                token=self.hf_token,
            )
        )
        with open(preprocessor_path) as preprocessor_file:
            image_processor = json.load(preprocessor_file)
        processor_config = {
            "processor": {
                "name": "qwen4_exp_image_processor",
                "transforms": [
                    {
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    },
                    {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                    {
                        "operation": {
                            "name": "resize",
                            "type": "Resize",
                            "attrs": {
                                "width": 540,
                                "height": 360,
                                "smart_resize": 1,
                                "min_pixels": image_processor["size"]["shortest_edge"],
                                "max_pixels": image_processor["size"]["longest_edge"],
                                "patch_size": image_processor["patch_size"],
                                "merge_size": image_processor["merge_size"],
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "rescale",
                            "type": "Rescale",
                            "attrs": {"rescale_factor": 1 / 255},
                        }
                    },
                    {
                        "operation": {
                            "name": "normalize",
                            "type": "Normalize",
                            "attrs": {
                                "mean": image_processor["image_mean"],
                                "std": image_processor["image_std"],
                                "qwen3_vl": 1,
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "patch_image",
                            "type": "PatchImage",
                            "attrs": {
                                "patch_size": image_processor["patch_size"],
                                "temporal_patch_size": image_processor["temporal_patch_size"],
                                "merge_size": image_processor["merge_size"],
                            },
                        }
                    },
                ],
            }
        }
        with open(os.path.join(out_dir, "processor_config.json"), "w") as processor_config_file:
            json.dump(processor_config, processor_config_file, indent=4)


class Qwen4ExpMTPTextModel(Qwen38MTP, Qwen4ExpTextModel):
    """Qwen4-Exp one-layer self-speculative MTP head builder."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config = copy.deepcopy(config)
        config.text_config.num_hidden_layers = 1
        config.text_config.layer_types = ["qwen_sparse_attention"]
        config.text_config.ple_layer_ids = []
        config.num_hidden_layers = 1
        config.layer_types = ["qwen_sparse_attention"]

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        extra_options["text_only"] = True
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.tile_first_hidden_state = False
        self.emit_pre_final_hidden_states = True
        self.include_hidden_states = True
        self.output_names["hidden_states"] = "hidden_states_out"
        self.output_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hc_hidden_size)
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hc_hidden_size)

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
            from loaders.qwen import Qwen4ExpMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.qwen import Qwen4ExpMTPModel  # noqa: PLC0415

        self.mtp_weights = Qwen4ExpMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.hf_load_config.text_config,
            preserve_quantization=False,
            load_quantized_model=self.load_weights,
        )
