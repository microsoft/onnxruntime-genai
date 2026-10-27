# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy
import json
import os

import onnx_ir as ir
import torch
from builder_config import serialize_genai_config
from huggingface_hub import hf_hub_download, snapshot_download

from .mtp import MTPModel
from .qwen3_8_embedding import Qwen4ExpEmbeddingModel
from .qwen3_8_engram import Qwen4ExpEngramModel
from .qwen3_8_mtp import Qwen4ExpMTPTextModel
from .qwen3_8_text import Qwen4ExpTextModel
from .qwen3_8_vision import Qwen4ExpVisionModel


class Qwen38Engram:
    """Non-paged Qwen3.8 Engram graph construction."""

    def make_ple(self, layer_id, ple, root_input):
        basename = f"/model/layers.{layer_id}/ple"
        ngram_op_type = "NGramHashMapping"
        ngram_name = f"{basename}/{ngram_op_type}"
        ngram_ids = f"{ngram_name}/output_0"
        embedding = ple.ple_embedding
        token_shape = ["batch_size", "sequence_length"]
        flatten_dims = [0, 0, self.ple_attrs["embed_dim"]]
        if not self.ple_attrs["external"]:
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
            ngram_inputs.extend(
                [self.input_names["past.ple_tokens"][layer_id], offsets, eos]
            )
            ngram_outputs = [
                ngram_ids,
                self.output_names["present.ple_tokens"][layer_id],
            ]
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
            )
            ngram_heads = (self.ple_attrs["ngram_size"] - 1) * self.ple_attrs[
                "heads_per_ngram"
            ]
            self.make_value(
                ngram_ids,
                ir.DataType.INT64,
                ["batch_size", "sequence_length", ngram_heads],
            )
        if self.ple_attrs["external"]:
            engram_embeddings = self.input_names["engram_embeddings"]
            self.make_node(
                "Identity",
                inputs=[self.input_names["past.ple_tokens"][layer_id]],
                outputs=[self.output_names["present.ple_tokens"][layer_id]],
                name=f"{basename}/token_state/Identity",
            )
        else:
            engram_embeddings = self.make_ple_embedding(
                basename, embedding, ngram_ids, ngram_heads, token_shape, flatten_dims
            )
        grouped_dims = [
            0,
            0,
            self.hyper_connection_attrs["num_branches"],
            self.hidden_size,
        ]
        gate_inputs, grouped_shape, key_scale, query_scale, conv_scale = (
            self.make_ple_projections(
                layer_id, ple, root_input, engram_embeddings, token_shape, grouped_dims
            )
        )
        gate_name = f"{basename}/EngramGate"
        gated_value = f"{gate_name}/output_0"
        gated_value_normed = f"{gate_name}/output_1"
        gate_shape = grouped_shape
        self.make_ple_gate(
            gate_name,
            gated_value,
            gated_value_normed,
            gate_inputs,
            key_scale,
            query_scale,
            conv_scale,
            gate_shape,
        )
        self.make_shape(f"{gate_name}/token_shape", self.input_names["input_ids"], [2])
        self.make_gather(
            f"{gate_name}/sequence_length",
            [f"{gate_name}/token_shape/output_0", "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            [1],
            axis=0,
        )
        self.make_node(
            "Neg",
            [f"{gate_name}/sequence_length/output_0"],
            [f"{gate_name}/start"],
            name=f"{gate_name}/start/Neg",
        )
        self.make_value(f"{gate_name}/start", ir.DataType.INT64, [1])
        mask_name = f"{gate_name}/padding/Slice"
        self.make_slice(
            mask_name,
            [
                self.input_names["attention_mask"],
                f"{gate_name}/start",
                "/model/constants/INT64/[9223372036854775807]",
                "/model/constants/INT64/[1]",
            ],
            ir.DataType.INT64,
            ["batch_size", "sequence_length"],
        )
        self.make_cast(
            f"{gate_name}/padding/Cast",
            f"{mask_name}/output_0",
            ir.DataType.BOOL,
            token_shape,
        )
        self.make_unsqueeze(
            f"{gate_name}/padding/Unsqueeze",
            [f"{gate_name}/padding/Cast/output_0", "/model/constants/INT64/[2, 3]"],
            ir.DataType.BOOL,
            [*token_shape, 1, 1],
        )
        for name, value in (("value", gated_value), ("normed", gated_value_normed)):
            self.make_where(
                f"{gate_name}/padding/{name}",
                [
                    f"{gate_name}/padding/Unsqueeze/output_0",
                    value,
                    f"/model/constants/{self.io_dtype}/0",
                ],
                self.io_dtype,
                grouped_shape,
            )
        gated_value = f"{gate_name}/padding/value/output_0"
        gated_value_normed = f"{gate_name}/padding/normed/output_0"
        ple_shape, gated_value_flat, gated_value_normed_flat, conv_weight = (
            self.make_ple_conv_inputs(
                layer_id,
                ple,
                gate_name,
                gated_value,
                gated_value_normed,
                token_shape,
                flatten_dims,
            )
        )
        conv_op_type = "CausalConvWithState"
        conv_name = f"{basename}/{conv_op_type}"
        conv_output = f"{conv_name}/output_0"
        self.make_node(
            "CausalConvWithState",
            inputs=[
                f"{gated_value_normed_flat}/output_0",
                conv_weight,
                "",
                self.input_names["past.ple_conv"][layer_id],
            ],
            outputs=[conv_output, self.output_names["present.ple_conv"][layer_id]],
            name=conv_name,
            domain="com.microsoft",
            ndim=1,
            dilation=self.ple_attrs["conv_dilation"],
            channels_last=1,
            activation="silu",
            state_window=getattr(self, "context_length_attrs", {}).get(
                "state_window", 0
            ),
        )
        self.make_value(conv_output, self.io_dtype, ple_shape)
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [f"{gated_value_flat}/output_0", conv_output],
            self.io_dtype,
            ple_shape,
        )
        return f"{add_name}/output_0"


class Qwen4ExpModel(MTPModel):
    """Composite builder that emits Qwen4-Exp vision, embedding, and text graphs."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__()
        self.config = config
        self.extra_options = copy.deepcopy(extra_options)
        self.text_only = self.extra_options.get("text_only", False)
        decoder_options = self.make_mtp_init(config, self.extra_options)
        decoder_options["external_engram"] = True
        self.decoder = Qwen4ExpTextModel(copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options)
        self.decoder.model_type = "qwen4_exp_text" if self.text_only else "qwen3_5"
        self.mtp = None
        if self.mtp_attrs["build"]:
            self.decoder.hyper_connection_attrs["emit_pre_final_hidden_states"] = True
            self.decoder.output_shapes["hidden_states"] = self.decoder.make_hidden_state_shape(
                last_dim=self.decoder.hyper_connection_attrs["hidden_size"]
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
        self.mtp_attrs["multimodal_rotary"] = not extra_options.get("text_only", False)
        if not self.mtp_attrs["build"]:
            return decoder_options
        if num_mtp_layers != 1:
            raise ValueError(f"Qwen4-Exp MTP export requires exactly one MTP layer, got {num_mtp_layers}.")
        if extra_options.get("use_paged_attention", False) and int(extra_options.get("max_draft_tokens", 7)) > 7:
            raise ValueError("Paged Qwen3.8 main/MTP index sharing supports max_draft_tokens between 1 and 7.")
        incompatible_options = [option for option in ("exclude_lm_head",) if extra_options.get(option, False)]
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
        if (
            getattr(self.decoder, "quant_type", None) in {"modelopt", "compressed-tensors", "fp8"}
            and "mtp_quant_config" not in extra_options
        ):
            mtp_quant_config = copy.deepcopy(self.decoder.quant_config)
            mtp_quant_config.weights.type = "none"
            mtp_quant_config.moe.type = "none"
            self.mtp_attrs["extra_options"]["_quant_config"] = mtp_quant_config
            self.mtp_attrs["onnx_dtype"] = io_dtype
        self.resolve_mtp_model_config(extra_options)
        mtp_options = self.mtp_attrs["extra_options"]
        for option in (
            "use_paged_attention",
            "prune_lm_head",
            "paged_block_size",
            "enable_webgpu_graph",
            "gpu_utilization_factor",
            "max_batch_size",
            "state_update_capacity",
            "max_draft_tokens",
        ):
            if option in extra_options:
                mtp_options[option] = copy.deepcopy(extra_options[option])
        mtp_options["text_only"] = True
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        mtp_options.pop("external_engram", None)
        self.mtp = Qwen4ExpMTPTextModel(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
            multimodal_rotary=self.mtp_attrs.get("multimodal_rotary", False),
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
        if self.input_path is None:
            raise RuntimeError("make_model must be called before save_model.")
        weights = self.decoder.load_weights(self.input_path)
        language_model = weights.model.language_model
        if len(self.decoder.ple_attrs["layer_ids"]) != 1:
            raise ValueError(
                f"Qwen4-Exp Engram export requires exactly one PLE layer, got {sorted(self.decoder.ple_attrs['layer_ids'])}."
            )
        ple_layer_id = next(iter(self.decoder.ple_attrs["layer_ids"]))
        engram_model = Qwen4ExpEngramModel(
            language_model.layers[ple_layer_id].ple,
            self.config.text_config,
            self.decoder.io_dtype,
            scale_dtype=ir.DataType.FLOAT,
        )
        engram_model.save_model(output_dir)
        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
            if self.mtp.use_paged_attention:
                self.mtp_attrs["index_share"] = self.export_indexshare_graphs(
                    output_dir, self.mtp.filename, min(int(self.extra_options.get("max_draft_tokens", 7)), 7)
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
            options.update(
                {
                    "ep.cuda.fpa_intb_gemm": "1",
                    "ep.cuda.qmoe_skip_nvfp4_gemv_profiling": "1",
                    "ep.cuda.sparse_paged_attention_grouped": "1",
                    "ep.cuda.sparse_paged_attention_grouped_tile_size": "8",
                    "ep.cuda.sparse_paged_attention_grouped_vectorized": "1",
                    "session.use_device_allocator_for_initializers": "1",
                }
            )

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
        if self.mtp_attrs.get("multimodal_rotary", False):
            genai_config["model"]["mtp"]["inputs"]["rotary_position_ids"] = "rotary_position_ids"
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
            speculative = genai_config.setdefault("speculative", {})
            requested = speculative.get("max_draft_tokens", metadata["max_draft_tokens"])
            if not 1 <= requested <= metadata["max_draft_tokens"]:
                raise ValueError(
                    f"max_draft_tokens={requested} exceeds the exported IndexShare capacity "
                    f"{metadata['max_draft_tokens']}."
                )
            speculative["max_draft_tokens"] = requested
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
