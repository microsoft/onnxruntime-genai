# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import os
from copy import deepcopy

import numpy as np
import onnx_ir as ir
from builder_config import serialize_genai_config
from loaders.vision import load_bailing_visual
from transformers import PretrainedConfig

from .bailing import BailingMoeV3Model
from .base import Model
from .expansions.standard import Standard
from .qwen3_8.vision import Qwen4ExpVisionModel, save_image_processor


class BailingVLModel(Model):
    @classmethod
    def load_config(cls, model_name_or_path, **kwargs):
        config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path, **kwargs)
        if config_dict.get("architectures") != ["BailingMoeV3VLForConditionalGeneration"]:
            return None
        config = PretrainedConfig.from_dict(config_dict)
        config.name_or_path = model_name_or_path
        return config

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(graph_name="bailing_vl_package")
        self.config = deepcopy(config)
        if ep != "cuda" or not extra_options.get("use_paged_attention", False):
            raise ValueError("Ling Flash-VL requires the CUDA EP with paged attention")
        self.text_only = extra_options.get("text_only", False)
        if bool(extra_options.get("exclude_embeds", False)) == bool(self.text_only):
            raise ValueError("Ling complete vision export requires exclude_embeds=true; text_only requires false")
        if any(extra_options.get(name) for name in ("dflash2_path", "dspark_path", "mtp_quant_config")):
            raise ValueError("Ling Flash-VL supports target-only export, without MTP or block drafters")

        if getattr(config, "text_config", None) is None or (
            not self.text_only and getattr(config, "vision_config", None) is None
        ):
            raise ValueError("Ling Flash-VL requires text configuration and, for complete export, vision configuration")
        for name in ("text_config", "vision_config"):
            nested = getattr(self.config, name, None)
            if isinstance(nested, dict):
                setattr(self.config, name, PretrainedConfig.from_dict(nested))
        text_config = self.config.text_config
        text_config.hidden_act = getattr(text_config, "hidden_act", "silu")
        text_config.name_or_path = config.name_or_path
        text_config.architectures = ["BailingMoeV3ForCausalLM"]
        text_config.mrope_section = deepcopy(config.mrope_section)
        text_config.tie_word_embeddings = getattr(
            config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)
        )
        if hasattr(config, "quantization_config"):
            text_config.quantization_config = deepcopy(config.quantization_config)
        self.decoder = BailingVLTextModel(text_config, io_dtype, onnx_dtype, ep, cache_dir, deepcopy(extra_options))
        if not self.text_only and self.decoder.filename in {"embedding.onnx", "vision.onnx"}:
            raise ValueError("Ling decoder filename conflicts with an emitted companion graph")
        self.decoder.model_type = "bailing_hybrid" if self.text_only else "bailing_moe_v3_vl"
        self.vision = None
        self.vocab_size = self.decoder.vocab_size
        self.exclude_embeds = self.decoder.exclude_embeds
        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length
        self.model_type = self.decoder.model_type

    def make_model(self, input_path):
        self.decoder.make_model(input_path)
        if not self.text_only:
            vision_config = self.config.vision_config
            visual = load_bailing_visual(
                vision_config,
                self.decoder.hidden_size,
                input_path if os.path.isdir(input_path) else self.decoder.model_name_or_path,
                cache_dir=self.decoder.cache_dir,
                token=self.hf_token,
                revision=self.decoder.revision,
            )
            self.vision = BailingVLVisualModel(vision_config, visual, self.decoder.hidden_size, self.decoder.io_dtype)

    def save_model(self, output_dir):
        if not self.text_only and self.vision is None:
            raise RuntimeError("make_model must construct Ling vision before save_model")
        self.decoder.save_model(output_dir)
        if self.vision is not None:
            self.vision.save_model(output_dir)

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(self.decoder.config, extra_kwargs, out_dir)
        if self.text_only:
            return
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path, encoding="utf-8") as handle:
            genai_config = json.load(handle)
        model_config = genai_config["model"]
        model_config["image_token_id"] = config.image_token_id
        model_config["vision_start_token_id"] = config.vision_start_token_id
        vision_config = self.config.vision_config
        model_config["vision"] = {
            "filename": "vision.onnx",
            "spatial_merge_size": vision_config.spatial_merge_size,
            "patch_size": vision_config.patch_size,
            "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
            "outputs": {"image_features": "image_features"},
        }
        model_config.setdefault(
            "embedding",
            {
                "filename": "embedding.onnx",
                "session_options": {"intra_op_num_threads": 1},
                "inputs": {"input_ids": "input_ids"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            },
        )
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(serialize_genai_config(genai_config))

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)
        if not self.text_only:
            save_image_processor(model_name_or_path, extra_kwargs, out_dir, self.hf_token)


class BailingVLTextModel(BailingMoeV3Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if not self.use_paged_attention:
            raise ValueError("Ling vision decoder requires packed paged attention")
        self.rope_attrs["mrope_section"] = list(config.mrope_section)
        self.rope_attrs["op_type"] = "RotaryEmbedding" if extra_options.get("text_only", False) else "MRotaryEmbedding"
        self.rope_attrs["interleaved"] = 1
        if self.rope_attrs["op_type"] == "MRotaryEmbedding":
            self.rope_attrs["position_start"] = -self.original_context_length
            self.rope_attrs["cache_length"] = self.original_context_length + self.context_length
            self.rope_attrs["cast"]["use_fp32"] = True
        self.get_mrope_owners(self.qk_rope_head_dim)

    def get_mrope_owners(self, rotary_dim):
        sections = self.rope_attrs["mrope_section"]
        if (
            rotary_dim <= 0
            or rotary_dim % 2
            or len(sections) != 3
            or any(section < 0 for section in sections)
            or sections[1] != sections[2]
            or sum(sections) * 2 != rotary_dim
        ):
            raise ValueError("Ling MRoPE requires matching H/W sections summing with T to half the rotary dimension")
        temporal, height = sections[:2]
        return np.concatenate([np.tile(np.array([1, 2], dtype=np.int64), height), np.zeros(temporal, dtype=np.int64)])

    def make_mla_rope(self, name, root_input, num_heads, token_shape):
        if self.rope_attrs["op_type"] != "MRotaryEmbedding":
            return super().make_mla_rope(name, root_input, num_heads, token_shape)
        return Standard.make_packed_mrope(
            self, name, root_input, num_heads, self.qk_rope_head_dim, self.get_mrope_owners(self.qk_rope_head_dim)
        )


class BailingVLVisualModel(Qwen4ExpVisionModel):
    def __init__(self, config, visual, text_hidden_size, io_dtype):
        config = deepcopy(config)
        config.out_hidden_size = text_hidden_size
        # FP16 vision arithmetic avoids accumulated BF16 rounding across the full tower.
        compute_dtype = ir.DataType.FLOAT16 if io_dtype == ir.DataType.BFLOAT16 else io_dtype
        super().__init__(config, visual, io_dtype, compute_dtype=compute_dtype)
        self.graph.name = "bailing_moe_v3_vl_vision"

    def make_vision_projector(self, merged_input, merged_hidden_size):
        shape = ["num_image_tokens", self.config.out_hidden_size]
        projected = self.make_linear("linear_proj.0", self.visual.projector.linear_fc1, merged_input, shape)
        activated = self.make_vision_gelu("/linear_proj/Gelu", projected, shape, approximate="none")
        return self.make_linear(
            "linear_proj.2",
            self.visual.projector.linear_fc2,
            activated,
            shape,
            output="image_features" if self.feature_dtype == self.io_dtype else None,
        )
