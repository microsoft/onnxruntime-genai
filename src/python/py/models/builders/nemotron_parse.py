# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import json
import math
import os
import warnings

import onnx_ir as ir
import torch
from transformers import AutoModel, AutoProcessor, GenerationConfig

from .base import Model
from .nemotron_parse_decoder import NemotronParseDecoderComponent, make_decoder_config
from .nemotron_parse_encoder import NemotronParseEncoderComponent


class NemotronParseModel(Model):
    """Build the RADIO/cross-KV encoder and unified mBART decoder."""

    default_user_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
    # The examples append the current user message before applying the template.
    chat_template = (
        "{%- set content = messages[-1]['content'] -%}"
        "{%- if content is string -%}{{ content }}"
        "{%- else -%}"
        "{%- for part in content -%}"
        "{%- if part['type'] == 'text' -%}{{ part['text'] }}{%- endif -%}"
        "{%- endfor -%}"
        "{%- endif -%}"
    )

    def __init__(
        self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options
    ):
        self.config = config
        io_dtype = (
            ir.DataType.FLOAT16 if io_dtype is None else ir.DataType(io_dtype)
        )
        onnx_dtype = (
            io_dtype if onnx_dtype is None else ir.DataType(onnx_dtype)
        )
        self.extra_options = dict(extra_options)
        removed_options = self.extra_options.keys() & {
            "image_height", "image_width", "prefill_sequence_length",
            "cache_sequence_length", "export_components", "torch_dtype",
        }
        if removed_options:
            raise ValueError(
                f"Nemotron Parse no longer accepts extra options: {', '.join(sorted(removed_options))}. "
                "Image size and cache capacity come from checkpoint metadata, prefill length "
                "from the default task prompt, and loading dtype from export precision; both graphs are always exported."
            )
        self.extra_options.setdefault("block_size", 32)
        self.generation_config = None
        self.processor = None

        self.image_height, self.image_width = self.resolve_image_size()
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Checkpoint image_size dimensions must be positive.")

        self.cache_sequence_length = int(self.config.max_sequence_length)
        if self.cache_sequence_length <= 0:
            raise ValueError("Checkpoint max_sequence_length must be positive.")

        patch_size = int(getattr(config.encoder, "patch_size", 16))
        encoder_grid_h = self.image_height // patch_size
        encoder_grid_w = self.image_width // patch_size
        compressed_grid_w = ((encoder_grid_w - 4) // 4) + 1
        if encoder_grid_h <= 0 or compressed_grid_w <= 0:
            raise ValueError(
                "The image size is too small for the encoder patch geometry."
            )
        self.encoder_sequence_length = (
            encoder_grid_h * compressed_grid_w + 1
        )

        super().__init__(
            make_decoder_config(config, self.cache_sequence_length),
            io_dtype, onnx_dtype, ep, cache_dir, self.extra_options,
        )
        self.model_type = "nemotron_parse"
        self.encoder_filename = "encoder.onnx"
        self.filename = self.decoder_filename = "decoder.onnx"
        self.input_names["input_ids"] = "decoder_input_ids"
        self.input_names["attention_mask"] = "decoder_attention_mask"
        self.input_names.pop("position_ids", None)

    def is_gqa_supported(self):
        return False

    def is_packed_attn_supported(self):
        return False

    def resolve_image_size(self):
        image_size = getattr(self.config, "image_size", None)
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("Nemotron Parse checkpoint image_size must contain height and width.")
        return int(image_size[0]), int(image_size[1])

    def torch_dtype(self):
        return {
            ir.DataType.FLOAT: torch.float32,
            ir.DataType.FLOAT16: torch.float16,
            ir.DataType.BFLOAT16: torch.bfloat16,
        }.get(self.onnx_dtype, "auto")

    def load_model(self, input_path):
        self.model_name_or_path = (
            input_path
            if os.path.isdir(input_path)
            else self.config._name_or_path
        )
        extra_kwargs = (
            {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        )
        torch_dtype = self.torch_dtype()
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        model.eval()
        self.generation_config = getattr(model, "generation_config", None)

        if getattr(model.config.decoder, "_attn_implementation", None) != "eager":
            model.config.decoder._attn_implementation = "eager"
        if (
            getattr(model.decoder.config, "_attn_implementation", None)
            != "eager"
        ):
            model.decoder.config._attn_implementation = "eager"
        return model

    def make_encoder_component(self, model):
        return NemotronParseEncoderComponent(
            self.config,
            model,
            self.io_dtype,
            self.onnx_dtype,
            self.ep,
            self.cache_dir,
            self.extra_options,
            image_height=self.image_height,
            image_width=self.image_width,
            encoder_sequence_length=self.encoder_sequence_length,
        )

    def make_decoder_component(self):
        return NemotronParseDecoderComponent(
            self.config,
            self.io_dtype,
            self.onnx_dtype,
            self.ep,
            self.cache_dir,
            self.extra_options,
            encoder_sequence_length=self.encoder_sequence_length,
            cache_sequence_length=self.cache_sequence_length,
        )

    def make_model(self, input_path):
        self.weights = self.load_model(input_path)

    def save_model(self, output_dir):
        try:
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
            component = self.make_encoder_component(self.weights)
            component.build()
            component.save_model(output_dir)

            # The explicit graph builder serializes parameters from CPU.
            # The RADIO encoder is no longer needed once its graph is saved.
            self.weights.encoder = None
            self.weights.decoder.to("cpu")
            self.weights.lm_head.to("cpu")
            component = self.make_decoder_component()
            component.build(self.weights)
            component.save_model(output_dir)
        finally:
            del self.weights

    def load_generation_config(self, extra_kwargs):
        if self.generation_config is None:
            try:
                self.generation_config = super().load_generation_config(extra_kwargs)
            except OSError as exc:
                warnings.warn(
                    f"Could not load generation_config.json from {self.model_name_or_path}: {exc}. "
                    "Using decoder configuration defaults.", stacklevel=2,
                )
                return GenerationConfig()
        return self.generation_config

    def resolve_special_token_ids(self, config, extra_kwargs):
        # Parse uses the decoder-start token, not a chat tokenizer's BOS/EOS rules.
        return (
            self.config.decoder_start_token_id,
            self.config.decoder.eos_token_id,
            self.config.decoder.pad_token_id,
        )

    def make_genai_config(self, config, extra_kwargs, out_dir):
        processor = self.load_processor(self.model_name_or_path, extra_kwargs)
        # Match native tokenization, then account for the model's decoder-start token.
        prompt_ids = processor.tokenizer.encode(self.default_user_prompt, add_special_tokens=True)
        self.prefill_sequence_length = len(prompt_ids) + 1
        if self.cache_sequence_length <= self.prefill_sequence_length:
            raise ValueError("Checkpoint max_sequence_length must leave room for at least one decoded token.")
        super().make_genai_config(copy.deepcopy(self.config.decoder), extra_kwargs, out_dir)

    def update_genai_config(self, genai_config):
        model = genai_config["model"]
        model["default_user_prompt"] = self.default_user_prompt
        model["encoder"] = {
            "outputs": {
                "cross_present_key_names": "cross_present.%d.key",
                "cross_present_value_names": "cross_present.%d.value",
            },
        }
        model["vision"] = {
            "filename": self.encoder_filename,
            "config_filename": "vision_processing.json",
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "encoder_hidden_states"},
            "num_visual_tokens": self.encoder_sequence_length,
        }
        decoder = model["decoder"]
        decoder["prefill_sequence_length"] = self.prefill_sequence_length
        decoder["inputs"].update({
            "cross_past_key_names": "cross_past_key_values.%d.key",
            "cross_past_value_names": "cross_past_key_values.%d.value",
            "cache_write_indices": "cache_write_indices",
        })
        if self.ep == "trt-rtx":
            # Profiles must be present before the base runtime appends the provider.
            # Only the decoder's token dimension is dynamic; all other inputs are fixed.
            vision_options = copy.deepcopy(decoder["session_options"])
            model["vision"]["session_options"] = vision_options
            for kind in ("min", "opt", "max"):
                key = f"ep.nvtensorrtrtxexecutionprovider.nv_profile_{kind}_shapes"
                decoder["session_options"][key] = f"{decoder['inputs']['input_ids']}:1x1"
                vision_options[key] = f"pixel_values:1x3x{self.image_height}x{self.image_width}"
        if self.ep == "cuda" and self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            # CUDA MatMulNBits cannot accept the bias attached by MatMulAddFusion.
            # TRT-RTX QDQ exports must retain this optimizer for INT4 fusion.
            decoder["session_options"]["optimization.disable_specified_optimizers"] = "MatMulAddFusion"

        search = genai_config["search"]
        search.update(num_beams=1, num_return_sequences=1, past_present_share_buffer=True)
        generation_config = self.generation_config or self.config.decoder
        for key in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "length_penalty"):
            value = getattr(generation_config, key, None)
            if value is not None:
                search[key] = value

    def load_processor(self, model_name_or_path, extra_kwargs):
        if self.processor is None:
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                token=self.hf_token,
                trust_remote_code=self.hf_remote,
                **extra_kwargs,
            )
            if getattr(processor, "tokenizer", None) is None:
                raise RuntimeError("Nemotron Parse processor does not expose a tokenizer")
            self.processor = processor
        return self.processor

    def save_processing(
        self, model_name_or_path, extra_kwargs, out_dir
    ):
        processor = self.load_processor(model_name_or_path, extra_kwargs)
        normalization = self.validate_image_processor(getattr(processor, "image_processor", None))
        tokenizer = processor.tokenizer

        print(
            f"Saving tokenizer and native image processor config in {out_dir}"
        )
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "chat_template.jinja"), "w", encoding="utf-8") as template_file:
            template_file.write(self.chat_template)
        processor_config = {
            **normalization,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "processor": {
                "name": "nemotron_parse_image_processor",
                "transforms": [
                    {
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    },
                ],
            }
        }
        with open(
            os.path.join(out_dir, "vision_processing.json"), "w"
        ) as processor_file:
            json.dump(processor_config, processor_file, indent=2)

    def validate_image_processor(self, processor):
        # The native processor implements this checkpoint's resize/pad algorithm,
        # not the full Transformers image-processing API.
        if processor is None or type(processor).__name__ != "NemotronParseImageProcessor":
            raise ValueError("Nemotron Parse requires a NemotronParseImageProcessor with the supported native contract")
        expected = {
            "do_resize": True, "do_rescale": True, "do_normalize": True, "do_pad": True,
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "resample": 2, "interpolation": 1,
            "padding_value": 255, "padding_mode": "constant", "padding_position": "center",
        }
        for name, supported in expected.items():
            value = getattr(processor, name, supported)
            if isinstance(supported, list):
                matches = isinstance(value, (list, tuple)) and len(value) == len(supported) and all(
                    math.isclose(float(a), b, rel_tol=0, abs_tol=1e-7) for a, b in zip(value, supported)
                )
            elif isinstance(supported, float):
                matches = isinstance(value, (int, float)) and math.isclose(value, supported, rel_tol=0, abs_tol=1e-9)
            else:
                matches = value == supported
            if not matches:
                raise ValueError(
                    f"Nemotron Parse native preprocessing does not support {name}={value!r}; expected {supported!r}"
                )
        transforms = getattr(getattr(processor, "transform", None), "transforms", [])
        if len(transforms) != 1 or type(transforms[0]).__name__ != "PadIfNeeded":
            raise ValueError("Nemotron Parse native preprocessing requires a single centered white PadIfNeeded transform")
        padding = transforms[0]
        fill = getattr(padding, "fill", getattr(padding, "value", None))
        white = fill == 255 if isinstance(fill, (int, float)) else fill in ([255, 255, 255], (255, 255, 255))
        position = getattr(padding, "position", None)
        position = getattr(position, "value", position)
        if not white or getattr(padding, "border_mode", None) != 0 or position != "center":
            raise ValueError(
                "Nemotron Parse native preprocessing requires centered constant white padding; "
                "check the source processor and its albumentations version"
            )
        tensor_transforms = getattr(getattr(processor, "torch_transform", None), "transforms", [])
        if len(tensor_transforms) != 1 or type(tensor_transforms[0]).__name__ != "ToTensor":
            raise ValueError("Nemotron Parse native preprocessing requires the checkpoint's ToTensor transform")
        return {
            name: [float(value) for value in getattr(processor, name, expected[name])]
            for name in ("image_mean", "image_std")
        }
