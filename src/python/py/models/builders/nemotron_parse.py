# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import json
import os

import onnx
import onnx_ir as ir
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    QuantFormat,
)
from transformers import AutoModel, AutoProcessor

from .nemotron_parse_decoder import NemotronParseDecoderComponent


def _resolve_image_size(config, extra_options):
    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        default_height, default_width = image_size[:2]
    else:
        default_height = default_width = 768

    return (
        int(extra_options.get("image_height", default_height)),
        int(extra_options.get("image_width", default_width)),
    )


def _save_model_with_external_data(model, onnx_path):
    external_data_path = onnx_path + ".data"
    if os.path.exists(external_data_path):
        os.unlink(external_data_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(external_data_path),
        size_threshold=1024,
        convert_attribute=False,
    )


class _NemotronParseEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(
            pixel_values, return_dict=True
        ).last_hidden_state


class NemotronParseModel:
    """Build the RADIO encoder plus explicit mBART prefill/decode components."""

    def __init__(
        self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options
    ):
        self.config = config
        self.io_dtype = (
            ir.DataType.FLOAT16 if io_dtype is None else ir.DataType(io_dtype)
        )
        self.onnx_dtype = (
            self.io_dtype if onnx_dtype is None else ir.DataType(onnx_dtype)
        )
        self.ep = ep
        self.cache_dir = cache_dir
        self.extra_options = dict(extra_options)
        self.extra_options.setdefault("block_size", 32)
        self.hf_token = self.extra_options.get("hf_token", True)
        self.hf_remote = self.extra_options.get("hf_remote", False)
        self.model_name_or_path = None
        self.model_type = "nemotron_parse"

        self.image_height, self.image_width = _resolve_image_size(
            config, self.extra_options
        )
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive.")

        self.prefill_sequence_length = int(
            self.extra_options.get("prefill_sequence_length", 8)
        )
        self.cache_sequence_length = int(
            self.extra_options.get(
                "cache_sequence_length", self.config.max_sequence_length
            )
        )
        if self.prefill_sequence_length <= 0:
            raise ValueError("prefill_sequence_length must be positive.")
        if self.cache_sequence_length <= 0:
            raise ValueError("cache_sequence_length must be positive.")
        if self.cache_sequence_length <= self.prefill_sequence_length:
            raise ValueError(
                "cache_sequence_length must leave room for at least one decoded token."
            )

        self.export_components = {
            component.strip()
            for component in self.extra_options.get(
                "export_components", "encoder,decoder"
            ).split(",")
            if component.strip()
        }
        unsupported_components = self.export_components - {
            "encoder",
            "decoder",
        }
        if not self.export_components or unsupported_components:
            raise ValueError(
                "export_components must contain only encoder and/or decoder."
            )

        self.export_device = str(
            self.extra_options.get("export_device", "cpu")
        ).lower()
        if self.export_device not in {"cpu", "cuda"}:
            raise ValueError("export_device must be cpu or cuda.")

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

        self.encoder_filename = "encoder.onnx"
        self.decoder_filename = "decoder.onnx"
        self.decoder_prefill_filename = "decoder_prefill.onnx"
        self.encoder_opset_version = int(
            self.extra_options.get("opset_version", 20)
        )

    def _provider_options(self):
        if self.ep == "cpu":
            return []
        ep_name = self.ep.replace("trt-rtx", "NvTensorRtRtx")
        attrs = (
            {"enable_cuda_graph": "1"} if self.ep == "trt-rtx" else {}
        )
        return [{ep_name: attrs}]

    def _torch_dtype(self):
        dtype = self.extra_options.get("torch_dtype")
        if dtype == "fp32":
            return torch.float32
        if dtype == "bf16":
            return torch.bfloat16
        return torch.float16

    def _load_model(self, input_path):
        self.model_name_or_path = (
            input_path
            if os.path.isdir(input_path)
            else self.config._name_or_path
        )
        extra_kwargs = (
            {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        )
        torch_dtype = self._torch_dtype()
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        if self.export_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "export_device=cuda requested, but CUDA is not available."
            )
        model.to(
            device=torch.device(self.export_device), dtype=torch_dtype
        )
        model.eval()

        if getattr(model.config.decoder, "_attn_implementation", None) != "eager":
            model.config.decoder._attn_implementation = "eager"
        if (
            getattr(model.decoder.config, "_attn_implementation", None)
            != "eager"
        ):
            model.decoder.config._attn_implementation = "eager"
        return model

    def _maybe_quantize_encoder_int4(self, onnx_path):
        if self.onnx_dtype not in {
            ir.DataType.INT4,
            ir.DataType.UINT4,
        }:
            return

        print(f"Quantizing encoder MatMul weights in {onnx_path}")
        model = onnx.load(onnx_path, load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model,
            bits=4,
            block_size=int(self.extra_options.get("block_size", 32)),
            is_symmetric=self.extra_options.get("is_symmetric", True),
            accuracy_level=int(
                self.extra_options.get("accuracy_level", 0)
            ),
            quant_format=(
                QuantFormat.QDQ
                if self.extra_options.get("use_qdq", False)
                else QuantFormat.QOperator
            ),
            op_types_to_quantize=self.extra_options.get(
                "op_types_to_quantize", ("MatMul",)
            ),
            nodes_to_exclude=self.extra_options.get(
                "nodes_to_exclude", []
            ),
        )
        quantizer.process()
        _save_model_with_external_data(
            quantizer.model.model, onnx_path
        )

    def _export_encoder(self, model, output_dir):
        out_path = os.path.join(output_dir, self.encoder_filename)
        print(f"Exporting Nemotron Parse RADIO encoder to {out_path}")
        external_data_path = f"{out_path}.data"
        if os.path.exists(external_data_path):
            os.unlink(external_data_path)
        wrapper = _NemotronParseEncoder(model)
        dtype = next(wrapper.parameters()).dtype
        device = next(wrapper.parameters()).device
        pixel_values = torch.randn(
            (1, 3, self.image_height, self.image_width),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            wrapper(pixel_values)
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            out_path,
            input_names=["pixel_values"],
            output_names=["encoder_hidden_states"],
            opset_version=self.encoder_opset_version,
            external_data=True,
            dynamo=False,
        )
        self._maybe_quantize_encoder_int4(out_path)

    def _make_decoder_component(self, phase):
        return NemotronParseDecoderComponent(
            self.config,
            self.io_dtype,
            self.onnx_dtype,
            self.ep,
            self.cache_dir,
            self.extra_options,
            phase=phase,
            encoder_sequence_length=self.encoder_sequence_length,
            prefill_sequence_length=self.prefill_sequence_length,
            cache_sequence_length=self.cache_sequence_length,
        )

    def make_model(self, input_path):
        self._model = self._load_model(input_path)

    def save_model(self, output_dir):
        try:
            if "encoder" in self.export_components:
                self._export_encoder(self._model, output_dir)

            if "decoder" in self.export_components:
                # The explicit graph builder serializes parameters from CPU.
                # The RADIO encoder is no longer needed once its graph is saved.
                self._model.encoder = None
                self._model.decoder.to("cpu")
                self._model.lm_head.to("cpu")
                for phase in ("prefill", "decode"):
                    if self.cache_dir:
                        os.makedirs(self.cache_dir, exist_ok=True)
                    component = self._make_decoder_component(phase)
                    component.build(self._model)
                    component.save_model(output_dir)
        finally:
            del self._model

    def make_genai_config(
        self, model_name_or_path, extra_kwargs, out_dir
    ):
        decoder_config = self.config.decoder
        genai_config = {
            "model": {
                "type": self.model_type,
                "bos_token_id": self.config.decoder_start_token_id,
                "eos_token_id": decoder_config.eos_token_id,
                "pad_token_id": decoder_config.pad_token_id,
                "context_length": self.cache_sequence_length,
                "vocab_size": decoder_config.vocab_size,
                "vision": {
                    "filename": self.encoder_filename,
                    "config_filename": "processor_config.json",
                    "inputs": {"pixel_values": "pixel_values"},
                    "outputs": {
                        "image_features": "encoder_hidden_states"
                    },
                    "num_visual_tokens": self.encoder_sequence_length,
                },
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": self._provider_options(),
                    },
                    "filename": self.decoder_filename,
                    "prefill_filename": self.decoder_prefill_filename,
                    "prefill_sequence_length": self.prefill_sequence_length,
                    "hidden_size": decoder_config.d_model,
                    "head_size": (
                        decoder_config.d_model
                        // decoder_config.decoder_attention_heads
                    ),
                    "num_attention_heads": (
                        decoder_config.decoder_attention_heads
                    ),
                    "num_hidden_layers": decoder_config.decoder_layers,
                    "num_key_value_heads": (
                        decoder_config.decoder_attention_heads
                    ),
                    "inputs": {
                        "input_ids": "decoder_input_ids",
                        "attention_mask": "decoder_attention_mask",
                        "encoder_hidden_states": "encoder_hidden_states",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value",
                        "cross_past_key_names": (
                            "cross_past_key_values.%d.key"
                        ),
                        "cross_past_value_names": (
                            "cross_past_key_values.%d.value"
                        ),
                        "cache_write_indices": "cache_write_indices",
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value",
                        "cross_present_key_names": (
                            "cross_present.%d.key"
                        ),
                        "cross_present_value_names": (
                            "cross_present.%d.value"
                        ),
                    },
                },
            },
            "search": {
                "do_sample": False,
                "early_stopping": True,
                "max_length": self.cache_sequence_length,
                "min_length": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": True,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
            },
        }

        out_path = os.path.join(out_dir, "genai_config.json")
        print(f"Saving GenAI config in {out_path}")
        with open(out_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)

    def save_processing(
        self, model_name_or_path, extra_kwargs, out_dir
    ):
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Nemotron Parse processor does not expose a tokenizer"
            )

        print(
            f"Saving tokenizer and native image processor config in {out_dir}"
        )
        tokenizer.save_pretrained(out_dir)
        processor_config = {
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
            os.path.join(out_dir, "processor_config.json"), "w"
        ) as processor_file:
            json.dump(processor_config, processor_file, indent=2)
