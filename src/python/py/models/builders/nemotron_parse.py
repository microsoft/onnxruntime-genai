# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.

import json
import os

import onnx
import onnx_ir as ir
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, QuantFormat
from transformers import AutoModel, AutoProcessor

from .base import parse_hf_token


class _NemotronParseEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values, return_dict=True).last_hidden_state


class _NemotronParseDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return self.lm_head(decoder_outputs.last_hidden_state)


class NemotronParseModel:
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        self.io_dtype = io_dtype
        self.onnx_dtype = onnx_dtype
        self.ep = ep
        self.cache_dir = cache_dir
        self.extra_options = extra_options
        self.hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
        self.hf_remote = extra_options.get("hf_remote", True)
        self.model_name_or_path = None
        self.model_type = "nemotron_parse"

        self.image_height = int(extra_options.get("image_height", 768))
        self.image_width = int(extra_options.get("image_width", 768))
        self.decoder_sequence_length = int(extra_options.get("decoder_sequence_length", 8))
        self.export_components = extra_options.get("export_components", "encoder,decoder")

        patch_size = int(getattr(config.encoder, "patch_size", 16))
        encoder_grid_h = self.image_height // patch_size
        encoder_grid_w = self.image_width // patch_size
        compressed_grid_w = ((encoder_grid_w - 4) // 4) + 1
        self.encoder_sequence_length = encoder_grid_h * compressed_grid_w + 1

        self.encoder_filename = "encoder.onnx"
        self.decoder_filename = "decoder.onnx"
        self.opset_version = int(extra_options.get("opset_version", 20))

    def _provider_options(self):
        if self.ep == "cpu":
            return []
        ep_name = self.ep.replace("trt-rtx", "NvTensorRtRtx")
        attrs = {"enable_cuda_graph": "1"} if self.ep == "trt-rtx" else {}
        return [{ep_name: attrs}]

    def _torch_dtype(self):
        if self.extra_options.get("torch_dtype", None) == "fp32":
            return torch.float32
        if self.extra_options.get("torch_dtype", None) == "fp16":
            return torch.float16
        if self.extra_options.get("torch_dtype", None) == "bf16":
            return torch.bfloat16
        # HF "auto" currently exports BF16 Conv inputs for this model, which
        # ONNX Runtime rejects before provider execution. Use FP16 by default.
        return torch.float16

    def _load_model(self, input_path):
        self.model_name_or_path = input_path if os.path.isdir(input_path) else self.config._name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        torch_dtype = self._torch_dtype()
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        if isinstance(torch_dtype, torch.dtype):
            model.to(dtype=torch_dtype)
        model.eval()

        if getattr(model.config.decoder, "_attn_implementation", None) != "eager":
            model.config.decoder._attn_implementation = "eager"
        if getattr(model.decoder.config, "_attn_implementation", None) != "eager":
            model.decoder.config._attn_implementation = "eager"

        return model

    def _maybe_quantize_int4(self, onnx_path):
        if self.onnx_dtype not in {ir.DataType.INT4, ir.DataType.UINT4}:
            return

        print(f"Quantizing MatMul weights in {onnx_path}")
        model = onnx.load(onnx_path, load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=int(self.extra_options.get("int4_block_size", 32)),
            is_symmetric=self.extra_options.get("int4_is_symmetric", True),
            accuracy_level=int(self.extra_options.get("int4_accuracy_level", 0)),
            quant_format=QuantFormat.QDQ if self.extra_options.get("use_qdq", False) else QuantFormat.QOperator,
            op_types_to_quantize=self.extra_options.get("int4_op_types_to_quantize", ("MatMul",)),
            nodes_to_exclude=self.extra_options.get("int4_nodes_to_exclude", []),
        )
        quantizer.process()
        data_location = os.path.basename(onnx_path) + ".data"
        onnx.save_model(
            quantizer.model.model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_location,
            size_threshold=1024,
            convert_attribute=False,
        )

    def _export_encoder(self, model, output_dir):
        out_path = os.path.join(output_dir, self.encoder_filename)
        print(f"Exporting Nemotron Parse encoder to {out_path}")
        wrapper = _NemotronParseEncoder(model)
        dtype = next(wrapper.parameters()).dtype
        pixel_values = torch.randn((1, 3, self.image_height, self.image_width), dtype=dtype)
        with torch.no_grad():
            wrapper(pixel_values)
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            out_path,
            input_names=["pixel_values"],
            output_names=["encoder_hidden_states"],
            opset_version=self.opset_version,
            external_data=True,
            dynamo=False,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        )
        self._maybe_quantize_int4(out_path)

    def _export_decoder(self, model, output_dir):
        out_path = os.path.join(output_dir, self.decoder_filename)
        print(f"Exporting Nemotron Parse decoder to {out_path}")
        wrapper = _NemotronParseDecoder(model)
        dtype = next(model.decoder.parameters()).dtype
        decoder_input_ids = torch.ones((1, self.decoder_sequence_length), dtype=torch.long)
        decoder_attention_mask = torch.ones((1, self.decoder_sequence_length), dtype=torch.long)
        encoder_hidden_states = torch.randn((1, self.encoder_sequence_length, self.config.decoder.d_model), dtype=dtype)
        torch.onnx.export(
            wrapper,
            (decoder_input_ids, decoder_attention_mask, encoder_hidden_states),
            out_path,
            input_names=["decoder_input_ids", "decoder_attention_mask", "encoder_hidden_states"],
            output_names=["logits"],
            opset_version=self.opset_version,
            external_data=True,
            dynamo=False,
            dynamic_axes={
                "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
                "decoder_attention_mask": {0: "batch_size", 1: "decoder_sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "logits": {0: "batch_size", 1: "decoder_sequence_length"},
            },
        )
        self._maybe_quantize_int4(out_path)

    def make_model(self, input_path):
        model = self._load_model(input_path)
        self._model = model

    def save_model(self, output_dir):
        components = {component.strip() for component in self.export_components.split(",")}
        if "encoder" in components:
            self._export_encoder(self._model, output_dir)
        if "decoder" in components:
            self._export_decoder(self._model, output_dir)
        del self._model

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        decoder_config = self.config.decoder
        genai_config = {
            "model": {
                "type": self.model_type,
                "bos_token_id": self.config.decoder_start_token_id,
                "eos_token_id": decoder_config.eos_token_id,
                "pad_token_id": decoder_config.pad_token_id,
                "context_length": self.config.max_sequence_length,
                "vocab_size": decoder_config.vocab_size,
                "vision": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": self._provider_options(),
                    },
                    "filename": self.encoder_filename,
                    "inputs": {"pixel_values": "pixel_values"},
                    "outputs": {"encoder_hidden_states": "encoder_hidden_states"},
                    "image_height": self.image_height,
                    "image_width": self.image_width,
                    "encoder_sequence_length": self.encoder_sequence_length,
                },
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": self._provider_options(),
                    },
                    "filename": self.decoder_filename,
                    "hidden_size": decoder_config.d_model,
                    "head_size": decoder_config.d_model // decoder_config.decoder_attention_heads,
                    "num_attention_heads": decoder_config.decoder_attention_heads,
                    "num_hidden_layers": decoder_config.decoder_layers,
                    "num_key_value_heads": decoder_config.decoder_attention_heads,
                    "inputs": {
                        "input_ids": "decoder_input_ids",
                        "attention_mask": "decoder_attention_mask",
                        "encoder_hidden_states": "encoder_hidden_states",
                    },
                    "outputs": {"logits": "logits"},
                },
            },
            "search": {
                "do_sample": False,
                "early_stopping": True,
                "max_length": self.config.max_sequence_length,
                "min_length": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": False,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
            },
        }

        out_path = os.path.join(out_dir, "genai_config.json")
        print(f"Saving GenAI config in {out_path}")
        with open(out_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )
        image_processor = getattr(processor, "image_processor", None)
        for target in (processor, image_processor):
            if target is None:
                continue
            if hasattr(target, "target_height"):
                target.target_height = self.image_height
            if hasattr(target, "target_width"):
                target.target_width = self.image_width
            if hasattr(target, "final_size"):
                target.final_size = [self.image_height, self.image_width]
            if hasattr(target, "size"):
                target.size = {"height": self.image_height, "width": self.image_width}
        print(f"Saving processing files in {out_dir} for Nemotron Parse")
        processor.save_pretrained(out_dir)
