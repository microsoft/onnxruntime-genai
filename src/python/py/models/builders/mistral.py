# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import torch
from transformers import Mistral3ForConditionalGeneration

from .base import Model


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Mistral3TextModel(MistralModel):
    """Builder for the text decoder component of Mistral3 VLM models.

    Mistral3ForConditionalGeneration is a VLM whose text backbone
    is architecturally identical to MistralModel. This builder loads
    the full VLM and dequantizes FP8 weights if present.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # Cache image_token_id from the HF config to avoid a redundant network
        # call in make_genai_config (the config is already loaded by builder.py).
        self.image_token_id = getattr(config, "image_token_id", None)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        # Patch the generated genai_config.json with image_token_id so that
        # the runtime can locate image tokens in the input sequence.
        if self.image_token_id is not None:
            config_path = os.path.join(out_dir, "genai_config.json")
            with open(config_path) as f:
                genai_config = json.load(f)

            genai_config["model"]["image_token_id"] = self.image_token_id

            with open(config_path, "w") as f:
                json.dump(genai_config, f, indent=4)

    def load_weights(self, input_path):
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
        print("Loading Mistral3ForConditionalGeneration model...")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )

        # Dequantize FP8 weights in-place: dequantized = fp8_value * scale_inv
        fp8_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.float8_e4m3fn:
                scale_inv = getattr(module, "weight_scale_inv", None)
                if scale_inv is not None:
                    dequantized = module.weight.to(torch.bfloat16)
                    module.weight = torch.nn.Parameter(
                        dequantized * scale_inv.to(torch.bfloat16).reshape(-1, 1),
                        requires_grad=False,
                    )
                else:
                    raise ValueError(
                        f"FP8 weight '{name}' has no weight_scale_inv attribute. "
                        "FP8 weights require a scale for correct dequantization."
                    )
                fp8_count += 1
        if fp8_count > 0:
            print(f"Dequantized {fp8_count} FP8 linear layers to bfloat16")

        return model
