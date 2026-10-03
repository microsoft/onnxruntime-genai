# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# ------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.

import os

from transformers import Qwen2ForCausalLM

from .base import Model
from .expansions.qwen import Qwen
from .qwen3_5 import (
    Qwen35DenseMTPModel,
    Qwen35Model,
    Qwen35MoEModel,
    Qwen35MoETextModel,
    Qwen35MTPModel,
    Qwen35TextModel,
)
from .qwen3_8 import (
    Qwen4ExpEmbeddingModel,
    Qwen4ExpEngramModel,
    Qwen4ExpModel,
    Qwen4ExpMTPTextModel,
    Qwen4ExpTextModel,
    Qwen4ExpVisionModel,
)


class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)



class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)



class Qwen25VLTextModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Compute LayerNorms in FP32 for better accuracy
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # Compute RoPE in FP32 for better accuracy
        self.rope_attrs["cast"]["use_fp32"] = True
        self.rope_attrs["cast"]["root_input"] = True
        self.rope_attrs["cast"]["output_0"] = True

    def is_packed_matmul_supported(self):
        # We need separate Q, K, V tensors to apply MRoPE manually.
        return False

    def is_fused_rope_supported(self):
        # Qwen 2.5 VL applies MRoPE manually before attention, not fused in the op
        return False

    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = (
            [3, "num_tokens"] if self.use_paged_attention else [3, "batch_size", "sequence_length"]
        )
        super().make_inputs_and_outputs()



class Qwen3VLTextModel(Qwen, Qwen25VLTextModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Avoid duplicate Cast nodes that form a SkipLayerNorm --> Cast --> Cast --> SkipLayerNorm pattern
        self.layernorm_attrs["cast"]["output_3"] = False

        # Qwen3-VL uses QK norms whose outputs will have already been casted to FP32
        self.rope_attrs["cast"]["root_input"] = False

        # Qwen3 attention uses QK normalization
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

        # Qwen3-VL uses the Interleaved MRotaryEmbedding layout.
        self.rope_attrs["mrope_layout"] = 1




class VideoChatFlashQwenModel(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def load_weights(self, input_path):
        # Load the standard Qwen2 backbone without importing the checkpoint's
        # custom video modeling code and its optional dependencies.
        extra_kwargs = {} if os.path.isdir(self.model_name_or_path) else {"cache_dir": self.cache_dir}
        return Qwen2ForCausalLM.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            **extra_kwargs,
        )



__all__ = [
    "Qwen3Model",
    "Qwen3VLTextModel",
    "Qwen4ExpEmbeddingModel",
    "Qwen4ExpEngramModel",
    "Qwen4ExpMTPTextModel",
    "Qwen4ExpModel",
    "Qwen4ExpTextModel",
    "Qwen4ExpVisionModel",
    "Qwen25VLTextModel",
    "Qwen35DenseMTPModel",
    "Qwen35MTPModel",
    "Qwen35MoEModel",
    "Qwen35MoETextModel",
    "Qwen35Model",
    "Qwen35TextModel",
    "QwenModel",
    "VideoChatFlashQwenModel",
]
