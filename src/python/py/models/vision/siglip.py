# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
SigLIP and SigLIP2 vision encoder export.

Supports:
  - google/siglip-base-patch16-224  (SigLIPModel)
  - google/siglip-so400m-patch14-384 (SigLIPModel)
  - google/siglip2-base-patch16-224  (SigLIP2Model)

Key differences from plain ViT:
  SigLIP:
    - No CLS token (output is all patch tokens, mean-pooled by downstream)
    - Absolute positional embedding (same as ViT)
    - FC1/GELU/FC2 MLP (same as ViT)
    - activation = "gelu_pytorch_tanh" (approximate GELU)

  SigLIP2 (adds on top of SigLIP):
    - SwiGLU MLP (gate_proj + up_proj → SiLU → down_proj)
    - QK-norm (RMSNorm after Q and K projections)
"""
from .base import VisionModel


class SigLIPModel(VisionModel):
    """
    SigLIP vision encoder (google/siglip-*).

    Differences from plain ViT:
      - No CLS token
      - activation = "gelu_pytorch_tanh"
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # SigLIP has no CLS token — output is all patch tokens
        self.patch_embed_attrs["has_cls_token"] = False

        # Update output shape: num_patches (not num_patches + 1)
        self.output_shapes["image_features"] = ["batch_size", "num_image_tokens", self.hidden_size]

        # SigLIP uses approximate GELU
        if self.activation in {"gelu", "gelu_new"}:
            self.activation = "gelu_pytorch_tanh"


class SigLIP2Model(SigLIPModel):
    """
    SigLIP2 vision encoder (google/siglip2-*).

    Adds on top of SigLIPModel:
      - SwiGLU MLP (gate_proj + up_proj → SiLU → down_proj)
      - QK-norm (RMSNorm after Q and K projections)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # SigLIP2 uses SwiGLU MLP instead of FC1/FC2
        self.mlp_attrs["use_fc"] = False
        self.mlp_attrs["use_proj"] = True

        # SigLIP2 uses QK-norm
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

        # QK-norm requires separate Q/K/V MatMuls (can't pack when norms are between them)
        self.attention_attrs["use_packed_matmul"] = False
