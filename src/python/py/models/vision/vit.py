# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Plain ViT (Vision Transformer) export.

Supports: google/vit-base-patch16-224, google/vit-large-patch16-224, etc.

Architecture:
  - Patch embedding: Conv2d(16x16) → reshape → CLS prepend → absolute pos embed
  - Encoder: N × ViTLayer (LayerNorm → MHA → LayerNorm → FC1/GELU/FC2)
  - Final LayerNorm
  - Optional: linear projection head (for classification models)

All defaults in VisionModel match plain ViT exactly — zero extra code needed.
"""
from .base import VisionModel


class ViTModel(VisionModel):
    """
    Plain ViT (google/vit-*).

    VisionModel defaults match ViT exactly:
      - has_cls_token = True
      - has_pos_embed = True, pos_embed_type = "absolute"
      - LayerNorm with bias (simple=False)
      - FC1/GELU/FC2 MLP (use_fc=True)
      - Full bidirectional MHA (no RoPE, no KV cache)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # All VisionModel defaults are correct for plain ViT.
        # No overrides needed.
