# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Vision encoder export package for onnxruntime-genai model builder.

Provides VisionModel base class and concrete implementations for:
  - Plain ViT (google/vit-*)
  - SigLIP / SigLIP2 (google/siglip-*)
  - DINOv2 (facebook/dinov2-*)
  - VLM-specific vision encoders (vision/vlm/)
"""
from .base import VisionModel
from .dinov2 import DINOv2Model
from .siglip import SigLIP2Model, SigLIPModel
from .vit import ViTModel
from .vlm.qwen import Qwen25VLVisionModel, Qwen3VLVisionModel

__all__ = [
    "VisionModel",
    "ViTModel",
    "SigLIPModel",
    "SigLIP2Model",
    "DINOv2Model",
    "Qwen25VLVisionModel",
    "Qwen3VLVisionModel",
]
