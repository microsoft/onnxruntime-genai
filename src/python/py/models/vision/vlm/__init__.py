# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
VLM-specific vision encoder implementations.

These classes export the vision encoder component of VLMs (Vision-Language Models).
They are used alongside the text decoder (builders/qwen.py etc.) to produce the
full set of ONNX models required by the onnxruntime-genai runtime:
  - vision.onnx       (this package)
  - embedding.onnx    (text embedding + multimodal projector)
  - model.onnx        (LLM text decoder)
"""
from .qwen import Qwen25VLVisionModel, Qwen3VLVisionModel

__all__ = [
    "Qwen25VLVisionModel",
    "Qwen3VLVisionModel",
]
