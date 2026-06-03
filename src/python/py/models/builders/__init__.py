# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
from .base import Model
from .chatglm import ChatGLMModel
from .ernie import ErnieModel
from .gemma import GemmaModel, Gemma2Model, Gemma3Model
from .gptoss import GPTOSSModel
from .granite import GraniteModel
from .hunyuan import HunyuanDenseV1Model
from .internlm import InternLM2Model
from .lfm2 import LFM2Model
from .llama import LlamaModel
from .mistral import MistralModel, Mistral3Model
from .nemotron import NemotronModel
from .olmo import OLMoModel
from .phi import (
    PhiModel,
    Phi3MiniModel,
    Phi3MiniLongRoPEModel,
    Phi3SmallModel,
    Phi3SmallLongRoPEModel,
    Phi3VModel,
    Phi3MoELongRoPEModel,
    Phi4MMModel,
)
from .qwen import QwenModel, Qwen25VLModel, Qwen3Model, Qwen3VLModel, Qwen35Model, Qwen35MoEModel, VideoChatFlashQwenModel
from .smollm import SmolLM3Model
from .whisper import WhisperModel

__all__ = [
    "ChatGLMModel",
    "ErnieModel",
    "GPTOSSModel",
    "Gemma2Model",
    "Gemma3Model",
    "GemmaModel",
    "GraniteModel",
    "HunyuanDenseV1Model",
    "InternLM2Model",
    "LFM2Model",
    "LlamaModel",
    "MistralModel",
    "Mistral3Model",
    "Model",
    "NemotronModel",
    "OLMoModel",
    "Phi3MiniLongRoPEModel",
    "Phi3MiniModel",
    "Phi3MoELongRoPEModel",
    "Phi3SmallLongRoPEModel",
    "Phi3SmallModel",
    "Phi3VModel",
    "Phi4MMModel",
    "PhiModel",
    "QwenModel",
    "Qwen25VLModel",
    "Qwen3Model",
    "Qwen3VLModel",
    "Qwen35Model",
    "Qwen35MoEModel",
    "SmolLM3Model",
    "VideoChatFlashQwenModel",
    "WhisperModel",
]
