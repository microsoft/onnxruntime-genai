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
from .gemma import Gemma2Model, Gemma3Model, Gemma4MoEModel, Gemma4Model, GemmaModel
from .gptoss import GPTOSSModel
from .granite import GraniteModel, GraniteMoEHybridModel
from .hunyuan import HunyuanDenseV1Model
from .internlm import InternLM2Model
from .lfm2 import LFM2Model
from .llama import LlamaModel
from .mistral import Mistral3TextModel, MistralModel
from .mtp import MTPModel
from .nemotron import NemotronModel
from .olmo import OLMoModel
from .phi import (
    Phi3MiniLongRoPEModel,
    Phi3MiniModel,
    Phi3MoELongRoPEModel,
    Phi3SmallLongRoPEModel,
    Phi3SmallModel,
    Phi3VModel,
    Phi4MMModel,
    PhiModel,
)
from .qwen import (
    Qwen3Model,
    Qwen3VLTextModel,
    Qwen25VLTextModel,
    Qwen35DenseMTPModel,
    Qwen35Model,
    Qwen35MoEModel,
    Qwen35MoETextModel,
    Qwen35MTPModel,
    Qwen35TextModel,
    QwenModel,
    VideoChatFlashQwenModel,
)
from .smollm import SmolLM3Model
from .whisper import WhisperModel

__all__ = [
    "ChatGLMModel",
    "ErnieModel",
    "GPTOSSModel",
    "Gemma2Model",
    "Gemma3Model",
    "Gemma4MoEModel",
    "Gemma4Model",
    "GemmaModel",
    "GraniteMoEHybridModel",
    "GraniteModel",
    "HunyuanDenseV1Model",
    "InternLM2Model",
    "LFM2Model",
    "LlamaModel",
    "MTPModel",
    "Mistral3TextModel",
    "MistralModel",
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
    "Qwen3Model",
    "Qwen3VLTextModel",
    "Qwen25VLTextModel",
    "Qwen35DenseMTPModel",
    "Qwen35Model",
    "Qwen35MoEModel",
    "Qwen35MoETextModel",
    "Qwen35MTPModel",
    "Qwen35TextModel",
    "QwenModel",
    "SmolLM3Model",
    "VideoChatFlashQwenModel",
    "WhisperModel",
]
