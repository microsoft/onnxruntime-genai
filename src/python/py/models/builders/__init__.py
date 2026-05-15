# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
from .base import Model
from .chatglm import ChatGLMModel
from .ernie import ErnieModel
from .gemma import Gemma2Model, Gemma3Model, GemmaModel
from .gptoss import GPTOSSModel
from .granite import GraniteModel
from .internlm import InternLM2Model
from .lfm2 import LFM2Model
from .llama import LlamaModel
from .mistral import Mistral3TextModel, MistralModel
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
from .qwen import Qwen3Model, Qwen3VLTextModel, Qwen25VLTextModel, Qwen35TextModel, QwenModel
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
    "InternLM2Model",
    "LFM2Model",
    "LlamaModel",
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
    "Qwen35TextModel",
    "QwenModel",
    "SmolLM3Model",
    "WhisperModel",
]
