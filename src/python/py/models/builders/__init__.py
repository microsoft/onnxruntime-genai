# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
from importlib import import_module

__all__ = [
    "ChatGLMModel",
    "ErnieModel",
    "GPTOSSModel",
    "Gemma2Model",
    "Gemma3Model",
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


def __getattr__(name):
    modules = {
        "ChatGLMModel": "chatglm",
        "ErnieModel": "ernie",
        "GPTOSSModel": "gptoss",
        "Gemma2Model": "gemma",
        "Gemma3Model": "gemma",
        "GemmaModel": "gemma",
        "GraniteMoEHybridModel": "granite",
        "GraniteModel": "granite",
        "HunyuanDenseV1Model": "hunyuan",
        "InternLM2Model": "internlm",
        "LFM2Model": "lfm2",
        "LlamaModel": "llama",
        "MTPModel": "mtp",
        "Mistral3TextModel": "mistral",
        "MistralModel": "mistral",
        "Model": "base",
        "NemotronModel": "nemotron",
        "OLMoModel": "olmo",
        "Phi3MiniLongRoPEModel": "phi",
        "Phi3MiniModel": "phi",
        "Phi3MoELongRoPEModel": "phi",
        "Phi3SmallLongRoPEModel": "phi",
        "Phi3SmallModel": "phi",
        "Phi3VModel": "phi",
        "Phi4MMModel": "phi",
        "PhiModel": "phi",
        "Qwen3Model": "qwen",
        "Qwen3VLTextModel": "qwen",
        "Qwen25VLTextModel": "qwen",
        "Qwen35DenseMTPModel": "qwen",
        "Qwen35Model": "qwen",
        "Qwen35MoEModel": "qwen",
        "Qwen35MoETextModel": "qwen",
        "Qwen35MTPModel": "qwen",
        "Qwen35TextModel": "qwen",
        "QwenModel": "qwen",
        "SmolLM3Model": "smollm",
        "VideoChatFlashQwenModel": "qwen",
        "WhisperModel": "whisper",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{modules[name]}", __name__), name)
