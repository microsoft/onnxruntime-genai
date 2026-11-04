# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Unit tests for the Gemma3Model builder's RoPE caches.

Gemma 3 checkpoints with rope_scaling (e.g. linear, factor 8) apply the scaling
to the global (full attention) layers only. The sliding-window layers use
unscaled default RoPE with rope_local_base_freq, as in Hugging Face.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import onnx_ir as ir
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.gemma import Gemma3Model

HEAD_DIM = 16
CACHE_LENGTH = 64
ROPE_THETA = 1000000.0
ROPE_LOCAL_THETA = 10000.0
LINEAR_FACTOR = 8.0


def make_config():
    return types.SimpleNamespace(
        architectures=["Gemma3ForCausalLM"],
        model_type="gemma3_text",
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=6,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=CACHE_LENGTH,
        hidden_activation="gelu_pytorch_tanh",
        head_dim=HEAD_DIM,
        query_pre_attn_scalar=HEAD_DIM,
        sliding_window=16,
        sliding_window_pattern=6,
        rms_norm_eps=1e-6,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        rope_theta=ROPE_THETA,
        rope_local_base_freq=ROPE_LOCAL_THETA,
        rope_scaling={"rope_type": "linear", "factor": LINEAR_FACTOR},
        tie_word_embeddings=True,
        _name_or_path="",
    )


def hf_cos_sin(base, factor):
    """HF default (factor 1) or linear RoPE reference (cos/sin, halved)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM)) / factor
    t = torch.arange(CACHE_LENGTH, dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().numpy(), freqs.sin().numpy()


class TestGemma3Rope:
    def make_model(self):
        return Gemma3Model(
            make_config(),
            io_dtype=ir.DataType.FLOAT,
            onnx_dtype=ir.DataType.FLOAT,
            ep="cpu",
            cache_dir=None,
            extra_options={},
        )

    def test_global_rope_cache_uses_linear_scaling(self):
        m = self.make_model()
        hf_cos, hf_sin = hf_cos_sin(ROPE_THETA, LINEAR_FACTOR)
        np.testing.assert_allclose(m.values["cos_cache_global"].const_value.numpy(), hf_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(m.values["sin_cache_global"].const_value.numpy(), hf_sin, rtol=1e-5, atol=1e-5)

    def test_local_rope_cache_is_unscaled(self):
        m = self.make_model()
        hf_cos, hf_sin = hf_cos_sin(ROPE_LOCAL_THETA, 1.0)
        np.testing.assert_allclose(m.values["cos_cache_local"].const_value.numpy(), hf_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(m.values["sin_cache_local"].const_value.numpy(), hf_sin, rtol=1e-5, atol=1e-5)
