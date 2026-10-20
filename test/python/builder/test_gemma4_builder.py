# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Unit tests for the Gemma4Model builder (gemma4-12b-it text component).

These exercise the builder's structural deviations from Gemma3 without loading
weights or running the full pipeline:
  - RMSNorm uses the weight directly (add_offset == 0), unlike Gemma1/2/3.
  - Attention scale is 1.0 (Q/K are RMS-normed), not 1/sqrt(head_dim).
  - Per-layer geometry: sliding layers use head_dim/kv_heads distinct from the
    full ("global") layers, emitted as concrete KV-cache I/O shapes.
  - Proportional RoPE on the global layers (partial rotary on the global
    head_dim with a zero-padded NoPE tail) matches the HF reference.
  - Local (sliding) RoPE uses the sliding theta on the sliding head_dim.

Run with:
    python -m pytest test/python/models/test_gemma4_builder.py -v --test_models <any>
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import onnx_ir as ir
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.gemma import Gemma4Model

# 6-layer synthetic config: 5 sliding + 1 full, mirroring the gemma4 pattern.
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 4


def _make_minimal_config(**overrides):
    layer_types = overrides.pop("layer_types", LAYER_TYPES)
    cfg = types.SimpleNamespace(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        model_type="gemma4_unified_text",
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,  # sliding KV heads
        num_global_key_value_heads=1,  # full KV heads
        num_hidden_layers=len(layer_types),
        intermediate_size=512,
        vocab_size=128,
        max_position_embeddings=64,
        hidden_activation="gelu_pytorch_tanh",
        head_dim=32,  # sliding head dim
        global_head_dim=64,  # full head dim
        sliding_window=16,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        _name_or_path="",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _hf_proportional_cos_sin(head_dim, base, partial_rotary_factor, cache_length):
    """HF _compute_proportional_rope_parameters reference (cos/sin, halved)."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim))
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)), dim=0)
    else:
        inv_freq = inv_freq_rotated
    t = torch.arange(cache_length, dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, : head_dim // 2].numpy()
    sin = emb.sin()[:, : head_dim // 2].numpy()
    return cos, sin


def _hf_default_cos_sin(head_dim, base, cache_length):
    """HF default RoPE reference (cos/sin, halved)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    t = torch.arange(cache_length, dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, : head_dim // 2].numpy()
    sin = emb.sin()[:, : head_dim // 2].numpy()
    return cos, sin


class TestGemma4Model:
    def _make_model(self, **overrides):
        cfg = _make_minimal_config(**overrides)
        return Gemma4Model(
            cfg,
            io_dtype=ir.DataType.FLOAT,
            onnx_dtype=ir.DataType.FLOAT,
            ep="cpu",
            cache_dir=None,
            extra_options={},
        )

    def test_layernorm_no_offset(self):
        """Gemma4 RMSNorm uses the weight directly (add_offset == 0)."""
        m = self._make_model()
        assert m.layernorm_attrs["add_offset"] == 0

    def test_attention_scale_is_one(self):
        """Q/K are RMS-normed, so the attention scale is 1.0."""
        m = self._make_model()
        assert m.attention_attrs["scale"] == 1.0

    def test_is_local_matches_layer_types(self):
        """is_local reflects the config's per-layer attention type."""
        m = self._make_model()
        for i, lt in enumerate(LAYER_TYPES):
            assert m.is_local(i) == (lt == "sliding_attention")

    def test_per_layer_kv_cache_shape(self):
        """Sliding vs full layers emit concrete, distinct KV-cache shapes."""
        m = self._make_model()
        template = ["batch_size", m.num_kv_heads, "past_sequence_length", "kv_cache_dim"]
        sliding = m.make_key_value_cache_shape(0, list(template))
        full = m.make_key_value_cache_shape(5, list(template))
        # [batch, kv_heads, seq, head_dim]
        assert sliding[1] == 4 and sliding[3] == 32
        assert full[1] == 1 and full[3] == 64

    def test_external_rope_and_position_ids(self):
        """Partial rotary needs the standalone RotaryEmbedding op + position_ids."""
        m = self._make_model()
        assert m.attention_attrs["use_rope_in_attn"] is False
        assert "position_ids" in m.input_names

    def test_proportional_rope_cache_parity(self):
        """Global cache matches HF proportional RoPE (partial rotary + NoPE tail)."""
        m = self._make_model()
        builder_cos = m.values["cos_cache_global"].const_value.numpy()
        builder_sin = m.values["sin_cache_global"].const_value.numpy()
        hf_cos, hf_sin = _hf_proportional_cos_sin(
            head_dim=64, base=1000000.0, partial_rotary_factor=0.25, cache_length=64
        )
        np.testing.assert_allclose(builder_cos, hf_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(builder_sin, hf_sin, rtol=1e-5, atol=1e-5)

    def test_proportional_rope_nope_tail_is_identity(self):
        """The NoPE tail (zero frequencies) yields cos=1, sin=0 (no rotation)."""
        m = self._make_model()
        cos = m.values["cos_cache_global"].const_value.numpy()
        sin = m.values["sin_cache_global"].const_value.numpy()
        rope_angles = int(0.25 * 64 // 2)  # = 8
        assert np.allclose(cos[:, rope_angles:], 1.0)
        assert np.allclose(sin[:, rope_angles:], 0.0)

    def test_local_rope_cache_parity(self):
        """Local cache uses the sliding theta (1e4) on the sliding head_dim."""
        m = self._make_model()
        builder_cos = m.values["cos_cache_local"].const_value.numpy()
        builder_sin = m.values["sin_cache_local"].const_value.numpy()
        hf_cos, hf_sin = _hf_default_cos_sin(head_dim=32, base=10000.0, cache_length=64)
        np.testing.assert_allclose(builder_cos, hf_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(builder_sin, hf_sin, rtol=1e-5, atol=1e-5)

    def test_global_and_local_caches_differ(self):
        """Global (proportional, 1e6) and local (default, 1e4) caches must differ."""
        m = self._make_model()
        # Compare on the overlapping rotary columns only (local head_dim/2 = 16).
        gcos = m.values["cos_cache_global"].const_value.numpy()
        lcos = m.values["cos_cache_local"].const_value.numpy()
        assert not np.allclose(gcos[:, : lcos.shape[1]], lcos, rtol=1e-3, atol=1e-3)
