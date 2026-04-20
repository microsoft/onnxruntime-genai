# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
YaRN RoPE cache parity test: verifies that the model builder's cos/sin
cache computation matches the HuggingFace transformers reference
implementation for YaRN-style rotary position embeddings.

This test covers the 4 bugs fixed in PR #2076:
  (a) hasattr on dict — config.rope_scaling is a dict, not an object with attributes
  (b) rope_theta fallback — theta stored only inside rope_scaling, not top-level
  (c) mscale=1.0 override — explicit mscale must be respected, not recomputed
  (d) inv_freq double-inversion — must use inv_freq/factor, not 1/(factor*inv_freq)

Run with:
    python -m pytest test/python/test_yarn_rope_parity.py -v --test_models <path-or-any-value>

The repository's pytest configuration requires ``--test_models`` during
argument parsing even though this test does not use model data.
"""

from __future__ import annotations

import math
import os
import sys
import types
from collections.abc import Mapping

import numpy as np
import torch

# Import Model from the source tree so tests always run against the working copy.
# The installed onnxruntime_genai package may be out of date during development.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))

from models.builders.base import Model

# ---------------------------------------------------------------------------
# Ministral-3-3B-Instruct-2512 YaRN configuration (text backbone)
# ---------------------------------------------------------------------------
MINISTRAL_3B_CONFIG = {
    "hidden_size": 3072,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 26,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 16.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 16384,
        "rope_theta": 1000000.0,
        "rope_type": "yarn",
        "type": "yarn",
    },
}

# ---------------------------------------------------------------------------
# Synthetic YaRN config with different parameters (no explicit mscale).
# Exercises the computed-mscale fallback path and different factor/theta.
# Modeled after DeepSeek-V2 style YaRN scaling.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GPT-OSS-20B YaRN configuration (from HuggingFace openai/gpt-oss-20b)
# Unlike Ministral-3-3B, rope_theta is a top-level config attribute (150000)
# and rope_scaling has no mscale/mscale_all_dim — exercises the computed
# mscale fallback path with a top-level theta.
# ---------------------------------------------------------------------------
GPTOSS_20B_CONFIG = {
    "hidden_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "num_hidden_layers": 24,
    "head_dim": 64,
    "max_position_embeddings": 131072,
    "rope_theta": 150000.0,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
    },
}

# ---------------------------------------------------------------------------
# Synthetic YaRN config with different parameters (no explicit mscale).
# Exercises the computed-mscale fallback path and different factor/theta.
# Modeled after DeepSeek-V2 style YaRN scaling.
# ---------------------------------------------------------------------------
YARN_NO_MSCALE_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "head_dim": 128,
    "max_position_embeddings": 163840,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 40.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rope_type": "yarn",
        "type": "yarn",
    },
}


def _make_hf_reference_cos_sin(config_dict: dict, cache_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute YaRN cos/sin caches using HuggingFace transformers reference."""
    rs = config_dict["rope_scaling"]
    base = rs.get("rope_theta", config_dict.get("rope_theta", 10000.0))
    dim = config_dict["head_dim"]
    factor = rs["factor"]
    original_max_position_embeddings = rs["original_max_position_embeddings"]
    beta_fast = rs["beta_fast"]
    beta_slow = rs["beta_slow"]

    # Compute attention_factor per HF convention
    config_mscale = rs.get("mscale", None)
    config_mscale_all_dim = rs.get("mscale_all_dim", None)

    def get_mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if config_mscale and config_mscale_all_dim:
        attention_factor = float(get_mscale(factor, config_mscale) / get_mscale(factor, config_mscale_all_dim))
    else:
        attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot: float, high_rot: float) -> tuple[float, float]:
        low = find_correction_dim(low_rot)
        high = find_correction_dim(high_rot)
        return max(low, 0), min(high, dim - 1)

    # Compute inv_freq with YaRN NTK scaling
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow)

    linear_func = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    inv_freq_mask = 1 - ramp_func

    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    # Build cos/sin caches
    t = torch.arange(cache_length, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = (emb.cos() * attention_factor).numpy()
    sin_cache = (emb.sin() * attention_factor).numpy()
    return cos_cache, sin_cache


def _make_builder_cos_sin(config_dict: dict, cache_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute YaRN cos/sin caches by driving the real builder code path.

    Uses ``Model.make_rope_init()`` with a mock config to exercise the actual
    bug-fix logic for dict-based rope_scaling, rope_theta fallback, mscale
    override, and YaRN inverse-frequency rescaling — rather than manually
    reconstructing rope_attrs in the test.
    """
    rs = config_dict["rope_scaling"]
    head_dim = config_dict["head_dim"]

    model = object.__new__(Model)
    model.head_size = head_dim
    model.context_length = config_dict["max_position_embeddings"]

    # Reproduce the original_context_length logic from Model.__init__
    if isinstance(rs, Mapping) and "original_max_position_embeddings" in rs:
        model.original_context_length = rs["original_max_position_embeddings"]
    else:
        model.original_context_length = model.context_length

    # Build a mock config that looks like what AutoConfig.from_pretrained returns.
    # Crucially, rope_scaling is a dict (not an object). Some models (e.g.
    # Ministral-3-3B) store rope_theta only inside rope_scaling, while others
    # (e.g. GPT-OSS-20B) have it as a top-level attribute.
    mock_kwargs = dict(
        head_dim=head_dim,
        max_position_embeddings=config_dict["max_position_embeddings"],
        rope_scaling=rs,
    )
    if "rope_theta" in config_dict:
        mock_kwargs["rope_theta"] = config_dict["rope_theta"]
    mock_config = types.SimpleNamespace(**mock_kwargs)

    # Resolve rope_theta using the same fallback chain as Model.__init__
    rope_theta = (
        mock_config.rope_theta
        if hasattr(mock_config, "rope_theta")
        else mock_config.rope_embedding_base
        if hasattr(mock_config, "rope_embedding_base")
        else mock_config.rope_scaling["rope_theta"]
        if hasattr(mock_config, "rope_scaling")
        and isinstance(mock_config.rope_scaling, Mapping)
        and "rope_theta" in mock_config.rope_scaling
        else 10000
    )

    # Initialize rope_attrs with defaults (as Model.__init__ does)
    model.rope_attrs = {
        "create_caches": True,
        "save_caches": False,
        "cache_length": cache_length,
        "theta": rope_theta,
        "partial_rotary_factor": 1.0,
        "interleaved": 0,
        "rotary_embedding_dim": 0,
        "rescale_factors": 1,
        "t_dtype": torch.int64,
        "position_scale": 1,
        "mscale": 1.0,
    }

    # Drive the real make_rope_init which sets mscale, rescale_inv_freq, etc.
    model.make_rope_init(mock_config)

    # Verify make_rope_init set the expected attributes
    expected_theta = rs.get("rope_theta", config_dict.get("rope_theta", 10000.0))
    assert model.rope_attrs["theta"] == expected_theta
    assert model.rope_attrs["mscale_policy"] == rs["rope_type"]
    if "mscale" in rs and rs["mscale"] > 0:
        assert model.rope_attrs["mscale"] == float(rs["mscale"])
    else:
        # When no explicit mscale, computed value should be > 0
        assert model.rope_attrs["mscale"] > 0
    assert model.rope_attrs["rescale_inv_freq"] == {
        "factor": rs["factor"],
        "ntk_alpha": rs["beta_slow"],
        "ntk_beta": rs["beta_fast"],
    }

    cos_cache, sin_cache = model.make_rotary_embedding_caches_from_scratch()
    return cos_cache.numpy(), sin_cache.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CACHE_LENGTH = 128  # Small cache for fast testing


class TestYarnRopeCacheParity:
    """Verify model builder produces identical cos/sin caches to HuggingFace."""

    def test_ministral_3b_cos_sin_match(self):
        """End-to-end parity: builder cos/sin caches match HF reference for Ministral-3-3B."""
        hf_cos, hf_sin = _make_hf_reference_cos_sin(MINISTRAL_3B_CONFIG, CACHE_LENGTH)
        builder_cos, builder_sin = _make_builder_cos_sin(MINISTRAL_3B_CONFIG, CACHE_LENGTH)

        np.testing.assert_allclose(builder_cos, hf_cos, rtol=1e-5, atol=1e-5, err_msg="cos_cache mismatch")
        np.testing.assert_allclose(builder_sin, hf_sin, rtol=1e-5, atol=1e-5, err_msg="sin_cache mismatch")

    def test_bug_a_hasattr_on_dict(self):
        """Bug (a): builder must resolve original_context_length from rope_scaling dict."""
        config = {**MINISTRAL_3B_CONFIG}
        rs = {**config["rope_scaling"]}
        config["rope_scaling"] = rs

        mock_config = types.SimpleNamespace(**config)

        # Verify the dict does NOT respond to hasattr for keys
        assert not hasattr(rs, "original_max_position_embeddings"), "dict should not respond to hasattr for keys"
        assert "original_max_position_embeddings" in rs
        assert isinstance(rs, Mapping)

        original_ctx = rs["original_max_position_embeddings"]
        assert original_ctx == 16384
        assert original_ctx != mock_config.max_position_embeddings

        # Exercise the builder code path and verify it produces correct caches
        hf_cos, hf_sin = _make_hf_reference_cos_sin(config, CACHE_LENGTH)
        builder_cos, builder_sin = _make_builder_cos_sin(config, CACHE_LENGTH)

        np.testing.assert_allclose(
            builder_cos,
            hf_cos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="builder must honor rope_scaling['original_max_position_embeddings']",
        )
        np.testing.assert_allclose(
            builder_sin,
            hf_sin,
            rtol=1e-5,
            atol=1e-5,
            err_msg="builder must honor rope_scaling['original_max_position_embeddings']",
        )

        # Guard: output must differ from what wrong fallback (max_position_embeddings) would give
        wrong_config = {
            **config,
            "rope_scaling": {**rs, "original_max_position_embeddings": config["max_position_embeddings"]},
        }
        wrong_cos, wrong_sin = _make_hf_reference_cos_sin(wrong_config, CACHE_LENGTH)
        assert not np.allclose(builder_cos, wrong_cos, rtol=1e-5, atol=1e-5)
        assert not np.allclose(builder_sin, wrong_sin, rtol=1e-5, atol=1e-5)

    def test_bug_b_rope_theta_fallback(self):
        """Bug (b): builder must resolve rope_theta from rope_scaling when top-level is absent."""
        config = {**MINISTRAL_3B_CONFIG}
        config["rope_scaling"] = dict(MINISTRAL_3B_CONFIG["rope_scaling"])
        mock_config = types.SimpleNamespace(**config)

        # Ministral-3-3B has no top-level rope_theta
        assert not hasattr(mock_config, "rope_theta")
        assert not hasattr(mock_config, "rope_embedding_base")
        assert mock_config.rope_scaling["rope_theta"] == 1000000.0

        # Exercise builder and verify caches match HF reference
        cache_length = 32
        hf_cos, hf_sin = _make_hf_reference_cos_sin(config, cache_length)
        builder_cos, builder_sin = _make_builder_cos_sin(config, cache_length)

        np.testing.assert_allclose(
            builder_cos,
            hf_cos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="builder did not honor rope_scaling['rope_theta'] for cos_cache",
        )
        np.testing.assert_allclose(
            builder_sin,
            hf_sin,
            rtol=1e-5,
            atol=1e-5,
            err_msg="builder did not honor rope_scaling['rope_theta'] for sin_cache",
        )

        # Guard: output must differ from what default theta=10000 would give
        default_theta_config = {**config, "rope_scaling": {**config["rope_scaling"], "rope_theta": 10000.0}}
        default_cos, default_sin = _make_hf_reference_cos_sin(default_theta_config, cache_length)
        assert not np.allclose(builder_cos, default_cos, rtol=1e-5, atol=1e-5), (
            "builder cos_cache unexpectedly matches default theta=10000"
        )
        assert not np.allclose(builder_sin, default_sin, rtol=1e-5, atol=1e-5), (
            "builder sin_cache unexpectedly matches default theta=10000"
        )

    def test_bug_c_mscale_override(self):
        """Bug (c): explicit mscale=1.0 must be respected by make_rope_init()."""
        config = {**MINISTRAL_3B_CONFIG}
        config["rope_scaling"] = dict(MINISTRAL_3B_CONFIG["rope_scaling"])
        mock_config = types.SimpleNamespace(**config)

        model = object.__new__(Model)
        model.rope_attrs = {}
        model.context_length = config["max_position_embeddings"]
        model.original_context_length = config["rope_scaling"]["original_max_position_embeddings"]

        # make_mscale_yarn(factor=16) yields ~1.277, so explicit mscale=1.0
        # must be taken from rope_scaling rather than recomputed.
        computed_mscale = model.make_mscale_yarn(config["rope_scaling"]["factor"])
        assert computed_mscale > 1.2, f"Expected >1.2 from make_mscale_yarn(16), got {computed_mscale}"
        assert config["rope_scaling"]["mscale"] == 1.0

        # Exercise the real make_rope_init code path
        model.make_rope_init(mock_config)

        assert model.rope_attrs["mscale"] == 1.0, f"Expected mscale=1.0, got {model.rope_attrs['mscale']}"

    def test_mscale_fallback_when_absent(self):
        """When 'mscale' key is absent from rope_scaling, fall back to make_mscale(factor)."""
        model = object.__new__(Model)
        model.rope_attrs = {"mscale_policy": "yarn"}
        model.original_context_length = 16384

        # Simulate rope_scaling without 'mscale' key
        rope_scaling_no_mscale = {k: v for k, v in MINISTRAL_3B_CONFIG["rope_scaling"].items() if k != "mscale"}
        config_mscale = rope_scaling_no_mscale.get("mscale", 0)
        assert config_mscale == 0, "mscale should be absent, defaulting to 0"

        # Fallback: compute from factor via make_mscale_yarn
        fallback_mscale = model.make_mscale(rope_scaling_no_mscale["factor"])
        expected = 0.1 * math.log(16.0) + 1.0  # make_mscale_yarn formula
        assert abs(fallback_mscale - expected) < 1e-10, f"Expected {expected}, got {fallback_mscale}"
        assert fallback_mscale > 1.2, f"Computed mscale should be >1.2, got {fallback_mscale}"

    def test_mscale_all_dim_formula(self):
        """When both mscale and mscale_all_dim are provided, use the full HF formula."""
        model = object.__new__(Model)
        model.rope_attrs = {"mscale_policy": "yarn"}

        factor = 16.0
        # When mscale == mscale_all_dim, the ratio is 1.0 (Ministral-3B case)
        result = model.make_mscale(factor, config_mscale=1.0, config_mscale_all_dim=1.0)

        def get_mscale(s, ms):
            return (0.1 * ms * math.log(s) + 1.0) if s > 1 else 1.0

        expected = get_mscale(factor, 1.0) / get_mscale(factor, 1.0)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        assert result == 1.0, "Equal mscale and mscale_all_dim should yield 1.0"

        # When mscale != mscale_all_dim, result differs from raw config_mscale
        result_diff = model.make_mscale(factor, config_mscale=2.0, config_mscale_all_dim=1.0)
        expected_diff = get_mscale(factor, 2.0) / get_mscale(factor, 1.0)
        assert abs(result_diff - expected_diff) < 1e-10, f"Expected {expected_diff}, got {result_diff}"
        assert result_diff != 2.0, "Full formula should differ from raw config_mscale"

        # When only config_mscale is set (mscale_all_dim=0), fall back to direct value
        result_direct = model.make_mscale(factor, config_mscale=2.0, config_mscale_all_dim=0)
        assert result_direct == 2.0, "Without mscale_all_dim, should return config_mscale directly"

    def test_bug_d_inv_freq_no_double_inversion(self):
        """Bug (d): inv_freq must use inv_freq/factor, not 1/(factor*inv_freq)."""
        model = object.__new__(Model)
        model.head_size = 128
        model.original_context_length = 16384
        model.rope_attrs = {
            "theta": 1000000.0,
            "rescale_inv_freq": {
                "factor": 16.0,
                "ntk_alpha": 1.0,  # beta_slow
                "ntk_beta": 32.0,  # beta_fast
            },
        }

        dim = 128
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # Correct: inv_freq / factor (fixed code)
        correct_interpolation = inv_freq / 16.0

        # Wrong: 1/(factor * inv_freq) — double inversion bug
        wrong_interpolation = 1.0 / (16.0 * inv_freq)

        # The correct values should NOT equal the buggy values
        assert not torch.allclose(correct_interpolation, wrong_interpolation), (
            "Correct and buggy interpolation should differ"
        )

        # Verify the builder uses the correct formula
        result = model.make_inv_freq_rescaled_with_ntk(inv_freq)

        # Reference: HF computation
        pos_freqs = 1000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        hf_inv_freq_extrapolation = 1.0 / pos_freqs
        hf_inv_freq_interpolation = 1.0 / (16.0 * pos_freqs)

        d_half = dim / 2
        low = d_half * np.log(16384 / (32.0 * 2 * np.pi)) / np.log(1000000.0)
        high = d_half * np.log(16384 / (1.0 * 2 * np.pi)) / np.log(1000000.0)
        ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        hf_inv_freq = hf_inv_freq_interpolation * (1 - mask) + hf_inv_freq_extrapolation * mask

        np.testing.assert_allclose(
            result.numpy(),
            hf_inv_freq.numpy(),
            rtol=1e-5,
            atol=1e-7,
            err_msg="Builder inv_freq does not match HF reference (double-inversion bug?)",
        )

    def test_full_cache_length(self):
        """Parity check with larger cache length matching typical inference."""
        cache_length = 2048
        hf_cos, hf_sin = _make_hf_reference_cos_sin(MINISTRAL_3B_CONFIG, cache_length)
        builder_cos, builder_sin = _make_builder_cos_sin(MINISTRAL_3B_CONFIG, cache_length)

        np.testing.assert_allclose(builder_cos, hf_cos, rtol=1e-5, atol=1e-5, err_msg="cos_cache mismatch @ 2048")
        np.testing.assert_allclose(builder_sin, hf_sin, rtol=1e-5, atol=1e-5, err_msg="sin_cache mismatch @ 2048")

    def test_mapping_isinstance_with_frozen_dict(self):
        """isinstance(rope_scaling, Mapping) works for dict subclasses and Mapping types."""
        rope_scaling = MINISTRAL_3B_CONFIG["rope_scaling"]

        # Regular dict
        assert isinstance(rope_scaling, Mapping)

        # types.MappingProxyType (immutable/frozen dict-like)
        frozen = types.MappingProxyType(rope_scaling)
        assert isinstance(frozen, Mapping)
        assert not isinstance(frozen, dict)  # Would fail with old isinstance(_, dict) check

        # Verify key access still works
        assert "original_max_position_embeddings" in frozen
        assert frozen["rope_theta"] == 1000000.0

    def test_yarn_no_mscale_cos_sin_match(self):
        """Parity for a YaRN model without explicit mscale (computed from factor)."""
        hf_cos, hf_sin = _make_hf_reference_cos_sin(YARN_NO_MSCALE_CONFIG, CACHE_LENGTH)
        builder_cos, builder_sin = _make_builder_cos_sin(YARN_NO_MSCALE_CONFIG, CACHE_LENGTH)

        np.testing.assert_allclose(
            builder_cos,
            hf_cos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="cos_cache mismatch for YaRN model without explicit mscale",
        )
        np.testing.assert_allclose(
            builder_sin,
            hf_sin,
            rtol=1e-5,
            atol=1e-5,
            err_msg="sin_cache mismatch for YaRN model without explicit mscale",
        )

    def test_yarn_no_mscale_uses_computed_value(self):
        """When mscale is absent, make_mscale computes from factor via make_mscale_yarn."""
        config = YARN_NO_MSCALE_CONFIG
        rs = config["rope_scaling"]

        model = object.__new__(Model)
        model.rope_attrs = {}
        model.context_length = config["max_position_embeddings"]
        model.original_context_length = rs["original_max_position_embeddings"]

        mock_config = types.SimpleNamespace(**config)
        model.make_rope_init(mock_config)

        # mscale should be computed (not 1.0) since config has no explicit mscale
        assert model.rope_attrs["mscale"] > 1.0, (
            f"Expected computed mscale > 1.0 for factor=40, got {model.rope_attrs['mscale']}"
        )
        expected = 0.1 * math.log(40.0) + 1.0  # make_mscale_yarn formula
        assert abs(model.rope_attrs["mscale"] - expected) < 1e-10, (
            f"Expected mscale={expected}, got {model.rope_attrs['mscale']}"
        )

    def test_different_yarn_configs_produce_different_caches(self):
        """Two YaRN configs with different parameters must produce different caches."""
        cos1, sin1 = _make_builder_cos_sin(MINISTRAL_3B_CONFIG, CACHE_LENGTH)
        cos2, sin2 = _make_builder_cos_sin(YARN_NO_MSCALE_CONFIG, CACHE_LENGTH)

        assert not np.allclose(cos1, cos2, rtol=1e-3, atol=1e-3), (
            "Different YaRN configs should produce different cos caches"
        )
        assert not np.allclose(sin1, sin2, rtol=1e-3, atol=1e-3), (
            "Different YaRN configs should produce different sin caches"
        )

    def test_gptoss_20b_cos_sin_match(self):
        """End-to-end parity: builder cos/sin caches match HF reference for GPT-OSS-20B."""
        hf_cos, hf_sin = _make_hf_reference_cos_sin(GPTOSS_20B_CONFIG, CACHE_LENGTH)
        builder_cos, builder_sin = _make_builder_cos_sin(GPTOSS_20B_CONFIG, CACHE_LENGTH)

        np.testing.assert_allclose(
            builder_cos, hf_cos, rtol=1e-5, atol=1e-5, err_msg="cos_cache mismatch for GPT-OSS-20B"
        )
        np.testing.assert_allclose(
            builder_sin, hf_sin, rtol=1e-5, atol=1e-5, err_msg="sin_cache mismatch for GPT-OSS-20B"
        )

    def test_gptoss_20b_top_level_rope_theta(self):
        """GPT-OSS-20B has top-level rope_theta=150000, not inside rope_scaling."""
        config = GPTOSS_20B_CONFIG
        rs = config["rope_scaling"]

        assert "rope_theta" not in rs, "GPT-OSS-20B should NOT have rope_theta inside rope_scaling"
        assert config["rope_theta"] == 150000.0, "GPT-OSS-20B should have top-level rope_theta=150000"

        # Verify the builder resolves top-level rope_theta correctly via _make_builder_cos_sin
        # (which reproduces the full __init__ theta-resolution chain).
        hf_cos, _ = _make_hf_reference_cos_sin(config, 32)
        builder_cos, _ = _make_builder_cos_sin(config, 32)
        np.testing.assert_allclose(
            builder_cos,
            hf_cos,
            rtol=1e-5,
            atol=1e-5,
            err_msg="builder must resolve top-level rope_theta=150000 for GPT-OSS-20B",
        )

        # Guard: verify output differs from default theta=10000
        wrong_config = {**config, "rope_theta": 10000.0}
        wrong_cos, _ = _make_hf_reference_cos_sin(wrong_config, 32)
        assert not np.allclose(builder_cos, wrong_cos, rtol=1e-5, atol=1e-5), (
            "builder cos_cache should differ from default theta=10000"
        )

    def test_gptoss_20b_computed_mscale(self):
        """GPT-OSS-20B has no mscale in rope_scaling — must compute from factor=32."""
        config = GPTOSS_20B_CONFIG
        rs = config["rope_scaling"]

        assert "mscale" not in rs, "GPT-OSS-20B should not have explicit mscale"
        assert "mscale_all_dim" not in rs, "GPT-OSS-20B should not have explicit mscale_all_dim"

        model = object.__new__(Model)
        model.rope_attrs = {}
        model.context_length = config["max_position_embeddings"]
        model.original_context_length = rs["original_max_position_embeddings"]

        mock_config = types.SimpleNamespace(**config)
        model.make_rope_init(mock_config)

        # mscale should be computed via make_mscale_yarn(32)
        expected = 0.1 * math.log(32.0) + 1.0
        assert abs(model.rope_attrs["mscale"] - expected) < 1e-10, (
            f"Expected computed mscale={expected}, got {model.rope_attrs['mscale']}"
        )

    def test_gptoss_20b_full_cache_length(self):
        """Parity check with larger cache length for GPT-OSS-20B."""
        cache_length = 2048
        hf_cos, hf_sin = _make_hf_reference_cos_sin(GPTOSS_20B_CONFIG, cache_length)
        builder_cos, builder_sin = _make_builder_cos_sin(GPTOSS_20B_CONFIG, cache_length)

        np.testing.assert_allclose(
            builder_cos, hf_cos, rtol=1e-5, atol=1e-5, err_msg="cos_cache mismatch for GPT-OSS-20B @ 2048"
        )
        np.testing.assert_allclose(
            builder_sin, hf_sin, rtol=1e-5, atol=1e-5, err_msg="sin_cache mismatch for GPT-OSS-20B @ 2048"
        )
