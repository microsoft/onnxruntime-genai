# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Tests for the Gemma 4 builder scaffolding (issue #2062).

These tests avoid importing the full builder stack (which pulls in torch,
transformers, and a recent onnxruntime). They validate the new Gemma 4-specific
hooks (`is_local`, `make_key_value_cache_shape`, unsupported-feature gate) by
loading `builders/gemma4.py` with a stubbed `Gemma3Model` parent so the
test suite can run in a clean Python env.

When you have the full build env set up, also run `builder.py` with
`--extra_options config_only=true` against a Gemma 4 checkpoint to validate
end-to-end config generation - that path is intentionally not mocked here.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_gemma4_with_stubbed_parent():
    """Import `builders/gemma4.py` with a lightweight Gemma3Model stand-in.

    We cannot just `from builders import Gemma4TextModel` because that
    pulls in `base.py` -> onnx_ir + torch + a modern onnxruntime. For the
    tests below we only need the Gemma 4-specific overrides, so we replace
    the `.gemma` module with a shim that exposes a minimal `Gemma3Model`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gemma4_path = repo_root / "src" / "python" / "py" / "models" / "builders" / "gemma4.py"

    fake_pkg = types.ModuleType("gemma4_test_pkg")
    fake_pkg.__path__ = [str(gemma4_path.parent)]
    sys.modules["gemma4_test_pkg"] = fake_pkg

    gemma_shim = types.ModuleType("gemma4_test_pkg.gemma")

    class _StubGemma3Model:
        def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
            self.config = config
            self.io_dtype = io_dtype
            self.onnx_dtype = onnx_dtype
            self.ep = ep
            self.cache_dir = cache_dir
            self.extra_options = extra_options
            self.rope_local_theta = getattr(config, "rope_theta", 10_000.0)
            self.rope_attrs = {"create_caches": False, "partial_rotary_factor": 1.0, "theta": 10_000.0}
            self._rope_cache_calls = []

        def is_local(self, layer_id):
            return bool((layer_id + 1) % 6)

        def make_key_value_cache_shape(self, layer_id, shape):
            # Mirrors the base-class trt-rtx branch so our Gemma 4 override
            # can delegate without special-casing.
            if getattr(self, "ep", None) == "trt-rtx" and self.is_local(layer_id):
                return [shape[0], shape[1], shape[2].replace("sequence", "sliding"), shape[3]]
            return shape

        def make_rotary_embedding_caches(self, **kwargs):
            self._rope_cache_calls.append({
                "cos_cache_name": kwargs.get("cos_cache_name"),
                "sin_cache_name": kwargs.get("sin_cache_name"),
                "theta": self.rope_attrs["theta"],
                "partial_rotary_factor": self.rope_attrs["partial_rotary_factor"],
            })

    gemma_shim.Gemma3Model = _StubGemma3Model
    sys.modules["gemma4_test_pkg.gemma"] = gemma_shim

    spec = importlib.util.spec_from_file_location(
        "gemma4_test_pkg.gemma4", str(gemma4_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Layer pattern matches google/gemma-4-E2B-it (35 layers, every 5th is full)
GEMMA4_E2B_LAYER_TYPES = [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
]


def _make_e2b_config(**overrides):
    cfg = SimpleNamespace(
        layer_types=list(GEMMA4_E2B_LAYER_TYPES),
        head_dim=256,
        global_head_dim=512,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
        rope_theta=10_000.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class Gemma4BuilderScaffoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_gemma4_with_stubbed_parent()
        cls.Gemma4TextModel = cls.module.Gemma4TextModel

    def _make(self, config=None, extra_options=None):
        # We always pass config_only=True so the unsupported-feature gate
        # doesn't fire; each test that wants to exercise the gate flips
        # that flag off explicitly.
        extra_options = {"config_only": True, **(extra_options or {})}
        cfg = config or _make_e2b_config()
        return self.Gemma4TextModel(cfg, None, None, "cpu", "", extra_options)

    # is_local ------------------------------------------------------------

    def test_is_local_uses_layer_types(self):
        model = self._make()
        # Full-attention layers on E2B are at 4, 9, 14, 19, 24, 29, 34.
        full_layers = [4, 9, 14, 19, 24, 29, 34]
        for i in range(35):
            expected_local = i not in full_layers
            self.assertEqual(
                model.is_local(i),
                expected_local,
                f"layer {i}: expected is_local={expected_local}",
            )

    def test_is_local_falls_back_when_layer_types_missing(self):
        cfg = _make_e2b_config(layer_types=[])
        model = self._make(config=cfg)
        # Should defer to the stub Gemma 3 every-6th rule.
        self.assertTrue(model.is_local(0))
        self.assertFalse(model.is_local(5))

    # make_key_value_cache_shape -----------------------------------------

    def test_kv_cache_shape_uses_sliding_head_dim_on_sliding_layers(self):
        model = self._make()
        base = ["batch", 1, "past_seq", 256]
        self.assertEqual(
            model.make_key_value_cache_shape(0, base),
            ["batch", 1, "past_seq", 256],
        )

    def test_kv_cache_shape_uses_full_head_dim_on_full_layers(self):
        model = self._make()
        base = ["batch", 1, "past_seq", 256]
        # Layers 4, 9, 14, ... should be full => head_dim 512.
        for layer_id in (4, 9, 14, 19, 24, 29, 34):
            with self.subTest(layer_id=layer_id):
                shape = model.make_key_value_cache_shape(layer_id, base)
                self.assertEqual(shape[3], 512, f"layer {layer_id} should use global_head_dim=512")

    def test_kv_cache_shape_preserves_non_head_dim_axes(self):
        model = self._make()
        base = ["b", 4, "past_sequence_length", 256]
        shape = model.make_key_value_cache_shape(0, base)
        self.assertEqual(shape[:3], ["b", 4, "past_sequence_length"])

    # Unsupported-feature gate -------------------------------------------

    def test_gate_raises_on_ple(self):
        cfg = _make_e2b_config(num_kv_shared_layers=0, global_head_dim=256)
        with self.assertRaises(NotImplementedError) as cm:
            self._make(config=cfg, extra_options={"config_only": False})
        self.assertIn("Per-Layer Embeddings", str(cm.exception))

    def test_gate_raises_on_kv_sharing(self):
        cfg = _make_e2b_config(hidden_size_per_layer_input=0, global_head_dim=256)
        with self.assertRaises(NotImplementedError) as cm:
            self._make(config=cfg, extra_options={"config_only": False})
        self.assertIn("KV cache sharing", str(cm.exception))

    def test_gate_raises_on_variable_head_dim(self):
        cfg = _make_e2b_config(hidden_size_per_layer_input=0, num_kv_shared_layers=0)
        with self.assertRaises(NotImplementedError) as cm:
            self._make(config=cfg, extra_options={"config_only": False})
        self.assertIn("Variable attention head dimension", str(cm.exception))

    def test_gate_quiet_in_config_only_mode(self):
        # The whole point of the scaffold: config_only must still succeed so
        # downstream tooling can generate genai_config.json.
        model = self._make(extra_options={"config_only": True})
        self.assertIsNotNone(model)

    # RoPE cache wiring --------------------------------------------------

    def test_rope_caches_are_emitted_for_both_attention_types(self):
        model = self._make()
        model.make_rotary_embedding_multi_cache()
        calls = model._rope_cache_calls
        self.assertEqual(len(calls), 2, "expected one cache each for full and sliding RoPE")

        global_call = next(c for c in calls if c["cos_cache_name"] == "cos_cache_global")
        local_call = next(c for c in calls if c["cos_cache_name"] == "cos_cache_local")

        self.assertEqual(global_call["theta"], 1_000_000.0)
        self.assertEqual(global_call["partial_rotary_factor"], 0.25)
        self.assertEqual(local_call["theta"], 10_000.0)
        self.assertEqual(local_call["partial_rotary_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
