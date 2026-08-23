# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for base.py lm_head tensor loading and VLM key normalisation.

These tests verify that lm_head tensors are assigned correctly regardless
of the iteration order returned by safetensors.torch.load_file(), and that
the VLM/Quark checkpoint key normalisation introduced for Qwen3-VL-4B works
correctly so future refactors do not silently break quantised VLM loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders.base import (
    QuantizedExperts,
    QuantizedModel,
    QuantizedTensorModule,
    TensorModule,
)
from loaders.quark import QuarkModel

_BASE_MODEL = object.__new__(QuantizedModel)
_QUARK_MODEL = object.__new__(QuarkModel)


def test_quark_finalizes_generic_packed_experts():
    experts = QuantizedExperts()
    for expert_id in range(2):
        expert = experts.add_expert(expert_id)
        expert.gate_proj.qweight = torch.zeros(3, 4, dtype=torch.uint8)
        expert.gate_proj.group_size = 32
        expert.gate_proj.bias = torch.tensor([1.0, 2.0, 3.0]) + expert_id
        expert.up_proj.qweight = torch.zeros(3, 4, dtype=torch.uint8)
        expert.up_proj.bias = torch.tensor([4.0, 5.0, 6.0]) + expert_id
        expert.down_proj.qweight = torch.zeros(2, 3, dtype=torch.uint8)
        expert.down_proj.bias = torch.tensor([7.0, 8.0]) + expert_id
    experts.fc1_weights = torch.zeros(2, 6, 2, dtype=torch.uint8)
    experts.fc1_scales = torch.ones(2, 6, 1)
    experts.fc1_zero_points = torch.zeros(2, 6, 1, dtype=torch.uint8)
    experts.fc2_weights = torch.zeros(2, 2, 2, dtype=torch.uint8)
    experts.fc2_scales = torch.ones(2, 2, 1)
    experts.fc2_zero_points = torch.zeros(2, 2, 1, dtype=torch.uint8)

    _QUARK_MODEL.finalize_packed_experts(experts)

    assert experts.quant_type == "int"
    assert experts.block_size == 32
    assert experts.gate_up_qweight is experts.fc1_weights
    assert experts.down_zero_points is experts.fc2_zero_points
    assert torch.equal(experts.gate_up_bias[0], torch.tensor([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]))
    assert torch.equal(experts.down_bias[1], torch.tensor([8.0, 9.0]))


class _FakeQuantizedModel:
    """Minimal stand-in for QuantizedModel that only exposes the lm_head
    initialisation helpers so we can test assign_lm_head_tensors in isolation."""

    assign_lm_head_tensors = QuantizedModel.assign_lm_head_tensors

    def __init__(self):
        self.lm_head = TensorModule()

    def initialize_quantized_lm_head(self, bits, group_size):
        if not isinstance(self.lm_head, QuantizedTensorModule):
            q = QuantizedTensorModule()
            q.qweight = self.lm_head.weight
            q.bias = self.lm_head.bias
            q.bits = bits
            q.group_size = group_size
            self.lm_head = q


def _make_quant_tensors():
    return {
        "weight": torch.randint(0, 15, (2048, 12544), dtype=torch.int32),
        "scales": torch.randn(16, 100352, dtype=torch.float32),
        "zeros": torch.randint(0, 15, (16, 12544), dtype=torch.int32),
    }


def test_lm_head_scales_before_weight():
    """The original bug: weight_scale iterated before weight causes qweight=None."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    model.assign_lm_head_tensors(
        {
            "lm_head.weight_scale": (t["scales"], 4, 128),
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.weight_zero_point": (t["zeros"], 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.qweight is t["weight"]
    assert model.lm_head.scales is t["scales"]
    assert model.lm_head.qzeros is t["zeros"]


def test_lm_head_weight_before_scales():
    """Normal ordering: weight comes first."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    model.assign_lm_head_tensors(
        {
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.weight_scale": (t["scales"], 4, 128),
            "lm_head.weight_zero_point": (t["zeros"], 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.qweight is t["weight"]
    assert model.lm_head.scales is t["scales"]
    assert model.lm_head.qzeros is t["zeros"]


def test_lm_head_transformer_output_layer_names():
    """ChatGLM uses transformer.output_layer.* instead of lm_head.*."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    model.assign_lm_head_tensors(
        {
            "transformer.output_layer.weight_scale": (t["scales"], 4, 128),
            "transformer.output_layer.weight": (t["weight"], 4, 128),
            "transformer.output_layer.weight_zero_point": (t["zeros"], 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.qweight is t["weight"]
    assert model.lm_head.scales is t["scales"]
    assert model.lm_head.qzeros is t["zeros"]


def test_lm_head_non_quantized():
    """When only lm_head.weight is present (no quant params), stays as TensorModule."""
    model = _FakeQuantizedModel()
    plain_weight = torch.randn(100352, 2048)
    model.assign_lm_head_tensors(
        {
            "lm_head.weight": (plain_weight, 4, 128),
        }
    )

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.weight is plain_weight


def test_lm_head_empty_dict_shared_embeddings():
    """No lm_head tensors at all (embedding weights will be shared later)."""
    model = _FakeQuantizedModel()
    model.assign_lm_head_tensors({})

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.weight is None


def test_lm_head_explicit_qweight_key():
    """AWQ/GPTQ style with explicit lm_head.qweight key."""
    model = _FakeQuantizedModel()
    model.lm_head.weight = torch.randn(100352, 2048)
    t = _make_quant_tensors()
    qweight = torch.randint(0, 15, (2048, 12544), dtype=torch.int32)
    model.assign_lm_head_tensors(
        {
            "lm_head.qweight": (qweight, 4, 128),
            "lm_head.scales": (t["scales"], 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.qweight is qweight
    assert model.lm_head.scales is t["scales"]


def test_lm_head_qweight_and_weight_both_present():
    """If both weight and qweight are present, qweight wins (written second)."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    qweight = torch.randint(0, 15, (2048, 12544), dtype=torch.int32)
    model.assign_lm_head_tensors(
        {
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.qweight": (qweight, 4, 128),
            "lm_head.scales": (t["scales"], 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.qweight is qweight


def test_lm_head_g_idx_assigned():
    """Verify g_idx is correctly assigned when present."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    g_idx = torch.arange(2048, dtype=torch.int32)
    model.assign_lm_head_tensors(
        {
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.scales": (t["scales"], 4, 128),
            "lm_head.g_idx": (g_idx, 4, 128),
        }
    )

    assert isinstance(model.lm_head, QuantizedTensorModule)
    assert model.lm_head.g_idx is g_idx


def test_lm_head_bits_and_group_size():
    """Verify bits and group_size are set on the QuantizedTensorModule."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    model.assign_lm_head_tensors(
        {
            "lm_head.weight_scale": (t["scales"], 4, 128),
            "lm_head.weight": (t["weight"], 4, 128),
        }
    )

    assert model.lm_head.bits == 4
    assert model.lm_head.group_size == 128


def test_lm_head_bias_assigned():
    """Verify bias is correctly assigned."""
    model = _FakeQuantizedModel()
    t = _make_quant_tensors()
    bias = torch.randn(100352)
    model.assign_lm_head_tensors(
        {
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.bias": (bias, 4, 128),
        }
    )

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.bias is bias


# ---------------------------------------------------------------------------
# Regression tests for VLM / Quark checkpoint key normalisation (Qwen3-VL-4B)
# ---------------------------------------------------------------------------


def test_normalize_weight_name_skips_vision_keys():
    """Vision-tower tensors must be filtered out (return None)."""
    assert _BASE_MODEL.normalize_weight_name("model.visual.patch_embed.weight") is None
    assert _BASE_MODEL.normalize_weight_name("model.vision.encoder.layer.0.weight") is None
    assert _BASE_MODEL.normalize_weight_name("visual.embed.weight") is None


def test_normalize_weight_name_keeps_non_vision_keys():
    """Non-vision keys that do not match any normalisation rule pass through unchanged."""
    assert _BASE_MODEL.normalize_weight_name("model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert _BASE_MODEL.normalize_weight_name("lm_head.weight") == "lm_head.weight"
    assert _BASE_MODEL.normalize_weight_name("model.norm.weight") == "model.norm.weight"


def test_normalize_weight_name_strips_language_model_prefix():
    """'model.language_model.*' must be rewritten to 'model.*'."""
    assert _BASE_MODEL.normalize_weight_name("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert (
        _BASE_MODEL.normalize_weight_name("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert _BASE_MODEL.normalize_weight_name("model.language_model.norm.weight") == "model.norm.weight"


def test_quark_normalize_weight_name_renames_scale():
    """Quark '.weight_quantizer.scale' must map to '.weight_scale'."""
    raw = "model.layers.0.self_attn.q_proj.weight_quantizer.scale"
    assert _BASE_MODEL.normalize_weight_name(raw) == raw
    assert (
        _QUARK_MODEL.normalize_weight_name(raw)
        == "model.layers.0.self_attn.q_proj.weight_scale"
    )


def test_quark_normalize_weight_name_renames_zero_point():
    """Quark '.weight_quantizer.zero_point' must map to '.weight_zero_point'."""
    assert (
        _QUARK_MODEL.normalize_weight_name("model.layers.0.mlp.gate_proj.weight_quantizer.zero_point")
        == "model.layers.0.mlp.gate_proj.weight_zero_point"
    )


def test_quark_normalize_weight_name_combines_vlm_prefix_and_quark():
    """VLM prefix stripping and Quark renaming must compose correctly."""
    raw = "model.language_model.layers.2.self_attn.v_proj.weight_quantizer.scale"
    expected = "model.layers.2.self_attn.v_proj.weight_scale"
    assert _QUARK_MODEL.normalize_weight_name(raw) == expected
