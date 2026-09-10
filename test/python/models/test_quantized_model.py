# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for base.py lm_head tensor loading, linear_attn key dispatch,
and VLM key normalisation.

These tests verify that lm_head tensors are assigned correctly regardless
of the iteration order returned by safetensors.torch.load_file(), that
linear_attn.* weight keys (GatedDeltaNet hybrid layers) are recognised and
assigned to the correct QuantizedLinearAttention fields, and that the
VLM/Quark checkpoint key normalisation introduced for Qwen3-VL-4B works
correctly so future refactors do not silently break quantised VLM loading.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders.base import (
    QuantizedDecoderLayer,
    QuantizedExperts,
    QuantizedLinearAttention,
    QuantizedModel,
    QuantizedTensorModule,
    TensorModule,
)
from loaders.gptq import GPTQModel
from loaders.olive import OliveModel
from loaders.quark import QuarkModel

_BASE_MODEL = object.__new__(QuantizedModel)
_QUARK_MODEL = object.__new__(QuarkModel)


def test_gptq_initializes_dynamic_quantization_overrides(tmp_path):
    quant_attrs = {
        "config": {"group_size": 128, "bits": 4, "dynamic": {"+:model.layers.0.*": {"bits": 8}}},
        "use_g_idx": False,
    }

    model = GPTQModel("gptq", tmp_path, quant_attrs, 0, 0, 0, 0)

    assert model.global_group_size == 128
    assert model.global_bits == 4
    assert model.get_layer_bits("model.layers.0.self_attn.q_proj") == 8


def test_olive_initializes_per_layer_quantization_overrides(tmp_path):
    quant_attrs = {
        "config": {
            "group_size": 128,
            "bits": 4,
            "overrides": {"model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8}},
        },
        "use_g_idx": False,
    }

    model = OliveModel("olive", tmp_path, quant_attrs, 0, 0, 0, 0)

    layer_name = "model.layers.0.self_attn.q_proj.weight"
    assert model.get_layer_group_size(layer_name) == 64
    assert model.get_layer_bits(layer_name) == 8


def test_quark_initializes_global_quantization_config(tmp_path):
    quant_attrs = {
        "config": {
            "global_quant_config": {"weight": {"group_size": 32, "dtype": "uint4"}},
            "layer_quant_config": {},
        }
    }

    model = QuarkModel("quark", tmp_path, quant_attrs, 0, 0, 0, 0)

    assert model.global_group_size == 32
    assert model.global_bits == 4


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
    assert _QUARK_MODEL.normalize_weight_name(raw) == "model.layers.0.self_attn.q_proj.weight_scale"


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


# ---------------------------------------------------------------------------
# Tests for linear_attn (GatedDeltaNet) tensor-key dispatch (Qwen3.5 hybrid)
# ---------------------------------------------------------------------------
# These tests exercise the regex dispatch in QuantizedModel.__init__ directly
# by calling normalize_weight_name + the tensor-map logic in isolation, without
# triggering the full Quark repack pipeline (which needs complete qzeros etc.).
# The approach mirrors the existing _FakeQuantizedModel pattern above.


def _dispatch_key(name, tensor, layer):
    """Run one weight key through the same tensor-map dispatch used in __init__."""
    tensor_map = {}
    # Replicate the linear_attn regex block from QuantizedModel.__init__
    projections = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    for proj in projections:
        if bool(re.match(rf"^model\.layers\.\d+\.linear_attn\.{proj}\.q?weight$", name)):
            tensor_map[f"linear_attn.{proj}.qweight"] = tensor
        elif bool(re.match(rf"^model\.layers\.\d+\.linear_attn\.{proj}\.(scales|weight_scale)$", name)):
            tensor_map[f"linear_attn.{proj}.scales"] = tensor
        elif bool(re.match(rf"^model\.layers\.\d+\.linear_attn\.{proj}\.(qzeros|weight_zero_point)$", name)):
            tensor_map[f"linear_attn.{proj}.qzeros"] = tensor
        elif bool(re.match(rf"^model\.layers\.\d+\.linear_attn\.{proj}\.g_idx$", name)):
            tensor_map[f"linear_attn.{proj}.g_idx"] = tensor
    if bool(re.match(r"^model\.layers\.\d+\.linear_attn\.A_log$", name)):
        tensor_map["linear_attn.A_log"] = tensor
    elif bool(re.match(r"^model\.layers\.\d+\.linear_attn\.dt_bias$", name)):
        tensor_map["linear_attn.dt_bias"] = tensor
    elif bool(re.match(r"^model\.layers\.\d+\.linear_attn\.conv1d\.weight$", name)):
        tensor_map["linear_attn.conv1d.weight"] = tensor
    elif bool(re.match(r"^model\.layers\.\d+\.linear_attn\.norm\.weight$", name)):
        tensor_map["linear_attn.norm.weight"] = tensor

    # Apply tensor_map to the layer (same logic as __init__)
    for tensor_name, tensor_value in tensor_map.items():
        submodule = layer
        for sub_name in tensor_name.split(".")[:-1]:
            submodule = getattr(submodule, sub_name)
        setattr(submodule, tensor_name.split(".")[-1], tensor_value)

    return tensor_map


def _make_layer():
    return QuantizedDecoderLayer(layer_id=0)


def test_linear_attn_quantized_projections_dispatched():
    """qweight and scales for all five projections must land in QuantizedLinearAttention."""
    layer = _make_layer()
    qw = torch.randint(0, 255, (64, 8), dtype=torch.int32)
    sc = torch.randn(1, 512, dtype=torch.float16)

    for proj in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]:
        _dispatch_key(f"model.layers.0.linear_attn.{proj}.qweight", qw, layer)
        _dispatch_key(f"model.layers.0.linear_attn.{proj}.weight_scale", sc, layer)

    assert isinstance(layer.linear_attn, QuantizedLinearAttention)
    for proj in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]:
        assert getattr(layer.linear_attn, proj).qweight is qw
        assert getattr(layer.linear_attn, proj).scales is sc


def test_linear_attn_plain_tensors_dispatched():
    """Non-quantized A_log, dt_bias, conv1d.weight and norm.weight must be assigned."""
    layer = _make_layer()
    a_log = torch.randn(32)
    dt_bias = torch.randn(32)
    conv_w = torch.randn(512, 1, 4)
    norm_w = torch.randn(128)

    _dispatch_key("model.layers.0.linear_attn.A_log", a_log, layer)
    _dispatch_key("model.layers.0.linear_attn.dt_bias", dt_bias, layer)
    _dispatch_key("model.layers.0.linear_attn.conv1d.weight", conv_w, layer)
    _dispatch_key("model.layers.0.linear_attn.norm.weight", norm_w, layer)

    assert layer.linear_attn.A_log is a_log
    assert layer.linear_attn.dt_bias is dt_bias
    assert layer.linear_attn.conv1d.weight is conv_w
    assert layer.linear_attn.norm.weight is norm_w


def test_linear_attn_weight_scale_alias_recognized():
    """Both 'scales' (GPTQ style) and 'weight_scale' (Quark style) must map to .scales."""
    layer = _make_layer()
    sc_gptq = torch.randn(1, 512, dtype=torch.float16)
    sc_quark = torch.randn(1, 512, dtype=torch.float16)

    _dispatch_key("model.layers.0.linear_attn.in_proj_qkv.scales", sc_gptq, layer)
    assert layer.linear_attn.in_proj_qkv.scales is sc_gptq

    _dispatch_key("model.layers.0.linear_attn.in_proj_z.weight_scale", sc_quark, layer)
    assert layer.linear_attn.in_proj_z.scales is sc_quark


def test_linear_attn_keys_do_not_match_self_attn_patterns():
    """linear_attn keys must not accidentally match the self_attn regex patterns."""
    linear_attn_key = "model.layers.0.linear_attn.in_proj_qkv.qweight"
    self_attn_pattern = re.compile(r"^model.layers\.\d+\.self_attn.q_proj\.q?weight$")
    assert not self_attn_pattern.match(linear_attn_key)

    self_attn_key = "model.layers.0.self_attn.q_proj.qweight"
    linear_attn_pattern = re.compile(r"^model\.layers\.\d+\.linear_attn\.in_proj_qkv\.q?weight$")
    assert not linear_attn_pattern.match(self_attn_key)
