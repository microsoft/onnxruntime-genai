# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for quantized_model.py lm_head tensor loading.

These tests verify that lm_head tensors are assigned correctly regardless
of the iteration order returned by safetensors.torch.load_file().
"""

from __future__ import annotations

import torch
from onnxruntime_genai.models.quantized_model import (
    QuantizedModel,
    QuantizedTensorModule,
    TensorModule,
)


class _FakeQuantizedModel:
    """Minimal stand-in for QuantizedModel that only exposes the lm_head
    initialisation helpers so we can test _assign_lm_head_tensors in isolation."""

    _LM_HEAD_NAME_MAP = QuantizedModel._LM_HEAD_NAME_MAP
    _assign_lm_head_tensors = QuantizedModel._assign_lm_head_tensors

    def __init__(self):
        self.lm_head = TensorModule()

    def _initialize_quantized_lm_head(self, bits, group_size):
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
        {
            "lm_head.weight": (plain_weight, 4, 128),
        }
    )

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.weight is plain_weight


def test_lm_head_empty_dict_shared_embeddings():
    """No lm_head tensors at all (embedding weights will be shared later)."""
    model = _FakeQuantizedModel()
    model._assign_lm_head_tensors({})

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.weight is None


def test_lm_head_explicit_qweight_key():
    """AWQ/GPTQ style with explicit lm_head.qweight key."""
    model = _FakeQuantizedModel()
    model.lm_head.weight = torch.randn(100352, 2048)
    t = _make_quant_tensors()
    qweight = torch.randint(0, 15, (2048, 12544), dtype=torch.int32)
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
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
    model._assign_lm_head_tensors(
        {
            "lm_head.weight": (t["weight"], 4, 128),
            "lm_head.bias": (bias, 4, 128),
        }
    )

    assert isinstance(model.lm_head, TensorModule)
    assert model.lm_head.bias is bias
