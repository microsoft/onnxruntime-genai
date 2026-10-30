# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys

import onnx_ir as ir
from transformers import MistralConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.mistral import MistralModel


def make_mistral_builder(**config_overrides):
    sliding_window = config_overrides.pop("sliding_window", 8)
    config = MistralConfig(
        architectures=["MistralForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=sliding_window,
        **config_overrides,
    )
    return MistralModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})


def test_mistral_without_layer_types_uses_its_sliding_window():
    model = make_mistral_builder()

    assert model.window_size == 8
    assert model.layer_types == ["sliding_attention", "sliding_attention"]
    assert all(model.is_local(layer_id) for layer_id in range(model.num_layers))


def test_mistral_respects_explicit_layer_types():
    model = make_mistral_builder(layer_types=["sliding_attention", "full_attention"])

    assert model.layer_types == ["sliding_attention", "full_attention"]
    assert model.is_local(0)
    assert not model.is_local(1)


def test_mistral_without_window_remains_full_attention():
    model = make_mistral_builder(sliding_window=None)

    assert model.layer_types == ["full_attention", "full_attention"]
    assert not any(model.is_local(layer_id) for layer_id in range(model.num_layers))
