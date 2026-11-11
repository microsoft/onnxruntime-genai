# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Configs without ``layer_types`` must keep their ``sliding_window`` on every layer (issue #2832)."""

from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

base = load_builder_module("base")
mistral = load_builder_module("mistral")
phi = load_builder_module("phi")
qwen = load_builder_module("qwen")

SMALL_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}


@pytest.mark.parametrize(
    "config, expected",
    [
        (SimpleNamespace(sliding_window=4096), "sliding_attention"),
        (SimpleNamespace(sliding_window=None), "full_attention"),
        (SimpleNamespace(sliding_window=0), "full_attention"),
        (SimpleNamespace(), "full_attention"),
        (SimpleNamespace(sliding_window=4096, use_sliding_window=False), "full_attention"),
        (SimpleNamespace(sliding_window=4096, use_sliding_window=True), "sliding_attention"),
    ],
)
def test_default_layer_type(config, expected):
    assert base.Model.__new__(base.Model).default_layer_type(config) == expected


def transformers_config(class_name, architecture, **kwargs):
    transformers = pytest.importorskip("transformers")
    # _builder_test_utils installs an importable transformers stub when the package is absent.
    if not hasattr(transformers, class_name):
        pytest.skip(f"transformers does not provide {class_name}")
    config = getattr(transformers, class_name)(**SMALL_CONFIG, **kwargs)
    config.architectures = [architecture]
    return config


def local_window_sizes(model):
    # Run make_attention with its subgraph builders stubbed out and record the window each layer is built with.
    sizes = []
    model.make_attention_input_proj = lambda *args, **kwargs: None
    model.make_attention_qk_subgraph = lambda *args, **kwargs: sizes.append(
        model.get_attention_op_attributes()["local_window_size"]
    )
    model.make_attention_output_proj = lambda *args, **kwargs: None
    for layer_id in range(model.num_layers):
        model.make_attention(layer_id, attention=None, root_input="")
    return sizes


def build(model_class, config, tmp_path):
    return model_class(config, 1, 1, "cpu", str(tmp_path), {})


def test_mistral_sliding_window_applies_to_every_layer(tmp_path):
    config = transformers_config("MistralConfig", "MistralForCausalLM", sliding_window=4)
    model = build(mistral.MistralModel, config, tmp_path)

    assert model.layer_types == ["sliding_attention", "sliding_attention"]
    assert local_window_sizes(model) == [4, 4]


def test_mistral_without_sliding_window_uses_full_attention(tmp_path):
    config = transformers_config("MistralConfig", "MistralForCausalLM", sliding_window=None)
    model = build(mistral.MistralModel, config, tmp_path)

    assert model.layer_types == ["full_attention", "full_attention"]
    assert local_window_sizes(model) == [-1, -1]


def test_phi3_sliding_window_applies_to_every_layer(tmp_path):
    config = transformers_config("Phi3Config", "Phi3ForCausalLM", sliding_window=8)
    model = build(phi.Phi3MiniModel, config, tmp_path)

    assert model.layer_types == ["sliding_attention", "sliding_attention"]
    assert local_window_sizes(model) == [8, 8]


def test_qwen2_with_sliding_window_disabled_uses_full_attention(tmp_path):
    config = transformers_config("Qwen2Config", "Qwen2ForCausalLM", sliding_window=4, use_sliding_window=False)
    model = build(qwen.QwenModel, config, tmp_path)

    assert model.layer_types == ["full_attention", "full_attention"]
    assert local_window_sizes(model) == [-1, -1]
