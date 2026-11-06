# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""The base builder must read the norm epsilon under every name a config uses for it.

Most configs call it `rms_norm_eps`, but Phi uses `layer_norm_eps`, Phi-3-small `layer_norm_epsilon`, Nemotron
`norm_eps` and ChatGLM `layernorm_epsilon`. When the builder only reads `rms_norm_eps`, these models are exported
with the 1e-6 fallback instead of their own epsilon (1e-5 in the released checkpoints).
"""

from __future__ import annotations

import types

import pytest
from _builder_test_utils import load_builder_module

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not all(hasattr(transformers, name) for name in ("LlamaConfig", "NemotronConfig", "PhiConfig", "PhiForCausalLM")),
    reason="transformers does not provide the model classes these tests build",
)

base_module = load_builder_module("base")
ir = base_module.ir
Model = base_module.Model
NemotronModel = load_builder_module("nemotron").NemotronModel
PhiModel = load_builder_module("phi").PhiModel

# Neither the usual 1e-5 nor the builder's 1e-6 fallback, so no default can match it by chance.
EPSILON = 1e-3

TINY_SIZES = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 64,
}


def _llama_config_with(**norm_attrs):
    """A Llama config whose only norm epsilon attributes are `norm_attrs`."""
    # Transformers 5 configs keep class-level defaults, so `rms_norm_eps` cannot be deleted from an instance.
    config = transformers.LlamaConfig(architectures=["LlamaForCausalLM"], **TINY_SIZES).to_dict()
    del config["rms_norm_eps"]
    return types.SimpleNamespace(**config, **norm_attrs)


def _builder_epsilon(builder_class, config):
    return builder_class(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {}).layernorm_attrs["epsilon"]


@pytest.mark.parametrize(
    "builder_class, make_config",
    [
        pytest.param(
            PhiModel,
            lambda: transformers.PhiConfig(architectures=["PhiForCausalLM"], layer_norm_eps=EPSILON, **TINY_SIZES),
            id="phi-layer_norm_eps",
        ),
        pytest.param(
            NemotronModel,
            lambda: transformers.NemotronConfig(architectures=["NemotronForCausalLM"], norm_eps=EPSILON, **TINY_SIZES),
            id="nemotron-norm_eps",
        ),
        # Phi-3-small and ChatGLM ship remote-code configs, so a Llama config carries their attribute names.
        pytest.param(Model, lambda: _llama_config_with(layer_norm_epsilon=EPSILON), id="layer_norm_epsilon"),
        pytest.param(Model, lambda: _llama_config_with(layernorm_epsilon=EPSILON), id="layernorm_epsilon"),
    ],
)
def test_epsilon_is_read_from_the_config(builder_class, make_config):
    config = make_config()
    assert not hasattr(config, "rms_norm_eps")
    assert _builder_epsilon(builder_class, config) == EPSILON


def test_rms_norm_eps_takes_precedence():
    config = _llama_config_with(rms_norm_eps=1e-4, layer_norm_eps=EPSILON)
    assert _builder_epsilon(Model, config) == 1e-4


def test_epsilon_falls_back_when_the_config_has_none():
    assert _builder_epsilon(Model, _llama_config_with()) == 1e-06


def test_phi_layernorm_nodes_use_the_config_epsilon():
    config = transformers.PhiConfig(architectures=["PhiForCausalLM"], layer_norm_eps=EPSILON, **TINY_SIZES)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = transformers.PhiForCausalLM(config).eval()

    model = PhiModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")

    epsilons = {
        node.name: node.attributes["epsilon"].value
        for node in model.model.graph
        if node.op_type in ("LayerNormalization", "SkipLayerNormalization")
    }
    # Two decoder layers each have an input LayerNorm, and the final LayerNorm follows them.
    assert len(epsilons) == config.num_hidden_layers + 1
    assert epsilons == pytest.approx(dict.fromkeys(epsilons, EPSILON))
