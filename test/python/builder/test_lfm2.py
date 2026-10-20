# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the dense LFM2 model builder.

LFM2 configs call their RMSNorm epsilon `norm_eps` and have no `rms_norm_eps`, so the base builder's
epsilon falls back to 1e-6 and `LFM2Model` must set the checkpoint's value. Standalone Q/K norm nodes take
the layernorm epsilon; a GroupQueryAttention node that fuses the Q/K norms (CUDA and WebGPU) carries its
own `qk_norm_epsilon` attribute.
"""

from __future__ import annotations

import re

import pytest
from _builder_test_utils import load_builder_module

lfm2_module = load_builder_module("lfm2")
ir = lfm2_module.ir
LFM2Model = lfm2_module.LFM2Model

transformers = pytest.importorskip("transformers")
# `_builder_test_utils` substitutes a stub for transformers when it is missing, which has none of the
# LFM2 classes these tests build a model from.
pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "Lfm2ForCausalLM"), reason="transformers does not provide the LFM2 model classes"
)


def _tiny_lfm2_config():
    return transformers.Lfm2Config(
        architectures=["Lfm2ForCausalLM"],
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        # Neither LFM2's usual 1e-5 nor the builder's 1e-6 fallback, so no default can match it by chance.
        norm_eps=1e-3,
    )


def _qk_norm_epsilons(graph):
    """The epsilon of every Q/K norm in the graph, keyed by the name of the node that applies it."""
    epsilons = {}
    for node in graph:
        if node.op_type == "GroupQueryAttention" and "qk_norm_epsilon" in node.attributes:
            epsilons[node.name] = node.attributes["qk_norm_epsilon"].value
        elif node.op_type == "SimplifiedLayerNormalization" and re.search("/attn/[qk]_norm/", node.name):
            epsilons[node.name] = node.attributes["epsilon"].value
    return epsilons


@pytest.mark.parametrize(
    "ep, io_dtype, norm_nodes",
    [
        # CPU keeps the Q/K norms as separate nodes, which take the layernorm epsilon.
        ("cpu", ir.DataType.FLOAT, ["q_norm/SimplifiedLayerNormalization", "k_norm/SimplifiedLayerNormalization"]),
        # CUDA and WebGPU fuse both into GroupQueryAttention, which has an epsilon attribute of its own.
        ("cuda", ir.DataType.FLOAT16, ["GroupQueryAttention"]),
        ("webgpu", ir.DataType.FLOAT, ["GroupQueryAttention"]),
    ],
)
def test_lfm2_qk_norms_use_the_checkpoint_norm_eps(ep, io_dtype, norm_nodes):
    config = _tiny_lfm2_config()
    hf_model = transformers.Lfm2ForCausalLM(config).eval()

    model = LFM2Model(config, io_dtype, io_dtype, ep, None, {})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")

    attention_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
    expected = {f"/model/layers.{i}/attn/{node}": config.norm_eps for i in attention_layers for node in norm_nodes}
    assert _qk_norm_epsilons(model.model.graph) == pytest.approx(expected)
