# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Unit tests for the Gemma4MoEModel builder (gemma-4-26B-A4B-it text component).

Gemma4MoEModel subclasses the dense Gemma4Model and adds only the MoE FFN, so
these tests focus on the MoE-specific structure without loading weights:
  - moe_attrs: experts/top_k mapped from config, QMoE under int4, geglu +
    swiglu_fusion=1 + normalize_routing_weights (fused-path activation choice).
  - Each layer builds a parallel dense-MLP + MoE FFN combined by an Add.
  - The router pre-projection subgraph (scaleless RMSNorm -> scale -> proj) feeds
    logits to the fused MoE op.
  - per_expert_scale is folded into the expert down_proj initializer offline.

Run with:
    python -m pytest test/python/models/test_gemma4moe_builder.py -v --test_models <any>
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import onnx_ir as ir
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.gemma import Gemma4MoEModel

LAYER_TYPES = ["sliding_attention", "full_attention"]


def _make_minimal_config(**overrides):
    layer_types = overrides.pop("layer_types", LAYER_TYPES)
    cfg = types.SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4_text",
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
        num_hidden_layers=len(layer_types),
        intermediate_size=128,  # dense MLP intermediate
        moe_intermediate_size=32,  # MoE expert intermediate
        num_experts=8,
        top_k_experts=2,
        vocab_size=128,
        max_position_embeddings=128,
        hidden_activation="gelu_pytorch_tanh",
        head_dim=16,
        global_head_dim=32,
        sliding_window=32,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        enable_moe_block=True,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        _name_or_path="",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_model(onnx_dtype=ir.DataType.FLOAT, **overrides):
    return Gemma4MoEModel(
        _make_minimal_config(**overrides),
        io_dtype=ir.DataType.FLOAT,
        onnx_dtype=onnx_dtype,
        ep="cpu",
        cache_dir=None,
        extra_options={},
    )


def _mock_moe_layer(hidden, num_experts, moe_inter, dense_inter):
    def lin(o, i):
        return types.SimpleNamespace(weight=torch.randn(o, i), bias=None)

    def ln():
        return types.SimpleNamespace(weight=torch.randn(hidden))

    return types.SimpleNamespace(
        router=types.SimpleNamespace(
            proj=lin(num_experts, hidden),
            scale=torch.randn(hidden),
            per_expert_scale=torch.rand(num_experts) + 0.5,
        ),
        experts=types.SimpleNamespace(
            gate_up_proj=torch.randn(num_experts, 2 * moe_inter, hidden),
            down_proj=torch.randn(num_experts, hidden, moe_inter),
        ),
        post_feedforward_layernorm_1=ln(),
        post_feedforward_layernorm_2=ln(),
        pre_feedforward_layernorm_2=ln(),
    ), types.SimpleNamespace(
        gate_proj=lin(dense_inter, hidden), up_proj=lin(dense_inter, hidden), down_proj=lin(hidden, dense_inter)
    )


class TestGemma4MoEModel:
    def test_moe_attrs_mapping(self):
        """num_experts/top_k come from config; activation/fusion/normalize set for fused path."""
        m = _make_model()
        assert m.moe_attrs["num_experts"] == 8
        assert m.moe_attrs["top_k"] == 2
        assert m.moe_attrs["activation_type"] == "geglu"
        assert m.moe_attrs["swiglu_fusion"] == 1
        assert m.moe_attrs["normalize_routing_weights"] is True

    def test_qmoe_under_int4(self):
        """int4 build selects the fused QMoE op; float build uses MoE."""
        assert _make_model(onnx_dtype=ir.DataType.INT4).moe_attrs["moe_op_type"] == "QMoE"
        assert _make_model(onnx_dtype=ir.DataType.FLOAT).moe_attrs["moe_op_type"] == "MoE"

    def test_dense_and_moe_intermediate_sizes(self):
        """Dense MLP uses intermediate_size; MoE uses moe_intermediate_size."""
        m = _make_model()
        assert m.intermediate_size == 128
        assert m.moe_intermediate_size == 32

    def test_inherits_dense_attention(self):
        """MoE model inherits the dense Gemma4 attention config (scale 1.0, no offset)."""
        m = _make_model()
        assert m.attention_attrs["scale"] == 1.0
        assert m.layernorm_attrs["add_offset"] == 0
        assert m.is_local(0) is True and m.is_local(1) is False

    def _build_one_layer(self, m):
        hidden = m.hidden_size
        moe_layer, dense_mlp = _mock_moe_layer(
            hidden, m.moe_attrs["num_experts"], m.moe_intermediate_size, m.intermediate_size
        )
        # Seed layernorm_attrs as if the pre-FFN SkipLayerNorm had run.
        m.make_value("resid", ir.DataType.FLOAT, ["batch_size", "sequence_length", hidden])
        m.make_value("normed", ir.DataType.FLOAT, ["batch_size", "sequence_length", hidden])
        m.layernorm_attrs["root_input"] = "resid"
        m.layernorm_attrs["output_0"] = "normed"
        m.layernorm_attrs["skip_input"] = "normed"
        m._current_layer = moe_layer
        m.make_mlp(0, dense_mlp, "normed")
        return moe_layer

    def test_parallel_dense_moe_combine(self):
        """make_mlp builds both branches, a MoE op, and combines them with an Add."""
        m = _make_model()
        self._build_one_layer(m)
        names = {n.name for n in m.model.graph}
        assert any("/model/layers.0/moe/MoE" in n for n in names)
        assert any("/model/layers.0/ffn_combine/Add" in n for n in names)
        assert any("/model/layers.0/moe/router/MatMul" in n for n in names)
        # The FFN block contribution (skip_input) is the combine output.
        assert m.layernorm_attrs["skip_input"] == "/model/layers.0/ffn_combine/Add/output_0"

    def test_router_preprojection_subgraph(self):
        """Router path: scaleless RMSNorm -> * (scale * hidden^-0.5) -> proj -> reshape."""
        m = _make_model()
        self._build_one_layer(m)
        names = {n.name for n in m.model.graph}
        assert any("moe/router/norm/SimplifiedLayerNormalization" in n for n in names)
        assert any("moe/router/scale/Mul" in n for n in names)
        assert any("moe/router/Reshape" in n for n in names)

    def test_per_expert_scale_folded_into_down_proj(self):
        """down_proj initializer must equal raw_down_proj * per_expert_scale (folded offline)."""
        m = _make_model()
        layer = self._build_one_layer(m)
        expected = (layer.experts.down_proj * layer.router.per_expert_scale.reshape(-1, 1, 1)).numpy()
        init = m.values["model.layers.0.moe.experts.down_proj.weight"].const_value.numpy()
        np.testing.assert_allclose(init, expected, rtol=1e-5, atol=1e-5)
