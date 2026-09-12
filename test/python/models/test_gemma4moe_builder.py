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


def _mock_quark_moe_layer(hidden, num_experts, moe_inter, group_size):
    """Mock a pre-quantized Quark uint2 MoE layer, mirroring what the QuarkModel loader
    produces: `layer.mlp.experts` is a container exposing the re-fused fc1/fc2 packed
    tensors plus per-expert `gate_proj.input_prescale`, and `layer.router` carries the
    scale / per_expert_scale. Weights are packed 4 codes/uint8 (2-bit); scales and float
    zero-points are per-group ([E, out, in/group_size])."""
    blocks_in = hidden // group_size  # fc1 packs along hidden
    blocks_in_fc2 = moe_inter // group_size  # fc2 packs along moe_inter
    fc1_out = 2 * moe_inter  # gate|up concat
    packed_hidden = hidden // 4  # 4 uint2 codes per uint8
    packed_inter = moe_inter // 4

    class _Experts:
        def __init__(self):
            self.fc1_weights = torch.randint(0, 256, (num_experts, fc1_out, packed_hidden), dtype=torch.uint8)
            self.fc1_scales = torch.rand(num_experts, fc1_out, blocks_in) + 0.1
            self.fc1_zero_points = torch.full((num_experts, fc1_out, blocks_in), 1.5)
            self.fc2_weights = torch.randint(0, 256, (num_experts, hidden, packed_inter), dtype=torch.uint8)
            self.fc2_scales = torch.rand(num_experts, hidden, blocks_in_fc2) + 0.1
            self.fc2_zero_points = torch.full((num_experts, hidden, blocks_in_fc2), 1.5)
            prescale = types.SimpleNamespace(input_prescale=torch.rand(hidden) + 0.5)
            self._e = {i: types.SimpleNamespace(gate_proj=prescale) for i in range(num_experts)}

        def keys(self):
            return self._e.keys()

        def __getitem__(self, i):
            return self._e[i]

    return types.SimpleNamespace(
        mlp=types.SimpleNamespace(experts=_Experts()),
        router=types.SimpleNamespace(
            proj=types.SimpleNamespace(weight=torch.randn(num_experts, hidden), bias=None),
            scale=torch.randn(hidden),
            per_expert_scale=torch.rand(num_experts) + 0.5,
        ),
    )


class TestGemma4MoEQuarkPath:
    """Pre-quantized Quark uint2 (2-bit) expert path: make_moe_quark."""

    def _make_quark_model(self, ep="cpu", group_size=16):
        m = _make_model(onnx_dtype=ir.DataType.FLOAT)
        # The QuarkModel loader sets these; seed them so make_moe_quark can run standalone.
        m.quant_type = "quark"
        m.ep = ep
        m.moe_attrs["op_type"] = "QMoE"
        m.moe_attrs["expert_weight_bits"] = 2
        m.moe_attrs["swiglu_fusion"] = 1 if ep == "cpu" else 2
        m.moe_attrs["block_size"] = group_size
        m.shared_input_rotations = {m.hidden_size: torch.randn(m.hidden_size, m.hidden_size)}
        m.shared_rotation_initializers = set()
        return m

    def _build_quark_layer(self, m, group_size=16):
        hidden = m.hidden_size
        num_experts = m.moe_attrs["num_experts"]
        layer = _mock_quark_moe_layer(hidden, num_experts, m.moe_intermediate_size, group_size)
        m.make_value("resid", ir.DataType.FLOAT, ["batch_size", "sequence_length", hidden])
        m.make_moe_quark(0, layer, "resid", "resid")
        return layer

    def test_qmoe_node_emits_expert_weight_bits_2(self):
        """The fused QMoE op carries expert_weight_bits=2 for the 2-bit Quark path."""
        m = self._make_quark_model(ep="cpu")
        self._build_quark_layer(m)
        qmoe = next(n for n in m.model.graph if n.op_type == "QMoE")
        assert qmoe.attributes["expert_weight_bits"].as_int() == 2

    def test_cpu_emits_float_zero_points(self):
        """On CPU the float per-group zero-points are emitted as QMoE inputs (float io_dtype)."""
        m = self._make_quark_model(ep="cpu")
        self._build_quark_layer(m)
        zp_name = "model.layers.0.moe.experts.gate_up_proj.zero_points"
        assert zp_name in m.values
        assert m.values[zp_name].const_value.dtype == ir.DataType.FLOAT
        # And the QMoE node references it as an input (not the empty string).
        qmoe = next(n for n in m.model.graph if n.op_type == "QMoE")
        assert any(inp is not None and inp.name == zp_name for inp in qmoe.inputs)

    def test_cuda_omits_zero_points(self):
        """On CUDA the symmetric -1.5*scale bias is reconstructed by the op, so zp is omitted."""
        m = self._make_quark_model(ep="cuda")
        self._build_quark_layer(m)
        assert "model.layers.0.moe.experts.gate_up_proj.zero_points" not in m.values
        assert "model.layers.0.moe.experts.down_proj.zero_points" not in m.values


def _mock_quark_int4_moe_layer(hidden, num_experts, moe_inter, group_size):
    """Mock a plain Quark/AWQ int4 MoE layer. Unlike the factored uint2 checkpoint, these
    experts have NO input_prescale/rotation, are already emitted interleaved by
    combine_and_repack_gate_up (no fc1 reorder), and carry INTEGER (uint8) per-group
    zero-points. Weights pack 2 codes/uint8 (4-bit)."""
    blocks_in = hidden // group_size
    blocks_in_fc2 = moe_inter // group_size
    fc1_out = 2 * moe_inter
    packed_hidden = hidden // 2  # 2 uint4 codes per uint8
    packed_inter = moe_inter // 2

    class _Experts:
        def __init__(self):
            self.fc1_weights = torch.randint(0, 256, (num_experts, fc1_out, packed_hidden), dtype=torch.uint8)
            self.fc1_scales = torch.rand(num_experts, fc1_out, blocks_in) + 0.1
            self.fc1_zero_points = torch.randint(0, 16, (num_experts, fc1_out, blocks_in), dtype=torch.uint8)
            self.fc2_weights = torch.randint(0, 256, (num_experts, hidden, packed_inter), dtype=torch.uint8)
            self.fc2_scales = torch.rand(num_experts, hidden, blocks_in_fc2) + 0.1
            self.fc2_zero_points = torch.randint(0, 16, (num_experts, hidden, blocks_in_fc2), dtype=torch.uint8)
            # Plain Quark/AWQ experts have no prescale.
            noprescale = types.SimpleNamespace(input_prescale=None)
            self._e = {i: types.SimpleNamespace(gate_proj=noprescale) for i in range(num_experts)}

        def keys(self):
            return self._e.keys()

        def __getitem__(self, i):
            return self._e[i]

    return types.SimpleNamespace(
        mlp=types.SimpleNamespace(experts=_Experts()),
        router=types.SimpleNamespace(
            proj=types.SimpleNamespace(weight=torch.randn(num_experts, hidden), bias=None),
            scale=torch.randn(hidden),
            per_expert_scale=torch.rand(num_experts) + 0.5,
        ),
    )


class TestGemma4MoEQuarkInt4Path:
    """Plain Quark/AWQ int4 expert path: the shape-guarded generalization of make_moe_quark
    (no prescale/rotation, no fc1 reorder, integer zero-points)."""

    def _make_quark_model(self, ep="cpu", group_size=16):
        m = _make_model(onnx_dtype=ir.DataType.FLOAT)
        m.quant_type = "quark"
        m.ep = ep
        m.moe_attrs["op_type"] = "QMoE"
        m.moe_attrs["expert_weight_bits"] = 4
        m.moe_attrs["swiglu_fusion"] = 1 if ep == "cpu" else 2
        m.moe_attrs["block_size"] = group_size
        m.shared_input_rotations = {m.hidden_size: torch.randn(m.hidden_size, m.hidden_size)}
        m.shared_rotation_initializers = set()
        return m

    def _build_quark_layer(self, m, group_size=16):
        hidden = m.hidden_size
        num_experts = m.moe_attrs["num_experts"]
        layer = _mock_quark_int4_moe_layer(hidden, num_experts, m.moe_intermediate_size, group_size)
        m.make_value("resid", ir.DataType.FLOAT, ["batch_size", "sequence_length", hidden])
        m.make_moe_quark(0, layer, "resid", "resid")
        return layer

    def test_qmoe_node_emits_expert_weight_bits_4(self):
        """The fused QMoE op carries expert_weight_bits=4 for the int4 Quark path."""
        m = self._make_quark_model(ep="cpu")
        self._build_quark_layer(m)
        qmoe = next(n for n in m.model.graph if n.op_type == "QMoE")
        assert qmoe.attributes["expert_weight_bits"].as_int() == 4

    def test_no_input_prescale_or_rotation(self):
        """Without input_prescale the expert input feeds QMoE directly: no prescale Mul / rotation MatMul."""
        m = self._make_quark_model(ep="cpu")
        self._build_quark_layer(m)
        assert not any("input_prescale" in name for name in m.values)
        assert not any("shared_input_rotation" in name for name in m.values)

    def test_integer_zero_points_not_cast_to_float(self):
        """Plain int4 experts carry uint8 zero-points, emitted at native dtype (no io_dtype cast)."""
        m = self._make_quark_model(ep="cpu")
        self._build_quark_layer(m)
        zp_name = "model.layers.0.moe.experts.gate_up_proj.zero_points"
        assert zp_name in m.values
        assert m.values[zp_name].const_value.dtype == ir.DataType.UINT8

    def test_cuda_int4_keeps_zero_points(self):
        """The zp-omit path is 2-bit-only; int4 keeps its zero-points even on CUDA."""
        m = self._make_quark_model(ep="cuda")
        self._build_quark_layer(m)
        assert "model.layers.0.moe.experts.gate_up_proj.zero_points" in m.values


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
