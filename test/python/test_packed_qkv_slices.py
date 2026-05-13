# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Regression test for the packed-qkv slicing fix in PR #2160.

When the model builder routes Q/K/V through a packed `qkv_proj/MatMul`
(GroupQueryAttention with `use_packed_matmul`), it must emit three Slice
nodes that extract Q, K, V from the packed buffer along the last dim and
update `attention_attrs["q_path"|"k_path"|"v_path"]` accordingly. Without
this, GQA was fed the full packed tensor as Q while K/V stayed pointing
at the prior unpacked layer's outputs, producing shape-inference errors
at `o_proj/MatMulNBits` for WebGPU-converted models.

Run with:
    python -m pytest test/python/test_packed_qkv_slices.py -v --test_models <any-value>
"""

from __future__ import annotations

import os
import sys

import onnx_ir as ir

# Import Model from the source tree so tests run against the working copy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))

from models.builders.base import Model


# DeepSeek-R1-Distill-Qwen-1.5B GQA dims (matches FIX_GUIDE.md):
#   num_heads=12, kv_num_heads=2, head_dim=128
#   Q = 12 * 128 = 1536, K = V = 2 * 128 = 256, total = 2048
DEEPSEEK_DIMS = dict(num_attn_heads=12, num_kv_heads=2, head_size=128)

# Llama-3.2-1B GQA dims:
#   num_heads=32, kv_num_heads=8, head_dim=64
#   Q = 32 * 64 = 2048, K = V = 8 * 64 = 512, total = 3072
LLAMA_DIMS = dict(num_attn_heads=32, num_kv_heads=8, head_size=64)


def _make_minimal_model(dims: dict) -> Model:
    """Build a `Model` instance with just enough state for graph-construction helpers.

    Bypasses `__init__` so we don't need a real HF config; sets only the attributes
    that `make_packed_qkv_slices` / `make_slice` / `make_node` / `make_value` read.
    """
    model = object.__new__(Model)
    model.num_attn_heads = dims["num_attn_heads"]
    model.num_kv_heads = dims["num_kv_heads"]
    model.head_size = dims["head_size"]
    model.io_dtype = ir.DataType.FLOAT16
    model.graph = ir.Graph(
        inputs=(), outputs=(), nodes=(),
        opset_imports={"": 21, "com.microsoft": 1},
        name="main_graph",
    )
    model.model = ir.Model(model.graph, ir_version=10, producer_name="test")
    model.values = {}
    model.node_names = set()
    model.attention_attrs = {
        "q_path": "",
        "k_path": "",
        "v_path": "",
        "use_packed_matmul": True,
    }
    return model


def _slice_nodes_by_proj(graph: ir.Graph, layer_id: int) -> dict[str, ir.Node]:
    """Return Slice nodes emitted by `make_packed_qkv_slices` for a given layer."""
    expected = {
        f"/model/layers.{layer_id}/attn/q_proj/Slice": "q",
        f"/model/layers.{layer_id}/attn/k_proj/Slice": "k",
        f"/model/layers.{layer_id}/attn/v_proj/Slice": "v",
    }
    out: dict[str, ir.Node] = {}
    for node in graph:
        if node.op_type == "Slice" and node.name in expected:
            out[expected[node.name]] = node
    return out


def _input_names(node: ir.Node) -> list[str]:
    return [inp.name for inp in node.inputs]


class TestPackedQKVSlices:
    """Verify packed-qkv slicing emits the right graph and routes Q/K/V correctly."""

    def test_emits_three_slice_nodes(self):
        """One Slice node per projection on the current layer."""
        model = _make_minimal_model(DEEPSEEK_DIMS)
        layer_id = 0
        packed = f"/model/layers.{layer_id}/attn/qkv_proj/MatMul/output_0"

        model.make_packed_qkv_slices(layer_id, packed)

        nodes = _slice_nodes_by_proj(model.graph, layer_id)
        assert set(nodes) == {"q", "k", "v"}, f"Missing Slice nodes: {set(nodes)}"

    def test_q_path_points_to_q_slice_not_packed_buffer(self):
        """Q must point to the new Slice output, not the full packed MatMul output."""
        model = _make_minimal_model(DEEPSEEK_DIMS)
        packed = "/model/layers.0/attn/qkv_proj/Add/output_0"

        model.make_packed_qkv_slices(0, packed)

        assert model.attention_attrs["q_path"] == "/model/layers.0/attn/q_proj/Slice/output_0"
        assert model.attention_attrs["k_path"] == "/model/layers.0/attn/k_proj/Slice/output_0"
        assert model.attention_attrs["v_path"] == "/model/layers.0/attn/v_proj/Slice/output_0"
        # Crucially q_path no longer carries the full packed buffer.
        assert model.attention_attrs["q_path"] != packed

    def test_slice_indices_match_deepseek_layout(self):
        """DeepSeek-R1-Distill-Qwen-1.5B: Q[0:1536], K[1536:1792], V[1792:2048] on axis -1."""
        model = _make_minimal_model(DEEPSEEK_DIMS)
        packed = "/model/layers.0/attn/qkv_proj/MatMul/output_0"

        model.make_packed_qkv_slices(0, packed)
        nodes = _slice_nodes_by_proj(model.graph, 0)

        q_inputs = _input_names(nodes["q"])
        k_inputs = _input_names(nodes["k"])
        v_inputs = _input_names(nodes["v"])

        # Slice(data, starts, ends, axes)
        assert q_inputs[0] == packed
        assert q_inputs[1] == "/model/constants/INT64/[0]"
        assert q_inputs[2] == "/model/constants/INT64/[1536]"
        assert q_inputs[3] == "/model/constants/INT64/[-1]"

        assert k_inputs[0] == packed
        assert k_inputs[1] == "/model/constants/INT64/[1536]"
        assert k_inputs[2] == "/model/constants/INT64/[1792]"
        assert k_inputs[3] == "/model/constants/INT64/[-1]"

        assert v_inputs[0] == packed
        assert v_inputs[1] == "/model/constants/INT64/[1792]"
        assert v_inputs[2] == "/model/constants/INT64/[2048]"
        assert v_inputs[3] == "/model/constants/INT64/[-1]"

    def test_slice_indices_match_llama_layout(self):
        """Llama-3.2-1B: Q[0:2048], K[2048:2560], V[2560:3072] on axis -1."""
        model = _make_minimal_model(LLAMA_DIMS)
        packed = "/model/layers.5/attn/qkv_proj/MatMul/output_0"

        model.make_packed_qkv_slices(5, packed)
        nodes = _slice_nodes_by_proj(model.graph, 5)

        q_inputs = _input_names(nodes["q"])
        k_inputs = _input_names(nodes["k"])
        v_inputs = _input_names(nodes["v"])

        assert q_inputs[1:4] == [
            "/model/constants/INT64/[0]",
            "/model/constants/INT64/[2048]",
            "/model/constants/INT64/[-1]",
        ]
        assert k_inputs[1:4] == [
            "/model/constants/INT64/[2048]",
            "/model/constants/INT64/[2560]",
            "/model/constants/INT64/[-1]",
        ]
        assert v_inputs[1:4] == [
            "/model/constants/INT64/[2560]",
            "/model/constants/INT64/[3072]",
            "/model/constants/INT64/[-1]",
        ]

    def test_axes_passed_as_input_not_attribute(self):
        """`axes` must be an input tensor (opset-21), not a node attribute.

        Older ONNX consumers reject `Slice` with `axes` as an attribute; this is
        called out in FIX_GUIDE.md as a common mistake.
        """
        model = _make_minimal_model(DEEPSEEK_DIMS)
        model.make_packed_qkv_slices(0, "/packed")
        for node in _slice_nodes_by_proj(model.graph, 0).values():
            assert len(node.inputs) == 4, "Slice must have exactly 4 inputs (data, starts, ends, axes)"
            assert "axes" not in dict(node.attributes), "axes must not be a Slice attribute"

    def test_breaks_cross_layer_reference(self):
        """K/V from a prior unpacked layer must not leak into a later packed layer.

        Simulates: layer 0 takes the unpacked branch and sets k_path/v_path to
        per-layer-0 MatMul outputs; layer 1 then takes the packed branch. After
        slicing, layer 1's k_path/v_path must reference layer 1 — not layer 0.
        """
        model = _make_minimal_model(DEEPSEEK_DIMS)

        # Layer 0 (unpacked): paths point at layer-0 MatMuls.
        model.attention_attrs["q_path"] = "/model/layers.0/attn/q_proj/MatMul/output_0"
        model.attention_attrs["k_path"] = "/model/layers.0/attn/k_proj/MatMul/output_0"
        model.attention_attrs["v_path"] = "/model/layers.0/attn/v_proj/MatMul/output_0"

        # Layer 1 (packed): emit slices over the packed layer-1 buffer.
        layer1_packed = "/model/layers.1/attn/qkv_proj/MatMul/output_0"
        model.make_packed_qkv_slices(1, layer1_packed)

        # The previously-broken behavior would leave k_path/v_path pointing at layer 0.
        for path_key in ("q_path", "k_path", "v_path"):
            assert "layers.0" not in model.attention_attrs[path_key], (
                f"{path_key} still references layer 0: {model.attention_attrs[path_key]}"
            )
            assert "layers.1" in model.attention_attrs[path_key]

    def test_slice_outputs_registered_with_io_dtype(self):
        """Slice outputs must carry `io_dtype` (FLOAT16) so downstream nodes type-check."""
        model = _make_minimal_model(DEEPSEEK_DIMS)
        model.make_packed_qkv_slices(0, "/packed")

        for path_key in ("q_path", "k_path", "v_path"):
            out_name = model.attention_attrs[path_key]
            value = model.values[out_name]
            assert value.dtype == ir.DataType.FLOAT16, (
                f"{path_key} ({out_name}) has dtype {value.dtype}, expected FLOAT16"
            )

    def test_slice_output_shapes(self):
        """Each Slice output should advertise the correct per-projection last dim."""
        model = _make_minimal_model(DEEPSEEK_DIMS)
        model.make_packed_qkv_slices(0, "/packed")

        expected_last_dim = {
            "q_path": DEEPSEEK_DIMS["num_attn_heads"] * DEEPSEEK_DIMS["head_size"],  # 1536
            "k_path": DEEPSEEK_DIMS["num_kv_heads"] * DEEPSEEK_DIMS["head_size"],    # 256
            "v_path": DEEPSEEK_DIMS["num_kv_heads"] * DEEPSEEK_DIMS["head_size"],    # 256
        }
        for path_key, last_dim in expected_last_dim.items():
            out_name = model.attention_attrs[path_key]
            shape = list(model.values[out_name].shape)
            assert shape[-1] == last_dim, (
                f"{path_key} last dim is {shape[-1]}, expected {last_dim} (full shape: {shape})"
            )
