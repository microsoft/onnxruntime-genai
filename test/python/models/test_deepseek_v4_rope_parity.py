# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""DeepSeek V4 RoPE parity test.

DeepSeek V4 applies interleaved rotary embeddings to the *trailing*
``qk_rope_head_dim`` channels of each head, and applies the conjugate rotation
(``-sin``) when de-rotating the attention output.  The builder emits this with a
``com.microsoft::RotaryEmbedding`` node wrapped in slice/concat plumbing
(``DeepSeekV4Model.make_deepseek_rope``).

This test pins that emission against two independent references:

  * ``DeepSeekV4Model.make_deepseek_rope_manual`` — the decomposed subgraph that the
    builder emitted before the op was adopted, run through onnxruntime; and
  * HuggingFace's ``DeepseekV4RotaryEmbedding`` + ``apply_rotary_pos_emb``.

Both prefill (multi-token, positions 0..S-1) and decode (single token at a large
position offset) are covered, with and without ``neg_sin``.

Run with:
    python -m pytest test/python/models/test_deepseek_v4_rope_parity.py -v --test_models <path-or-any-value>

The repository's pytest configuration requires ``--test_models`` during argument
parsing even though this test does not use model data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx_ir as ir
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

# Import the builder from the source tree so tests always run against the working copy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.deepseek import DeepSeekV4Model  # noqa: E402

HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
CACHE_LENGTH = 128
THETA = 10000.0


def _make_builder(*, manual: bool) -> DeepSeekV4Model:
    """Create a bare ``DeepSeekV4Model`` with just enough state to emit RoPE nodes."""
    model = object.__new__(DeepSeekV4Model)
    model.graph = ir.Graph(
        inputs=(),
        outputs=(),
        nodes=(),
        opset_imports={"": 21, "com.microsoft": 1},
        name="main_graph",
    )
    model.model = ir.Model(model.graph, ir_version=10, producer_name="onnxruntime-genai")
    model.values = {}
    model.node_names = set()
    model.io_dtype = ir.DataType.FLOAT
    model.hidden_size = HIDDEN_SIZE
    model.head_size = HEAD_DIM
    model.qk_rope_head_dim = ROPE_DIM
    model.qk_nope_head_dim = HEAD_DIM - ROPE_DIM
    model.rope_attrs = {"theta": THETA, "cache_length": CACHE_LENGTH}
    model.input_names = {"position_ids": "position_ids"}
    model.use_manual_deepseek_rope = manual
    return model


def _build_rope_session(*, manual: bool, neg_sin: bool) -> onnxruntime.InferenceSession:
    """Emit a single RoPE subgraph into a standalone model and load it in onnxruntime."""
    model = _make_builder(manual=manual)

    x = model.make_value("x", ir.DataType.FLOAT, ["batch_size", NUM_HEADS, "sequence_length", HEAD_DIM])
    position_ids = model.make_value("position_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])

    output_name = model.make_deepseek_rope("/test/rope", "x", NUM_HEADS, HEAD_DIM, ROPE_DIM, neg_sin=neg_sin)

    model.graph.inputs.extend([x, position_ids])
    model.graph.outputs.append(model.make_value(output_name))

    proto = ir.to_proto(model.model)
    return onnxruntime.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


def _hf_reference(x: np.ndarray, position_ids: np.ndarray, *, neg_sin: bool) -> np.ndarray:
    """Run HuggingFace's DeepSeek V4 rotary embedding over the same inputs."""
    modeling = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    configuration = pytest.importorskip("transformers.models.deepseek_v4.configuration_deepseek_v4")

    config = configuration.DeepseekV4Config(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=CACHE_LENGTH,
    )
    # Keep only the "main" rope type and pin it to this test's theta / partial factor.
    config.rope_parameters = {
        "main": {
            "rope_type": "default",
            "rope_theta": THETA,
            "partial_rotary_factor": ROPE_DIM / HEAD_DIM,
        }
    }

    rotary = modeling.DeepseekV4RotaryEmbedding(config)
    tensor = torch.from_numpy(x)
    cos, sin = rotary(tensor, torch.from_numpy(position_ids), layer_type="main")
    if neg_sin:
        sin = -sin
    return modeling.apply_rotary_pos_emb(tensor, cos, sin).numpy()


def _inputs(batch_size: int, sequence_length: int, position_offset: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch_size, NUM_HEADS, sequence_length, HEAD_DIM), dtype=np.float32)
    position_ids = np.tile(
        np.arange(position_offset, position_offset + sequence_length, dtype=np.int64), (batch_size, 1)
    )
    return {"x": x, "position_ids": position_ids}


CASES = [
    pytest.param(2, 6, 0, id="prefill"),
    pytest.param(2, 1, 37, id="decode"),
]


class TestDeepSeekV4RopeParity:
    @pytest.mark.parametrize("neg_sin", [False, True], ids=["sin", "neg_sin"])
    @pytest.mark.parametrize(("batch_size", "sequence_length", "position_offset"), CASES)
    def test_matches_huggingface_reference(self, neg_sin, batch_size, sequence_length, position_offset):
        session = _build_rope_session(manual=False, neg_sin=neg_sin)
        feeds = _inputs(batch_size, sequence_length, position_offset)
        actual = session.run(None, feeds)[0]
        expected = _hf_reference(feeds["x"], feeds["position_ids"], neg_sin=neg_sin)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("neg_sin", [False, True], ids=["sin", "neg_sin"])
    @pytest.mark.parametrize(("batch_size", "sequence_length", "position_offset"), CASES)
    def test_matches_decomposed_subgraph(self, neg_sin, batch_size, sequence_length, position_offset):
        feeds = _inputs(batch_size, sequence_length, position_offset)
        fused = _build_rope_session(manual=False, neg_sin=neg_sin).run(None, feeds)[0]
        manual = _build_rope_session(manual=True, neg_sin=neg_sin).run(None, feeds)[0]
        np.testing.assert_allclose(fused, manual, rtol=1e-3, atol=1e-3)

    def test_nope_channels_are_untouched(self):
        """The leading (head_dim - rope_dim) channels must pass through unchanged."""
        session = _build_rope_session(manual=False, neg_sin=False)
        feeds = _inputs(1, 5, 0)
        actual = session.run(None, feeds)[0]
        nope_width = HEAD_DIM - ROPE_DIM
        np.testing.assert_array_equal(actual[..., :nope_width], feeds["x"][..., :nope_width])

    def test_emits_rotary_embedding_op_and_fewer_nodes(self):
        """The op-based emission must be materially smaller than the decomposed one."""
        fused = _make_builder(manual=False)
        fused.make_deepseek_rope("/test/rope", "x", NUM_HEADS, HEAD_DIM, ROPE_DIM)
        manual = _make_builder(manual=True)
        manual.make_deepseek_rope("/test/rope", "x", NUM_HEADS, HEAD_DIM, ROPE_DIM)

        fused_ops = [node.op_type for node in fused.graph]
        assert "RotaryEmbedding" in fused_ops
        assert "RotaryEmbedding" not in [node.op_type for node in manual.graph]
        assert len(fused_ops) < len([node.op_type for node in manual.graph])

        rotary = next(node for node in fused.graph if node.op_type == "RotaryEmbedding")
        assert rotary.domain == "com.microsoft"
        assert rotary.attributes["interleaved"].value == 1
        assert [value.name for value in rotary.inputs] == [
            "/test/rope/rope_in/Slice/output_0",
            "position_ids",
            "deepseek_cos_cache",
            "deepseek_sin_cache",
        ]

    def test_neg_sin_uses_the_negated_cache(self):
        """De-rotation must reuse a negated cache rather than add a Neg node."""
        model = _make_builder(manual=False)
        model.make_deepseek_rope("/test/rope", "x", NUM_HEADS, HEAD_DIM, ROPE_DIM, neg_sin=True)

        rotary = next(node for node in model.graph if node.op_type == "RotaryEmbedding")
        assert rotary.inputs[3].name == "deepseek_sin_cache_neg"
        assert "Neg" not in [node.op_type for node in model.graph]

        cos, sin, neg_sin = (
            model.graph.initializers[name].const_value.numpy()
            for name in ("deepseek_cos_cache", "deepseek_sin_cache", "deepseek_sin_cache_neg")
        )
        # Caches are stored one entry per rotated pair, matching the kernel's convention.
        assert cos.shape == sin.shape == (CACHE_LENGTH, ROPE_DIM // 2)
        np.testing.assert_array_equal(neg_sin, -sin)

    def test_full_head_rotation_skips_slice_and_concat(self):
        """When rope_dim == head_dim there is nothing to slice around the op."""
        model = _make_builder(manual=False)
        model.qk_rope_head_dim = HEAD_DIM
        output_name = model.make_deepseek_rope("/test/rope", "x", NUM_HEADS, HEAD_DIM, HEAD_DIM)

        op_types = [node.op_type for node in model.graph]
        assert op_types == ["RotaryEmbedding"]
        assert output_name == "/test/rope/RotaryEmbedding/output_0"
