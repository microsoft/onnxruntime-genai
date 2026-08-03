# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""DeepSeek V4 MoE router parity test.

Both DeepSeek V4 router types (``DeepseekV4TopKRouter`` and ``DeepseekV4HashRouter``)
compute per-token weights as a plain renormalization of the *selected* experts'
scores (``weights = scores.gather(indices); weights /= weights.sum()``) — there is
no softmax over the full expert set. The ``com.microsoft::MoE`` op, however, always
applies its own softmax over *all* experts before selecting the top-k and
(optionally) renormalizing. ``DeepSeekV4Model.make_moe_router_scores`` /
``make_moe_masked_router_logits`` bridge this gap by feeding the op
``log(selected_score)`` at the selected columns and a large negative fill
elsewhere, which is intended to reduce to the exact reference weighting once
passed through the op's internal softmax + renormalize.

This test pins that construction against HuggingFace's real
``DeepseekV4TopKRouter`` / ``DeepseekV4HashRouter`` (available in the
``transformers`` package), by:

  * building just the router subgraph (via ``make_topk_moe`` / ``make_hash_moe``
    with ``make_deepseek_moe_op`` monkeypatched out) followed by an *identity*
    ``com.microsoft::MoE`` op (identity FFN weights, ``activation_type="identity"``),
    so the op's output is exactly ``sum_k weight_k * hidden_state``; and
  * comparing the resulting per-token weighted combination against the same
    combination computed from HuggingFace's reference router weights (with
    ``routed_scaling_factor`` divided back out, since that scale is applied by
    ``make_deepseek_moe_op`` — not the router subgraph under test here — as a
    separate post-hoc ``Mul``).

Run with:
    python -m pytest test/python/models/test_deepseek_v4_moe_router.py -v --test_models <path-or-any-value>

The repository's pytest configuration requires ``--test_models`` during argument
parsing even though this test does not use model data.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

# Import the builder from the source tree so tests always run against the working copy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python", "py"))

from models.builders.deepseek import DeepSeekV4Model  # noqa: E402

HIDDEN_SIZE = 8
NUM_EXPERTS = 6
TOP_K = 2
VOCAB_SIZE = 16


def _make_builder(scoring_func: str) -> DeepSeekV4Model:
    """Create a bare ``DeepSeekV4Model`` with just enough state to emit the MoE router."""
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
    model.moe_attrs = {
        "op_type": "MoE",
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "activation_alpha": 1.702,
        "activation_beta": 1.0,
        "activation_type": "identity",
        "normalize_routing_weights": True,
        "swiglu_fusion": 0,
        "swiglu_limit": None,
        "use_sparse_mixer": False,
    }
    model.scoring_func = scoring_func
    model.input_names = {"input_ids": "input_ids"}
    model.use_paged_attention = False
    return model


def _build_router_session(*, is_hash: bool, scoring_func: str) -> onnxruntime.InferenceSession:
    """Emit ``make_topk_moe``/``make_hash_moe`` with an identity FFN in place of the real one.

    ``make_deepseek_moe_op`` is monkeypatched to skip real expert weights and instead emit an
    identity ``MoE`` op, so the graph's only output is exactly the router's weighted combine
    (``sum_k weight_k * hidden_state``), letting this test isolate router correctness from the
    (separately covered) expert FFN plumbing.
    """
    model = _make_builder(scoring_func)

    def identity_moe_op(layer_id, mlp, root_input, router_probs, op_type, weight_type, base):
        del layer_id, mlp, op_type, weight_type
        eye = torch.eye(HIDDEN_SIZE)
        w1_name = f"{base}/identity/weight1"
        w2_name = f"{base}/identity/weight2"
        model.make_initializer(eye.unsqueeze(0).repeat(NUM_EXPERTS, 1, 1), w1_name, to=model.io_dtype)
        model.make_initializer(eye.unsqueeze(0).repeat(NUM_EXPERTS, 1, 1), w2_name, to=model.io_dtype)
        moe_name = f"{base}/MoE"
        model.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=router_probs,
            weight1=w1_name,
            weight2=w2_name,
        )
        return f"{moe_name}/output_0"

    model.make_deepseek_moe_op = identity_moe_op

    hidden = model.make_value("hidden", ir.DataType.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE])
    model.graph.inputs.append(hidden)

    rng = np.random.default_rng(1)
    gate_weight = torch.from_numpy(rng.standard_normal((NUM_EXPERTS, HIDDEN_SIZE), dtype=np.float32))
    if is_hash:
        tid2eid = torch.from_numpy(rng.integers(0, NUM_EXPERTS, size=(VOCAB_SIZE, TOP_K)).astype(np.int64))
        mlp = SimpleNamespace(gate=SimpleNamespace(weight=SimpleNamespace(data=gate_weight), tid2eid=SimpleNamespace(data=tid2eid)))
        input_ids = model.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        model.graph.inputs.append(input_ids)
        output_name = model.make_hash_moe(0, mlp, "hidden", "MoE", "weight")
        extra = {"tid2eid": tid2eid.numpy()}
    else:
        bias = torch.from_numpy(rng.standard_normal((NUM_EXPERTS,), dtype=np.float32))
        mlp = SimpleNamespace(
            gate=SimpleNamespace(weight=SimpleNamespace(data=gate_weight), e_score_correction_bias=SimpleNamespace(data=bias))
        )
        output_name = model.make_topk_moe(0, mlp, "hidden", "MoE", "weight")
        extra = {"e_score_correction_bias": bias.numpy()}

    model.graph.outputs.append(model.make_value(output_name))

    proto = ir.to_proto(model.model)
    session = onnxruntime.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return session, gate_weight.numpy(), extra


def _reference_weighted_combine(
    hidden: np.ndarray, gate_weight: np.ndarray, *, is_hash: bool, scoring_func: str, extra: dict, input_ids: np.ndarray | None
) -> np.ndarray:
    """Compute ``sum_k weight_k * hidden`` using HuggingFace's real router forward.

    ``routed_scaling_factor`` is divided back out since it is applied by
    ``make_deepseek_moe_op`` (not under test here) as a separate post-hoc ``Mul``.
    """
    modeling = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    configuration = pytest.importorskip("transformers.models.deepseek_v4.configuration_deepseek_v4")

    config = configuration.DeepseekV4Config(
        hidden_size=HIDDEN_SIZE,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        vocab_size=VOCAB_SIZE,
        num_hidden_layers=1,
        mlp_layer_types=["hash_moe" if is_hash else "moe"],
        scoring_func=scoring_func,
    )

    hidden_t = torch.from_numpy(hidden)
    if is_hash:
        router = modeling.DeepseekV4HashRouter(config)
        router.weight.data = torch.from_numpy(gate_weight)
        router.tid2eid.data = torch.from_numpy(extra["tid2eid"])
        _, weights, indices = router(hidden_t, torch.from_numpy(input_ids))
    else:
        router = modeling.DeepseekV4TopKRouter(config)
        router.weight.data = torch.from_numpy(gate_weight)
        router.e_score_correction_bias.data = torch.from_numpy(extra["e_score_correction_bias"])
        _, weights, indices = router(hidden_t)

    weights = weights / config.routed_scaling_factor  # undo the post-hoc scale applied elsewhere
    del indices  # only affects which expert runs; identity experts make this irrelevant here

    batch, seq_len, hidden_dim = hidden.shape
    flat = hidden.reshape(-1, hidden_dim)
    out = np.zeros_like(flat)
    weights_np = weights.detach().numpy()
    for row in range(flat.shape[0]):
        for k in range(TOP_K):
            out[row] += weights_np[row, k] * flat[row]  # identity FFN: expert(x) == x
    return out.reshape(batch, seq_len, hidden_dim)


CASES = [
    pytest.param(False, "sqrtsoftplus", id="topk-sqrtsoftplus"),
    pytest.param(False, "sigmoid", id="topk-sigmoid"),
    pytest.param(True, "sqrtsoftplus", id="hash-sqrtsoftplus"),
]


class TestDeepSeekV4MoeRouter:
    @pytest.mark.parametrize(("is_hash", "scoring_func"), CASES)
    def test_matches_huggingface_reference(self, is_hash, scoring_func):
        session, gate_weight, extra = _build_router_session(is_hash=is_hash, scoring_func=scoring_func)

        rng = np.random.default_rng(2)
        batch_size, seq_len = 2, 3
        hidden = rng.standard_normal((batch_size, seq_len, HIDDEN_SIZE), dtype=np.float32)
        feeds = {"hidden": hidden}
        input_ids = None
        if is_hash:
            input_ids = rng.integers(0, VOCAB_SIZE, size=(batch_size, seq_len)).astype(np.int64)
            feeds["input_ids"] = input_ids

        actual = session.run(None, feeds)[0]
        expected = _reference_weighted_combine(
            hidden, gate_weight, is_hash=is_hash, scoring_func=scoring_func, extra=extra, input_ids=input_ids
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
