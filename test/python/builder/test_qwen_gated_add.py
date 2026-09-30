# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import MethodType, SimpleNamespace

import onnx_ir as ir
import pytest
from models.builders.qwen import Qwen35MoETextModel


def _make_model(ep):
    model = object.__new__(Qwen35MoETextModel)
    model.ep = ep
    model.hidden_size = 2048
    model.io_dtype = ir.DataType.FLOAT16
    model.calls = []

    def record(name):
        def call(self, *args, **kwargs):
            self.calls.append((name, args, kwargs))

        return MethodType(call, model)

    model.make_node = record("make_node")
    model.make_value = record("make_value")
    model.make_mul = record("make_mul")
    model.make_add = record("make_add")
    return model


@pytest.mark.parametrize("ep", ["cpu", "cuda", "webgpu"])
def test_moe_model_emits_one_gated_add(ep):
    model = _make_model(ep)
    name = "/model/layers.3/moe/GatedAdd"
    shape = ["batch_size", "sequence_length", model.hidden_size]

    model.make_gated_add(name, "routed", "shared", "gate", shape)

    assert [call[0] for call in model.calls] == ["make_node", "make_value"]
    _, args, kwargs = model.calls[0]
    assert args == ("GatedAdd",)
    assert kwargs["inputs"] == ["routed", "shared", "gate"]
    assert kwargs["outputs"] == [f"{name}/output_0"]
    assert kwargs["domain"] == "com.microsoft"


@pytest.mark.parametrize("ep", ["dml", "trt-rtx"])
def test_moe_model_emits_portable_gated_add(ep):
    model = _make_model(ep)
    name = "/model/layers.3/moe/GatedAdd"
    shape = ["batch_size", "sequence_length", model.hidden_size]

    model.make_ep_expansions_init()
    model.make_gated_add(name, "routed", "shared", "gate", shape)

    assert [call[0] for call in model.calls] == ["make_mul", "make_add"]
    _, args, kwargs = model.calls[0]
    assert args == (f"{name}/Mul", ["shared", "gate"], model.io_dtype)
    assert kwargs["shape"] == shape
    _, args, kwargs = model.calls[1]
    assert args == (name, ["routed", f"{name}/Mul/output_0"], model.io_dtype)
    assert kwargs["shape"] == shape


@pytest.mark.parametrize(
    "ep, use_paged_attention, hidden_rows_dim, expected_shape",
    [
        ("cuda", False, "num_tokens", ["batch_size", "sequence_length", 2048]),
        ("trt-rtx", False, "num_tokens", ["batch_size", "sequence_length", 2048]),
        ("cuda", True, "num_tokens", ["num_tokens", 2048]),
        ("cuda", True, "num_logits", ["num_logits", 2048]),
    ],
)
def test_moe_updates_decoder_residual(ep, use_paged_attention, hidden_rows_dim, expected_shape):
    model = _make_model(ep)
    model.use_paged_attention = use_paged_attention
    model.hidden_rows_dim = hidden_rows_dim
    model.make_ep_expansions_init()
    model.moe_attrs = {"op_type": "QMoE"}
    model.layernorm_attrs = {"skip_input": "attention_output"}
    model.make_moe_preprocessing = lambda *args: None
    model.make_moe_router = lambda *args: None
    model.make_moe_op = lambda *args, **kwargs: None
    model.make_shared_expert = lambda *args: ("shared", "gate")
    moe = SimpleNamespace(shared_expert=object(), shared_expert_gate=object())

    # make_moe ignores the subgraph's return value. The next decoder layer reads
    # skip_input, which must include the routed and gated shared experts.
    model.make_moe(3, moe, "normalized_hidden_states")

    assert model.layernorm_attrs["skip_input"] == "/model/layers.3/moe/GatedAdd/output_0"
    # The residual consumer must see the same rank and active row dimension as
    # the expert inputs, including rows selected for last-layer logit pruning.
    assert model.calls[-1][2]["shape"] == expected_shape


@pytest.mark.parametrize(
    "use_paged_attention, hidden_rows_dim, expected_shape",
    [
        (False, "num_tokens", ["batch_size", "sequence_length", 1]),
        (True, "num_tokens", ["num_tokens", 1]),
        (True, "num_logits", ["num_logits", 1]),
    ],
)
def test_shared_expert_gate_preserves_hidden_layout(use_paged_attention, hidden_rows_dim, expected_shape):
    model = _make_model("cuda")
    model.use_paged_attention = use_paged_attention
    model.hidden_rows_dim = hidden_rows_dim
    model.intermediate_size = 16
    model.shared_expert_intermediate_size = 8
    model.mlp_attrs = {"output_0": "shared"}
    model.make_mlp_proj = lambda *args: None
    model.make_matmul = lambda projection, name, root_input: name

    shared, gate = model.make_shared_expert(3, object(), object(), "normalized_hidden_states")

    assert shared == "shared"
    _, args, kwargs = model.calls[-1]
    assert args[0] == gate
    assert kwargs["shape"] == expected_shape
