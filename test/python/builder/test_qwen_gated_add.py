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


def test_moe_model_emits_portable_gated_add_for_dml():
    model = _make_model("dml")
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

def test_shared_expert_and_gated_add_node_names_are_unique():
    model = _make_model("cuda")
    model.intermediate_size = 64
    model.shared_expert_intermediate_size = 128
    model.mlp_attrs = {}
    node_names = []

    def make_matmul(self, weight, name, root_input):
        node_names.append(name)
        return name

    def make_activation(self, layer_id, root_input):
        name = f"/model/layers.{layer_id}/mlp/activation"
        node_names.append(name)
        return name

    def make_mul(self, name, inputs, dtype, shape):
        node_names.append(name)

    def make_sigmoid(self, name, input_path, dtype, shape):
        node_names.append(name)

    def make_hidden_state_shape(self, last_dim=None):
        return ["batch_size", "sequence_length", last_dim or self.hidden_size]

    def make_node(self, op_type, **kwargs):
        node_names.append(kwargs["name"])

    model.make_matmul = MethodType(make_matmul, model)
    model.make_activation = MethodType(make_activation, model)
    model.make_mul = MethodType(make_mul, model)
    model.make_sigmoid = MethodType(make_sigmoid, model)
    model.make_hidden_state_shape = MethodType(make_hidden_state_shape, model)
    model.make_node = MethodType(make_node, model)

    projection = SimpleNamespace(bias=None)
    shared_expert = SimpleNamespace(gate_proj=projection, up_proj=projection, down_proj=projection)
    shared_output, shared_gate = model.make_shared_expert(3, shared_expert, "shared_gate", "root")
    model.make_gated_add(
        "/model/layers.3/moe/GatedAdd",
        root_input="routed",
        scaled_input=shared_output,
        gate=shared_gate,
        shape=["batch_size", "sequence_length", model.hidden_size],
    )

    assert node_names == [
        "/model/layers.3/mlp/gate_proj/MatMul",
        "/model/layers.3/mlp/up_proj/MatMul",
        "/model/layers.3/mlp/activation",
        "/model/layers.3/mlp/Mul",
        "/model/layers.3/mlp/down_proj/MatMul",
        "/model/layers.3/shared_expert_gate/MatMul",
        "/model/layers.3/shared_expert_gate/Sigmoid",
        "/model/layers.3/moe/GatedAdd",
    ]
    assert len(node_names) == len(set(node_names))
