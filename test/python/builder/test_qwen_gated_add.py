# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import MethodType

import onnx_ir as ir

from models.builders.qwen import Qwen35MoeTextModel


def _make_model(fuse_shared_expert_gate):
    model = object.__new__(Qwen35MoeTextModel)
    model.fuse_shared_expert_gate = fuse_shared_expert_gate
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


def test_cuda_fusion_emits_one_gated_add():
    model = _make_model(True)

    output = model._combine_routed_and_shared_experts(3, "routed", "shared", "gate")

    assert output == "/model/layers.3/moe/GatedAdd/output_0"
    assert [call[0] for call in model.calls] == ["make_node", "make_value"]
    _, args, kwargs = model.calls[0]
    assert args == ("GatedAdd",)
    assert kwargs["inputs"] == ["routed", "shared", "gate"]
    assert kwargs["outputs"] == [output]
    assert kwargs["domain"] == "com.microsoft"


def test_fusion_opt_out_preserves_mul_add_graph():
    model = _make_model(False)

    output = model._combine_routed_and_shared_experts(3, "routed", "shared", "gate")

    assert output == "/model/layers.3/moe/Add/output_0"
    assert [call[0] for call in model.calls] == ["make_mul", "make_add"]
    assert model.calls[0][1] == (
        "/model/layers.3/shared_expert/gate/Mul",
        ["shared", "gate"],
    )
    assert model.calls[1][1] == (
        "/model/layers.3/moe/Add",
        ["routed", "/model/layers.3/shared_expert/gate/Mul/output_0"],
    )
