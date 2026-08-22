# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import MethodType

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


@pytest.mark.parametrize("ep", ["cpu", "cuda", "webgpu", "dml"])
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
