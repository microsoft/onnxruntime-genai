# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from models.builders.base import Model
from models.builders.qwen3_8 import Qwen4ExpGraphModel


def test_model_builder_does_not_emit_identity_nodes():
    root = Path(__file__).parents[3] / "src/python/py/models"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "make_node" and node.args and isinstance(node.args[0], ast.Constant):
                    assert node.args[0].value != "Identity", f"{path}:{node.lineno}"


def test_named_output_uses_original_producer_without_identity(tmp_path):
    model = Model(graph_name="named_output")
    model.io_dtype = ir.DataType.FLOAT
    data = model.make_value("data", ir.DataType.FLOAT, [2])
    model.graph.inputs.append(data)
    declared = model.make_value("result", ir.DataType.FLOAT, [2])
    declared.metadata_props["binding"] = "output"
    model.graph.outputs.append(declared)
    model.make_add("/sum", ["data", "/model/constants/FLOAT/1"], ir.DataType.FLOAT, [2])
    producer = model.values["/sum/output_0"].producer()
    model.rename_value("/sum/output_0", "result")
    assert model.graph.outputs[0] is producer.outputs[0]
    assert model.graph.outputs[0].metadata_props["binding"] == "output"
    assert not any(node.op_type == "Identity" for node in model.graph)
    path = tmp_path / "model.onnx"
    model.save_onnx_model(model.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"data": np.array([3, 5], np.float32)})[0], [4, 6])


@pytest.mark.parametrize("initializer", [False, True])
def test_renaming_does_not_change_input_or_weight_contracts(initializer):
    model = Model(graph_name="reject_alias")
    if initializer:
        model.make_initializer(np.ones(2, np.float32), "source")
    else:
        model.graph.inputs.append(model.make_value("source", ir.DataType.FLOAT, [2]))
    with pytest.raises(ValueError, match="Cannot rename graph input or initializer"):
        model.rename_value("source", "result")


def test_companion_graph_uses_base_initialization():
    model = Qwen4ExpGraphModel(ir.DataType.FLOAT, "companion.onnx", "companion")
    assert model.graph.name == "companion"
    assert model.model.graph is model.graph
    assert model.values == {}
    assert model.node_names == set()
