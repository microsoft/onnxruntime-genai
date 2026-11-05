# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Keep raw source validation ahead of lossy public ONNX IR conversion."""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders.component import ComponentModelLoader


@pytest.fixture
def component_proto():
    node = ir.Node(
        "",
        "Constant",
        [],
        attributes=[ir.AttrInt64("value_int", 1)],
    )
    node.outputs[0].name = "output"
    node.outputs[0].type = ir.TensorType(ir.DataType.INT64)
    node.outputs[0].shape = ir.Shape([])
    model = ir.Model(
        ir.Graph(
            [],
            [node.outputs[0]],
            nodes=[node],
            name="component",
            opset_imports={"": 21},
        ),
        ir_version=10,
    )
    return ir.serde.serialize_model(model)


@pytest.mark.parametrize(
    "placement",
    ["training", "sparse_initializer", "malformed_attribute", "sparse_attribute"],
)
def test_raw_external_validation_precedes_lossy_ir_loading(tmp_path, component_proto, placement):
    tensor = ir.serde.serialize_tensor(
        ir.ExternalTensor(
            "../outside.bin",
            offset=0,
            length=4,
            dtype=ir.DataType.FLOAT,
            shape=ir.Shape([1]),
            name="weight",
        )
    )
    # These are source-format payloads that cannot be represented faithfully in IR.
    if placement == "training":
        component_proto.training_info.add().initialization.initializer.append(tensor)
    elif placement == "sparse_initializer":
        component_proto.graph.sparse_initializer.add().values.CopyFrom(tensor)
    elif placement == "malformed_attribute":
        component_proto.graph.node[0].attribute[0].t.CopyFrom(tensor)
    else:
        attribute = component_proto.graph.node[0].attribute.add(name="sparse", type=onnx.AttributeProto.SPARSE_TENSOR)
        attribute.sparse_tensor.values.CopyFrom(tensor)
    path = tmp_path / "component.onnx"
    path.write_bytes(component_proto.SerializeToString())

    if placement == "sparse_attribute":
        with pytest.raises(ir.serde.SerdeError):
            ir.load(path)
    else:
        normalized = ir.serde.serialize_model(ir.load(path))
        assert b"../outside.bin" not in normalized.SerializeToString()

    with pytest.raises(ValueError, match="unsafe external data location"):
        ComponentModelLoader(path, "head").load()


def test_schema_checker_must_validate_original_source(tmp_path, component_proto):
    tensor = ir.serde.serialize_tensor(ir.Tensor(np.array([1.0], dtype=np.float32)))
    # INT is declared, but a second, incompatible tensor payload is also present.
    component_proto.graph.node[0].attribute[0].t.CopyFrom(tensor)
    path = tmp_path / "component.onnx"
    path.write_bytes(component_proto.SerializeToString())
    loader = ComponentModelLoader(path, "head")

    normalized = loader.load()
    onnx.checker.check_model(ir.serde.serialize_model(normalized))
    with pytest.raises(ValueError, match="source is not a valid ONNX model"):
        loader.validate()


def test_external_tensor_remains_lazy_public_ir(tmp_path, component_proto, monkeypatch):
    tensor = ir.ExternalTensor(
        "weights.bin",
        offset=4,
        length=4,
        dtype=ir.DataType.FLOAT,
        shape=ir.Shape([1]),
        name="weight",
    )
    component_proto.graph.initializer.append(ir.serde.serialize_tensor(tensor))
    path = tmp_path / "component.onnx"
    path.write_bytes(component_proto.SerializeToString())
    (tmp_path / "weights.bin").write_bytes(b"\0" * 8)

    def reject_materialization(self):
        pytest.fail("component metadata validation must not read external tensor data")

    monkeypatch.setattr(ir.ExternalTensor, "numpy", reject_materialization)
    loader = ComponentModelLoader(path, "head")
    model = loader.load()
    loader.validate()

    assert isinstance(model, ir.Model)
    assert isinstance(model.graph.initializers["weight"].const_value, ir.ExternalTensor)
    assert loader.external_data_files(model.meta["builder_config.external_tensors"]) == [
        ("weights.bin", tmp_path / "weights.bin")
    ]
