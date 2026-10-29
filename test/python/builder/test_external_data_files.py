# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx import numpy_helper

from models.builders.base import Model


def external_location(initializer):
    return next(entry.value for entry in initializer.external_data if entry.key == "location")


@pytest.mark.parametrize("opset,ir_version", [(20, 10), (18, 9)])
def test_make_graph_initializes_minimal_model(opset, ir_version):
    builder = Model.__new__(Model)

    builder.make_graph("head", opset=opset, ir_version=ir_version)

    assert builder.graph is builder.model.graph
    assert builder.graph.name == "head"
    assert builder.graph.opset_imports == {"": opset}
    assert builder.model.ir_version == ir_version
    assert builder.model.producer_name == "onnxruntime-genai"
    assert builder.values == {}
    assert builder.node_names == set()
    assert not builder.graph.inputs
    assert not builder.graph.outputs
    assert not builder.graph.initializers
    assert not list(builder.graph)


def test_save_model_places_all_initializers_in_its_own_external_data(tmp_path):
    values = {
        "model.layers.1.ple.ngram_embedding.weight": np.arange(12, dtype=np.uint8).reshape(3, 4),
        "model.layers.1.ple.ngram_embedding.weight_scale": np.array([0.5], dtype=np.float16),
        "model.layers.0.mlp.weight": np.arange(6, dtype=np.float16).reshape(2, 3),
    }
    initializers = [
        ir.Value(name=name, const_value=ir.tensor(value, name=name))
        for name, value in values.items()
    ]
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), initializers=initializers, opset_imports={"": 22})

    builder = Model.__new__(Model)
    builder.model = ir.Model(graph, ir_version=10, producer_name="onnxruntime-genai")
    builder.filename = "model.onnx"
    builder.cache_dir = os.fspath(tmp_path / "cache")
    builder.quant_type = None
    builder.onnx_dtype = ir.DataType.FLOAT16
    builder.save_model(tmp_path)

    assert (tmp_path / "model.onnx.data").is_file()
    assert not (tmp_path / "engram.onnx.data").exists()
    model = onnx.load(tmp_path / "model.onnx", load_external_data=False)
    locations = {initializer.name: external_location(initializer) for initializer in model.graph.initializer}
    assert locations == {
        "model.layers.1.ple.ngram_embedding.weight": "model.onnx.data",
        "model.layers.1.ple.ngram_embedding.weight_scale": "model.onnx.data",
        "model.layers.0.mlp.weight": "model.onnx.data",
    }

    loaded = onnx.load(tmp_path / "model.onnx", load_external_data=True)
    loaded_values = {initializer.name: numpy_helper.to_array(initializer) for initializer in loaded.graph.initializer}
    for name, expected in values.items():
        np.testing.assert_array_equal(loaded_values[name], expected)


def test_save_model_copies_existing_engram_data_without_rewriting_it(tmp_path):
    table_name = "model.ple.ngram_embedding.weight"
    table = np.arange(12, dtype=np.uint8).reshape(3, 4)
    engram_graph = ir.Graph(
        inputs=(), outputs=(), nodes=(),
        initializers=[
            ir.Value(name="small", const_value=ir.tensor(np.array([1], dtype=np.int64), name="small")),
            ir.Value(name=table_name, const_value=ir.tensor(table, name=table_name)),
        ],
        opset_imports={"": 22},
    )
    ir.save(ir.Model(engram_graph, ir_version=10), tmp_path / "engram.onnx",
            external_data="engram.onnx.data", size_threshold_bytes=0)
    data_path = tmp_path / "engram.onnx.data"
    original_data = data_path.read_bytes()

    decoder_graph = ir.Graph(
        inputs=(), outputs=(), nodes=(),
        initializers=[ir.Value(
            name=table_name, const_value=ir.load(tmp_path / "engram.onnx").graph.initializers[table_name].const_value
        )],
        opset_imports={"": 22},
    )
    builder = Model.__new__(Model)
    builder.model = ir.Model(decoder_graph, ir_version=10, producer_name="onnxruntime-genai")
    builder.filename = "model.onnx"
    builder.cache_dir = os.fspath(tmp_path / "cache")
    builder.quant_type = None
    builder.onnx_dtype = ir.DataType.FLOAT16
    builder.save_model(tmp_path)

    assert data_path.read_bytes() == original_data
    for filename in ("engram.onnx", "model.onnx"):
        graph = onnx.load(tmp_path / filename, load_external_data=False)
        initializer = next(value for value in graph.graph.initializer if value.name == table_name)
        assert external_location(initializer) == filename + ".data"
        loaded = onnx.load(tmp_path / filename, load_external_data=True)
        initializer = next(value for value in loaded.graph.initializer if value.name == table_name)
        np.testing.assert_array_equal(numpy_helper.to_array(initializer), table)


def test_save_model_externalizes_tensor_attributes_but_keeps_shape_constants_inline(tmp_path):
    weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [2], [1, 2])
    attribute = onnx.helper.make_tensor("attribute", onnx.TensorProto.FLOAT, [2], [3, 4])
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [1], [2])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Constant", [], ["shape"], value=shape),
            onnx.helper.make_node(
                "Probe", ["weight"], ["output"], domain="test",
                tensor=attribute, tensors=[attribute, attribute],
            ),
        ],
        "attributes", [], [], [weight],
    )
    model = ir.serde.deserialize_model(onnx.helper.make_model(graph, ir_version=10))
    path = tmp_path / "model.onnx"
    original_attribute = model.graph.node(1).attributes["tensor"]

    Model.save_onnx_model(model, path)
    Model.save_onnx_model(ir.load(path), path)

    exported = onnx.load(path, load_external_data=False)
    assert external_location(exported.graph.initializer[0]) == "model.onnx.data"
    assert not exported.graph.node[0].attribute[0].t.external_data
    for attribute in exported.graph.node[1].attribute:
        tensors = [attribute.t] if attribute.type == onnx.AttributeProto.TENSOR else attribute.tensors
        assert all(external_location(tensor) == "model.onnx.data" for tensor in tensors)
    loaded = onnx.load(path, load_external_data=True)
    np.testing.assert_array_equal(numpy_helper.to_array(loaded.graph.node[1].attribute[0].t), [3, 4])
    assert model.graph.node(1).attributes["tensor"] is original_attribute
    assert {file.name for file in tmp_path.iterdir()} == {"model.onnx", "model.onnx.data"}


def test_save_model_externalizes_large_constant_weights(tmp_path):
    import onnxruntime as ort  # noqa: PLC0415

    values = np.arange(512, dtype=np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(values, name="weight"))],
        "constant_weights", [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [512])],
    )
    model = ir.serde.deserialize_model(onnx.helper.make_model(
        graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 22)]
    ))
    path = tmp_path / "model.onnx"

    Model.save_onnx_model(model, path)

    exported = onnx.load(path, load_external_data=False)
    assert external_location(exported.graph.node[0].attribute[0].t) == "model.onnx.data"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {})[0], values)


@pytest.mark.parametrize("failure_stage", ["serialize", "replace"])
def test_save_model_failure_preserves_existing_files(tmp_path, monkeypatch, failure_stage):
    path = tmp_path / "model.onnx"
    data_path = tmp_path / "model.onnx.data"
    path.write_bytes(b"old model")
    data_path.write_bytes(b"old data")
    tensor = ir.tensor(np.arange(4, dtype=np.float32), name="weight")
    initializer = ir.Value(name="weight", const_value=tensor)
    model = ir.Model(
        ir.Graph(inputs=(), outputs=(), nodes=(), initializers=[initializer], opset_imports={"": 22}),
        ir_version=10,
    )
    if failure_stage == "serialize":
        def fail_save(*args, **kwargs):
            raise OSError("injected save failure")

        monkeypatch.setattr(ir, "save", fail_save)
    else:
        original_replace = os.replace

        def fail_replace(source, destination):
            if os.fspath(source).endswith(".onnx.saving"):
                raise OSError("injected save failure")
            original_replace(source, destination)

        monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected save failure"):
        Model.save_onnx_model(model, path)

    assert path.read_bytes() == b"old model"
    assert data_path.read_bytes() == b"old data"
    assert initializer.const_value is tensor
    assert {file.name for file in tmp_path.iterdir()} == {"model.onnx", "model.onnx.data"}
