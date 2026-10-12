# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import onnx
import onnx_ir as ir
from onnx import numpy_helper

from models.builders.base import Model


def external_location(initializer):
    return next(entry.value for entry in initializer.external_data if entry.key == "location")


def test_save_model_places_mapped_initializer_in_separate_external_data(tmp_path):
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
    builder.external_data_files = {
        "model.layers.1.ple.ngram_embedding.weight": "engram.onnx.data"
    }

    builder.save_model(tmp_path)

    assert (tmp_path / "model.onnx.data").is_file()
    assert (tmp_path / "engram.onnx.data").is_file()
    model = onnx.load(tmp_path / "model.onnx", load_external_data=False)
    locations = {initializer.name: external_location(initializer) for initializer in model.graph.initializer}
    assert locations == {
        "model.layers.1.ple.ngram_embedding.weight": "engram.onnx.data",
        "model.layers.1.ple.ngram_embedding.weight_scale": "model.onnx.data",
        "model.layers.0.mlp.weight": "model.onnx.data",
    }

    loaded = onnx.load(tmp_path / "model.onnx", load_external_data=True)
    loaded_values = {initializer.name: numpy_helper.to_array(initializer) for initializer in loaded.graph.initializer}
    for name, expected in values.items():
        np.testing.assert_array_equal(loaded_values[name], expected)


def test_save_model_references_existing_engram_data_without_rewriting_it(tmp_path):
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
        initializers=[ir.Value(name=table_name, const_value=ir.tensor(table, name=table_name))],
        opset_imports={"": 22},
    )
    builder = Model.__new__(Model)
    builder.model = ir.Model(decoder_graph, ir_version=10, producer_name="onnxruntime-genai")
    builder.filename = "model.onnx"
    builder.cache_dir = os.fspath(tmp_path / "cache")
    builder.quant_type = None
    builder.onnx_dtype = ir.DataType.FLOAT16
    builder.external_data_files = {table_name: "engram.onnx.data"}
    builder.external_data_tensors = {table_name: ir.load(tmp_path / "engram.onnx").graph.initializers[table_name].const_value}

    builder.save_model(tmp_path)

    assert data_path.read_bytes() == original_data
    for filename in ("engram.onnx", "model.onnx"):
        loaded = onnx.load(tmp_path / filename, load_external_data=True)
        initializer = next(value for value in loaded.graph.initializer if value.name == table_name)
        np.testing.assert_array_equal(numpy_helper.to_array(initializer), table)