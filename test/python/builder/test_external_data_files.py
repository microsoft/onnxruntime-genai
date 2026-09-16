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
        "model.layers.1.ple.ngram_embedding.weight": "engram.data"
    }

    builder.save_model(tmp_path)

    assert (tmp_path / "model.onnx.data").is_file()
    assert (tmp_path / "engram.data").is_file()
    model = onnx.load(tmp_path / "model.onnx", load_external_data=False)
    locations = {initializer.name: external_location(initializer) for initializer in model.graph.initializer}
    assert locations == {
        "model.layers.1.ple.ngram_embedding.weight": "engram.data",
        "model.layers.1.ple.ngram_embedding.weight_scale": "model.onnx.data",
        "model.layers.0.mlp.weight": "model.onnx.data",
    }

    loaded = onnx.load(tmp_path / "model.onnx", load_external_data=True)
    loaded_values = {initializer.name: numpy_helper.to_array(initializer) for initializer in loaded.graph.initializer}
    for name, expected in values.items():
        np.testing.assert_array_equal(loaded_values[name], expected)