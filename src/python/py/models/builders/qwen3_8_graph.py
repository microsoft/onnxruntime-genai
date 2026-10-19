# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os

import onnx_ir as ir

from .base import Model
from .expansions.graph import Qwen38Graph


class Qwen4ExpGraphModel(Qwen38Graph, Model):
    def __init__(self, io_dtype, filename, graph_name):
        # Companion graphs reuse emission helpers without decoder configuration or cache state.
        self.io_dtype = ir.DataType(io_dtype)
        self.filename = filename
        self.graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22}, name=graph_name)
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

    def save_model(self, output_dir):
        ir.save(
            self.model,
            os.path.join(output_dir, self.filename),
            external_data=f"{self.filename}.data",
            size_threshold_bytes=0,
        )
