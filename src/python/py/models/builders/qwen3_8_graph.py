# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os

import onnx_ir as ir

from .base import Model


class Qwen4ExpGraphModel(Model):
    def __init__(self, io_dtype, filename, graph_name):
        # Companion graphs reuse emission helpers without decoder configuration or cache state.
        self.io_dtype = ir.DataType(io_dtype)
        self.filename = filename
        self.graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22}, name=graph_name)
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

    def save_model(self, output_dir):
        self.save_onnx_model(self.model, os.path.join(output_dir, self.filename))

    def make_linear(self, name, linear, root_input, shape, output=None):
        weight_name = f"{name}.weight"
        self.make_initializer(linear.weight.T, weight_name, to=self.io_dtype)
        matmul_name = f"/{name.replace('.', '/')}/MatMul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [root_input, weight_name], [matmul_output], name=matmul_name)
        self.make_value(matmul_output, self.io_dtype, shape)
        if linear.bias is None:
            if output is not None:
                self.make_node("Identity", [matmul_output], [output], name=f"/{name.replace('.', '/')}/Identity")
                self.make_value(output, self.io_dtype, shape)
                return output
            return matmul_output
        bias_name = f"{name}.bias"
        self.make_initializer(linear.bias, bias_name, to=self.io_dtype)
        add_name = f"/{name.replace('.', '/')}/Add"
        add_output = output or f"{add_name}/output_0"
        self.make_node("Add", [matmul_output, bias_name], [add_output], name=add_name)
        self.make_value(add_output, self.io_dtype, shape)
        return add_output
    def make_layer_norm(self, name, layer_norm, root_input, shape):
        scale_name = f"{name}.weight"
        bias_name = f"{name}.bias"
        self.make_initializer(layer_norm.weight, scale_name, to=self.io_dtype)
        self.make_initializer(layer_norm.bias, bias_name, to=self.io_dtype)
        node_name = f"/{name.replace('.', '/')}/LayerNormalization"
        output = f"{node_name}/output_0"
        self.make_node(
            "LayerNormalization",
            [root_input, scale_name, bias_name],
            [output],
            name=node_name,
            axis=-1,
            epsilon=layer_norm.eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape)
        return output
    def make_binary(self, op_type, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node(op_type, inputs, [output], name=name)
        self.make_value(output, dtype, shape)
        return output

    def make_cast(self, name, root_input, dtype, shape):
        super().make_cast(name, root_input, dtype, shape)
        return f"{name}/output_0"
