# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnxruntime
import onnx
import numpy as np
from pathlib import Path

# Create the model with adapters
model = onnx.load(Path("phi-2") / "int4" / "cpu" / "model.onnx")

vocab_size = 51200
adapter_a = onnx.helper.make_tensor_value_info('adapter_a', onnx.TensorProto.FLOAT, [vocab_size])
adapter_b = onnx.helper.make_tensor_value_info('adapter_b', onnx.TensorProto.FLOAT, [vocab_size])

model.graph.input.extend([adapter_a, adapter_b])
add_node = onnx.helper.make_node('Add', ['adapter_a', 'adapter_b'], ['adapter_output'], name='adapter_add')
add_to_logits_node = onnx.helper.make_node('Add', ['adapter_output', 'logits'], ['logits_with_adapter'], name='add_to_logits')
model.graph.node.extend([add_node, add_to_logits_node])

model.graph.output[0].name = 'logits_with_adapter'

onnx.save(model, Path("adapters") / "model.onnx", save_as_external_data=True, location="model.data")

# Create adapters for the model
a = np.random.rand(vocab_size).astype(np.float32)
b = np.random.rand(vocab_size).astype(np.float32)

adapters = {"adapter_a": onnxruntime.OrtValue.ortvalue_from_numpy_with_onnxtype(a, 1), 
            "adapter_b": onnxruntime.OrtValue.ortvalue_from_numpy_with_onnxtype(b, 1)}

adapter_format = onnxruntime.AdapterFormat()
adapter_format.set_adapter_version(1)
adapter_format.set_model_version(1)
adapter_format.set_parameters(adapters)
adapter_format.export_adapter(str(Path("adapters") / "adapters.onnx_adapter"))
