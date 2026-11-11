# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate a Qwen38 target whose in-place shared-state counter exposes repeated ORT Runs."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def value(name, data_type, shape):
    return helper.make_tensor_value_info(name, data_type, shape)


inputs = [
    value("input_ids", TensorProto.INT64, [1, "sequence"]),
    value("past_position_ids", TensorProto.INT64, [1, 32]),
    value("past_key_values.0.index_key", TensorProto.FLOAT, [1, 8, 128]),
    value("ple_embeddings.0", TensorProto.BFLOAT16, [1, "sequence", 1]),
]
outputs = [
    value("present_position_ids", TensorProto.INT64, [1, 32]),
    value("present.0.index_key", TensorProto.FLOAT, [1, 8, 128]),
    value("next_tokens", TensorProto.INT64, [1, "sequence"]),
    value("logits", TensorProto.BFLOAT16, [1, "sequence", 128]),
    value("hidden_states", TensorProto.BFLOAT16, [1, "sequence", 1]),
    value("validation.aggregate", TensorProto.INT32, [1]),
]
initializers = [
    helper.make_tensor("one", TensorProto.INT64, [1], [1]),
    helper.make_tensor("counter_increment", TensorProto.FLOAT, [1], [1.0]),
    helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
    helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("vocab_ids", TensorProto.FLOAT, [128], list(range(128))),
]
nodes = [
    helper.make_node("Identity", ["past_position_ids"], ["present_position_ids"]),
    helper.make_node("Add", ["past_key_values.0.index_key", "counter_increment"], ["present.0.index_key"]),
    helper.make_node("Gather", ["present.0.index_key", "zero"], ["counter_row"], axis=1),
    helper.make_node("Gather", ["counter_row", "zero"], ["counter_value"], axis=2),
    helper.make_node("Reshape", ["counter_value", "one"], ["counter_float"]),
    helper.make_node("Cast", ["counter_float"], ["counter"], to=TensorProto.INT64),
    helper.make_node("Mul", ["input_ids", "zero"], ["zero_tokens"]),
    helper.make_node("Add", ["zero_tokens", "counter"], ["next_tokens"]),
    helper.make_node("Cast", ["next_tokens"], ["float_tokens"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["float_tokens", "axis2"], ["hidden_float"]),
    helper.make_node("Sub", ["vocab_ids", "hidden_float"], ["distance"]),
    helper.make_node("Mul", ["distance", "distance"], ["distance_squared"]),
    helper.make_node("Neg", ["distance_squared"], ["float_logits"]),
    helper.make_node("Cast", ["float_logits"], ["logits"], to=TensorProto.BFLOAT16),
    helper.make_node("Cast", ["hidden_float"], ["hidden_states"], to=TensorProto.BFLOAT16),
    helper.make_node("Gather", ["zero_tokens", "zero"], ["zero_status"], axis=1),
    helper.make_node("Reshape", ["zero_status", "one"], ["status"]),
    helper.make_node("Cast", ["status"], ["validation.aggregate"], to=TensorProto.INT32),
]
graph = helper.make_graph(nodes, "synthetic-qwen38", inputs, outputs, initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
helper.set_model_props(model, {"experiment.validation_status_names": '["status"]'})
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("decoder.onnx"))
Path(__file__).with_name("ple.data").write_bytes(b"\0")
