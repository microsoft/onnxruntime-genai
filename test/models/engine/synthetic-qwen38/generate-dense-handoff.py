# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate tiny dense prefill and handoff fixtures for the native counter decoder."""

import struct
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).parent


def value(name, data_type, shape):
    return helper.make_tensor_value_info(name, data_type, shape)


target = onnx.load(ROOT / "decoder.onnx")
for tail, data_type, shape in (
    ("index_buffer", TensorProto.BFLOAT16, [1, 4, 128]),
    ("index_lengths", TensorProto.INT32, [1, 2]),
):
    source, destination = f"past_key_values.0.{tail}", f"present.0.{tail}"
    target.graph.input.append(value(source, data_type, shape))
    target.graph.output.append(value(destination, data_type, shape))
    target.graph.node.append(helper.make_node("Identity", [source], [destination]))
onnx.checker.check_model(target)
onnx.save(target, ROOT / "decoder-handoff.onnx")

inputs = [
    value("input_ids", TensorProto.INT64, [1, "sequence"]),
    value("position_ids", TensorProto.INT64, [1, "sequence"]),
    value("past_position_ids", TensorProto.INT64, [1, "past"]),
    value("past_key_values.0.index_key", TensorProto.BFLOAT16, [1, "past", 128]),
]
outputs = [
    value("present_position_ids", TensorProto.INT64, [1, "total"]),
    value("present.0.index_key", TensorProto.BFLOAT16, [1, "total", 128]),
    value("logits", TensorProto.FLOAT, [1, "sequence", 128]),
    value("hidden_states", TensorProto.BFLOAT16, [1, "sequence", 1]),
]
initializers = [
    helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
    helper.make_tensor("one", TensorProto.INT64, [1], [1]),
    helper.make_tensor("ten", TensorProto.INT64, [1], [10]),
    helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("raw_basis", TensorProto.FLOAT, [128], [128.0] + [0.0] * 127),
    helper.make_tensor("vocab_ids", TensorProto.FLOAT, [128], list(range(128))),
]
nodes = [
    helper.make_node("Concat", ["past_position_ids", "position_ids"], ["present_position_ids"], axis=1),
    helper.make_node("Mul", ["input_ids", "zero"], ["zero_tokens"]),
    helper.make_node("Add", ["zero_tokens", "one"], ["ones"]),
    helper.make_node("Cast", ["ones"], ["float_ones"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["float_ones", "axis2"], ["rows"]),
    helper.make_node("Mul", ["rows", "raw_basis"], ["raw_keys"]),
    helper.make_node("Cast", ["raw_keys"], ["raw_bf16"], to=TensorProto.BFLOAT16),
    helper.make_node("Concat", ["past_key_values.0.index_key", "raw_bf16"], ["present.0.index_key"], axis=1),
    helper.make_node("Add", ["position_ids", "ten"], ["predictions"]),
    helper.make_node("Cast", ["predictions"], ["float_predictions"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["float_predictions", "axis2"], ["hidden_float"]),
    helper.make_node("Cast", ["hidden_float"], ["hidden_states"], to=TensorProto.BFLOAT16),
    helper.make_node("Sub", ["vocab_ids", "hidden_float"], ["distance"]),
    helper.make_node("Mul", ["distance", "distance"], ["distance_squared"]),
    helper.make_node("Neg", ["distance_squared"], ["logits"]),
]
graph = helper.make_graph(nodes, "synthetic-qwen38-dense", inputs, outputs, initializers)
dense = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(dense)
onnx.save(dense, ROOT / "dense-prefill.onnx")

data = bytearray()
index = ["qwen38_qsa_consts 1", "file qsa.data", "epsilon 0.000001", "width 2"]
for name, bits in (("cos", [0x3F80] * 64), ("sin", [0] * 64), ("knorm.0", [0x3F80] * 128)):
    payload = struct.pack(f"<{len(bits)}H", *bits)
    for group in ("target", "head"):
        index.append(f"tensor {group} {name} {len(data)} {len(payload)}")
    data.extend(payload)
(ROOT / "qsa.data").write_bytes(data)
(ROOT / "qsa.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
