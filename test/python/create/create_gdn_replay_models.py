# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate tiny real-ORT GatedDeltaNet fixtures for fixed-state replay parity."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def create_model(heads, key_heads, value_width, key_width, path):
    capacity = 7
    inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, ["tokens", key_heads, key_width]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, ["tokens", key_heads, key_width]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, ["tokens", heads, value_width]),
        helper.make_tensor_value_info("decay", TensorProto.FLOAT, ["tokens", heads]),
        helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["tokens", heads]),
        helper.make_tensor_value_info("initial_state", TensorProto.FLOAT, [1, heads, value_width, key_width]),
        helper.make_tensor_value_info("capture_count", TensorProto.INT32, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["tokens", heads, value_width]),
        helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [1, heads, value_width, key_width]),
        helper.make_tensor_value_info(
            "state_update", TensorProto.FLOAT, [1, capacity * (heads + key_heads * key_width + heads * value_width)]
        ),
    ]
    node = helper.make_node(
        "GatedDeltaNet",
        ["query", "key", "value", "", "decay", "beta", "initial_state", "", "", "capture_count"],
        ["output", "final_state", "state_update"],
        domain="com.microsoft",
        state_update_capacity=capacity,
    )
    model = helper.make_model(
        helper.make_graph([node], "gdn_replay", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[2] / "models/engine/synthetic-hybrid"
    create_model(4, 2, 40, 128, output / "gdn-replay-fast.onnx")
    create_model(3, 3, 5, 6, output / "gdn-replay-generic.onnx")
