# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Create tiny input-metadata fixtures for the Qwen position-input C++ tests."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    output_dir = Path(__file__).resolve().parents[2] / "models" / "qwen-position-inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dtype in (("int32", TensorProto.INT32), ("int64", TensorProto.INT64)):
        inputs = [
            helper.make_tensor_value_info("position_ids", dtype, [3, "batch", "sequence"]),
            helper.make_tensor_value_info("attention_mask", dtype, ["batch", "total"]),
        ]
        outputs = [
            helper.make_tensor_value_info("positions_out", dtype, [3, "batch", "sequence"]),
            helper.make_tensor_value_info("mask_out", dtype, ["batch", "total"]),
        ]
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["position_ids"], ["positions_out"]),
                helper.make_node("Identity", ["attention_mask"], ["mask_out"]),
            ],
            "position_inputs",
            inputs,
            outputs,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model, output_dir / f"{name}.onnx")


if __name__ == "__main__":
    main()
