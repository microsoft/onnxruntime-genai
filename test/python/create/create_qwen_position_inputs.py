#!/usr/bin/env python3
# Copyright (c) Intel® Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create four-row session metadata for the Qwen position-input component tests."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "qwen-position-inputs" / "four-row.onnx",
    )
    args = parser.parse_args()
    position_shape = [4, "batch", "sequence"]
    mask_shape = ["batch", "total_sequence"]
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["position_ids"], ["positions_out"]),
            helper.make_node("Identity", ["attention_mask"], ["mask_out"]),
        ],
        "qwen_four_row_position_metadata",
        [
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, position_shape),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, mask_shape),
        ],
        [
            helper.make_tensor_value_info("positions_out", TensorProto.INT64, position_shape),
            helper.make_tensor_value_info("mask_out", TensorProto.INT64, mask_shape),
        ],
    )
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
