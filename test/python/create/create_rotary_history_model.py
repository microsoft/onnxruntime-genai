# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Create the small session used by rotary position-history component tests."""

from pathlib import Path

from onnx import TensorProto, helper, save

output_dir = Path(__file__).resolve().parents[2] / "models" / "engine" / "synthetic-rotary-history"
output_dir.mkdir(exist_ok=True)
inputs = [
    helper.make_tensor_value_info("position_ids", TensorProto.INT64, [3, "batch_size", "sequence_length"]),
    helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
    helper.make_tensor_value_info("rotary_position_ids", TensorProto.INT64, [3, "batch_size", "history_length"]),
]
outputs = [
    helper.make_tensor_value_info(
        f"{value.name}_out",
        TensorProto.INT64,
        [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in value.type.tensor_type.shape.dim],
    )
    for value in inputs
]
nodes = [helper.make_node("Identity", [value.name], [f"{value.name}_out"]) for value in inputs]
model = helper.make_model(
    helper.make_graph(nodes, "rotary_position_history", inputs, outputs),
    opset_imports=[helper.make_opsetid("", 21)],
    ir_version=10,
)
save(model, output_dir / "positions.onnx")
