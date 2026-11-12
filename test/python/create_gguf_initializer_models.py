# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create GGUF loader fixtures whose external data must be supplied by name."""

from pathlib import Path

from onnx import TensorProto, helper, save


def create_external_tensor(name, shape, offset, length):
    tensor = TensorProto(name=name, data_type=TensorProto.FLOAT, dims=shape)
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in [
        ("location", "missing.data"),
        ("offset", str(offset)),
        ("length", str(length)),
    ]:
        tensor.external_data.add(key=key, value=value)
    return tensor


def save_model(graph, output_path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    save(model, output_path)


def create_model(output_path, scalar=False):
    weight_shape = [] if scalar else [1]
    initializers = [
        create_external_tensor(name, shape, offset, length)
        for name, shape, offset, length in [
            ("weight", weight_shape, 0, 4),
            ("other_weight", [2], 4, 8),
        ]
    ]

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["weight"], ["weight_out"]),
            helper.make_node("Identity", ["other_weight"], ["other_out"]),
        ],
        "gguf-initializers",
        [],
        [
            helper.make_tensor_value_info("weight_out", TensorProto.FLOAT, weight_shape),
            helper.make_tensor_value_info("other_out", TensorProto.FLOAT, [2]),
        ],
        initializer=initializers,
    )
    save_model(graph, output_path)


def create_large_model(output_path, count=64):
    # A 256-byte initializer is large enough for ONNX Runtime to borrow, and adding a runtime input keeps the
    # session reading it during Run instead of constant folding it away.
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "large_weight"], ["y"])],
        "gguf-large-initializer",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [count])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [count])],
        initializer=[create_external_tensor("large_weight", [count], 0, count * 4)],
    )
    save_model(graph, output_path)


if __name__ == "__main__":
    output_dir = Path(__file__).parents[1] / "models" / "gguf-initializers"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_model(output_dir / "model.onnx")
    create_model(output_dir / "scalar.onnx", scalar=True)
    create_large_model(output_dir / "large.onnx")
