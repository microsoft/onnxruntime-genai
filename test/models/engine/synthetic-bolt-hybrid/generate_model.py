# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Generate a small structural fixture for Bolt package conversion, not inference."""

from pathlib import Path

import numpy as np
import onnx_ir as ir


def main() -> None:
    def value(name, dtype, shape):
        return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

    hidden = value("hidden_states", ir.DataType.BFLOAT16, [1, 8, 16])
    tokens = value("input_ids", ir.DataType.INT64, [1, 8])
    weight = value("weight", ir.DataType.UINT8, [16, 1, 8])
    weight.const_value = ir.ExternalTensor(
        "weights.bin", offset=0, length=128, dtype=ir.DataType.UINT8, shape=weight.shape, name=weight.name
    )
    scales = value("scales", ir.DataType.FLOAT, [16, 1])
    scales.const_value = ir.tensor(np.ones((16, 1), dtype=np.float32))
    nodes = []
    for index in range(8):
        projections = []
        for name in ("query", "key", "value"):
            projection = value(f"{name}_{index}", ir.DataType.BFLOAT16, [1, 8, 16])
            nodes.append(
                ir.Node(
                    "com.microsoft",
                    "MatMulNBits",
                    [hidden, weight, scales],
                    outputs=[projection],
                    attributes={
                        key: ir.AttrInt64(key, number)
                        for key, number in (("K", 16), ("N", 16), ("bits", 4), ("block_size", 16))
                    },
                )
            )
            projections.append(projection)
        attention = value(f"attention_{index}", ir.DataType.BFLOAT16, [1, 8, 16])
        nodes.append(
            ir.Node(
                "com.microsoft",
                "GroupQueryAttention",
                projections,
                outputs=[attention],
                attributes={"num_heads": ir.AttrInt64("num_heads", 1), "kv_num_heads": ir.AttrInt64("kv_num_heads", 1)},
            )
        )
    logits = value("logits", ir.DataType.BFLOAT16, [1, 8, 16])
    auxiliary = value("aux_hidden_states", ir.DataType.BFLOAT16, [1, 8, 16])
    predictions = value("pred_ids", ir.DataType.INT64, [1, 8])
    nodes.extend(
        [
            ir.Node("", "Identity", [attention], outputs=[logits]),
            ir.Node("", "Identity", [attention], outputs=[auxiliary]),
            ir.Node(
                "",
                "ArgMax",
                [attention],
                outputs=[predictions],
                attributes={"axis": ir.AttrInt64("axis", 2), "keepdims": ir.AttrInt64("keepdims", 0)},
            ),
        ]
    )
    graph = ir.Graph(
        [tokens, hidden],
        [logits, predictions, auxiliary],
        nodes=nodes,
        initializers=[weight, scales],
        opset_imports={"": 21, "com.microsoft": 1},
        name="bolt_hybrid_conversion",
    )
    ir.save(ir.Model(graph, ir_version=10), Path(__file__).with_name("model.onnx"))


if __name__ == "__main__":
    main()
