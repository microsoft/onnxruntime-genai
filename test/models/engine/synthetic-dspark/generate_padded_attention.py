# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

from pathlib import Path

import onnx_ir as ir


def main() -> None:
    def value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
        return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

    inputs = [value(name, ir.DataType.FLOAT16, ["tokens", 64]) for name in ("query", "key", "value")]
    inputs.extend(value(name, ir.DataType.FLOAT16, [2, 16, 1, 64]) for name in ("key_cache", "value_cache"))
    inputs.extend(
        [
            value("cumulative_sequence_length", ir.DataType.INT32, [2]),
            value("past_seqlens", ir.DataType.INT32, [1]),
            value("block_table", ir.DataType.INT32, [1, 2]),
            value("attention_metadata", ir.DataType.INT32, [3]),
        ]
    )
    attention = ir.Node(
        "com.microsoft",
        "PagedAttention",
        [*inputs[:8], *([None] * 8), inputs[8]],
        attributes={
            name: ir.AttrInt64(name, number)
            for name, number in (("num_heads", 1), ("kv_num_heads", 1), ("is_causal", 0))
        },
        num_outputs=3,
    )
    for output, name, shape in zip(
        attention.outputs,
        ("output", "key_cache_out", "value_cache_out"),
        (["tokens", 64], [2, 16, 1, 64], [2, 16, 1, 64]),
        strict=True,
    ):
        output.name = name
        output.type = ir.TensorType(ir.DataType.FLOAT16)
        output.shape = ir.Shape(shape)
    graph = ir.Graph(
        inputs,
        attention.outputs,
        nodes=[attention],
        opset_imports={"": 21, "com.microsoft": 1},
        name="padded_attention",
    )
    ir.save(ir.Model(graph, ir_version=10), Path(__file__).with_name("padded_attention.onnx"))


if __name__ == "__main__":
    main()
