# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Create small state-binding sessions for recurrent-window compatibility tests."""

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

output_dir = Path(__file__).resolve().parents[2] / "models" / "engine" / "synthetic-rotary-history"
output_dir.mkdir(exist_ok=True)
for name, recurrent_shape, compact in (
    ("windowed", [4, 1, 1, 2, 2], False),
    ("mismatched", [3, 1, 1, 2, 2], False),
    ("compact", [1, 1, 2, 2], True),
):
    inputs = [
        helper.make_tensor_value_info("past.0.conv", TensorProto.FLOAT, [4, 1, 2, 1]),
        helper.make_tensor_value_info("past.0.recurrent", TensorProto.FLOAT, recurrent_shape),
    ]
    outputs = [
        helper.make_tensor_value_info("present.0.conv", TensorProto.FLOAT, [4, 1, 2, 1]),
        helper.make_tensor_value_info("present.0.recurrent", TensorProto.FLOAT, recurrent_shape),
    ]
    nodes = [
        helper.make_node("Identity", ["past.0.conv"], ["present.0.conv"]),
        helper.make_node("Identity", ["past.0.recurrent"], ["present.0.recurrent"]),
    ]
    if compact:
        inputs.extend(
            [
                helper.make_tensor_value_info("state_update_capture_count", TensorProto.INT32, [1]),
                helper.make_tensor_value_info("state_update_active", TensorProto.INT32, [1]),
            ]
        )
        outputs.append(helper.make_tensor_value_info("state_update.0.recurrent_capsule", TensorProto.FLOAT, [1, 15]))
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["state_update.0.recurrent_capsule"],
                value=numpy_helper.from_array(np.zeros((1, 15), dtype=np.float32)),
            )
        )
    model = helper.make_model(
        helper.make_graph(nodes, f"recurrent_{name}", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    save(model, output_dir / f"{name}.onnx")

engram = helper.make_graph(
    [
        helper.make_node("Cast", ["engram_input_ids"], ["ids_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_float", "embedding_axis"], ["engram_embeddings"]),
        helper.make_node("Concat", ["engram_past_tokens", "engram_input_ids"], ["history"], axis=1),
        helper.make_node("Slice", ["history", "tail_start", "tail_end", "history_axis"], ["engram_present_tokens"]),
    ],
    "engram_cache_review",
    [
        helper.make_tensor_value_info("engram_input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("engram_past_tokens", TensorProto.INT64, ["batch", 2]),
    ],
    [
        helper.make_tensor_value_info("engram_embeddings", TensorProto.FLOAT, ["batch", "sequence", 1]),
        helper.make_tensor_value_info("engram_present_tokens", TensorProto.INT64, ["batch", 2]),
    ],
    [
        numpy_helper.from_array(np.array([2], dtype=np.int64), "embedding_axis"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "history_axis"),
        numpy_helper.from_array(np.array([-2], dtype=np.int64), "tail_start"),
        numpy_helper.from_array(np.array([2**63 - 1], dtype=np.int64), "tail_end"),
    ],
)
save(helper.make_model(engram, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), output_dir / "engram.onnx")

for name, width in (("symbolic", "hidden_width"), ("static", 16)):
    shape = ["batch", "sequence", width]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden_states"], ["hidden_states_out"])],
        f"hidden_states_{name}",
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, shape)],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10),
        output_dir / f"hidden-{name}.onnx",
    )

indexer = helper.make_graph(
    [helper.make_node("Identity", ["past.0.indexer_key"], ["present.0.indexer_key"])],
    "dense_indexer_history",
    [
        helper.make_tensor_value_info("past.0.indexer_key", TensorProto.FLOAT, [1, 64, 4]),
        helper.make_tensor_value_info("past_sequence_length", TensorProto.INT32, [1]),
    ],
    [helper.make_tensor_value_info("present.0.indexer_key", TensorProto.FLOAT, [1, 64, 4])],
)
save(
    helper.make_model(indexer, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), output_dir / "indexer.onnx"
)

ple_inputs = [
    helper.make_tensor_value_info("past.0.ple_tokens", TensorProto.INT64, [1, 2]),
    helper.make_tensor_value_info("past.0.ple_conv", TensorProto.FLOAT, [1, 2, 4]),
]
ple_outputs = [
    helper.make_tensor_value_info("present.0.ple_tokens", TensorProto.INT64, [1, 2]),
    helper.make_tensor_value_info("present.0.ple_conv", TensorProto.FLOAT, [1, 2, 4]),
]
ple_graph = helper.make_graph(
    [
        helper.make_node("Identity", [source.name], [target.name])
        for source, target in zip(ple_inputs, ple_outputs, strict=True)
    ],
    "ple_reset_review",
    ple_inputs,
    ple_outputs,
)
save(helper.make_model(ple_graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10), output_dir / "ple.onnx")
