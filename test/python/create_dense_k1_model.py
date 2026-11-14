# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a tiny real dense-GQA decoder for K1 tests, without downloaded weights."""

import json
from pathlib import Path

from onnx import TensorProto, checker, helper, save


def create_model(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", 8]),
    ]
    for layer in range(2):
        for kind in ("key", "value"):
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{layer}.{kind}", TensorProto.FLOAT, ["batch", 1, "past_capacity", 4]
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{layer}.{kind}", TensorProto.FLOAT, ["batch", 1, "present_capacity", 4]
                )
            )
    constants = [
        helper.make_tensor(
            "embeddings", TensorProto.FLOAT, [8, 8], [((t + 1) * (d + 1) % 17) / 17 for t in range(8) for d in range(8)]
        ),
        helper.make_tensor(
            "projection", TensorProto.FLOAT, [8, 8], [float(i == j) for i in range(8) for j in range(8)]
        ),
        helper.make_tensor("bias", TensorProto.FLOAT, [8], [0, 10, 0, 0, 0, 0, 0, 0]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [4]),
        helper.make_tensor("slice_axes", TensorProto.INT64, [1], [2]),
        helper.make_tensor("reduce_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("sequence_axis", TensorProto.INT64, [], [1]),
        helper.make_tensor("one", TensorProto.INT32, [], [1]),
    ]
    nodes = [
        helper.make_node("Gather", ["embeddings", "input_ids"], ["query"], axis=0),
        helper.make_node("Slice", ["query", "slice_starts", "slice_ends", "slice_axes"], ["key_value"]),
        helper.make_node("ReduceSum", ["attention_mask", "reduce_axes"], ["sequence_lengths"], keepdims=0),
        helper.make_node("Cast", ["sequence_lengths"], ["sequence_lengths_i32"], to=TensorProto.INT32),
        helper.make_node("Sub", ["sequence_lengths_i32", "one"], ["seqlens_k"]),
        helper.make_node("Shape", ["attention_mask"], ["attention_shape"]),
        helper.make_node("Gather", ["attention_shape", "sequence_axis"], ["total_length_i64"], axis=0),
        helper.make_node("Cast", ["total_length_i64"], ["total_length"], to=TensorProto.INT32),
    ]
    for layer in range(2):
        nodes.append(
            helper.make_node(
                "GroupQueryAttention",
                [
                    "query" if layer == 0 else f"attention_output_{layer - 1}",
                    "key_value",
                    "key_value",
                    f"past_key_values.{layer}.key",
                    f"past_key_values.{layer}.value",
                    "seqlens_k",
                    "total_length",
                ],
                [f"attention_output_{layer}", f"present.{layer}.key", f"present.{layer}.value"],
                domain="com.microsoft",
                num_heads=2,
                kv_num_heads=1,
            )
        )
    nodes += [
        helper.make_node("MatMul", ["attention_output_1", "projection"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "dense-k1", inputs, outputs, initializer=constants)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)], ir_version=9
    )
    checker.check_model(model)
    save(model, output_dir / "model.onnx")
    config = {
        "model": {
            "type": "llama",
            "context_length": 32,
            "vocab_size": 8,
            "pad_token_id": 0,
            "eos_token_id": 7,
            "decoder": {
                "filename": "model.onnx",
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_size": 4,
                "hidden_size": 8,
                "session_options": {"intra_op_num_threads": 1},
            },
        },
        "search": {"max_length": 16, "past_present_share_buffer": False, "do_sample": False},
    }
    (output_dir / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    create_model(Path(__file__).parents[1] / "models" / "dense-k1")
