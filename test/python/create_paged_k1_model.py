# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a tiny native PagedAttention decoder for K1-v2 tests (CUDA required)."""

import json
from pathlib import Path

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper, save


def create_model(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["tokens"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_one"]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch"]),
        helper.make_tensor_value_info("attention_metadata", TensorProto.INT32, [3]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1, 8])]
    for layer in range(2):
        inputs.append(helper.make_tensor_value_info(f"block_table.{layer}", TensorProto.INT32, ["batch", "columns"]))
        for kind in ("key", "value"):
            inputs.append(
                helper.make_tensor_value_info(f"past_key_values.{layer}.{kind}", TensorProto.FLOAT16, [4, 16, 1, 64])
            )
            outputs.append(
                helper.make_tensor_value_info(f"present.{layer}.{kind}", TensorProto.FLOAT16, [4, 16, 1, 64])
            )
    initializers = [
        numpy_helper.from_array(
            np.asarray([[(t + 1) * (d + 3) % 19 / 19 for d in range(64)] for t in range(8)], dtype=np.float16),
            "embeddings",
        ),
        numpy_helper.from_array(
            np.asarray([[(d + 5) * (t + 1) % 23 / 23 for t in range(8)] for d in range(64)], dtype=np.float16),
            "projection",
        ),
        numpy_helper.from_array(np.asarray([0] * 7 + [-100], dtype=np.float32), "bias"),
        helper.make_tensor("start_one", TensorProto.INT64, [1], [1]),
        helper.make_tensor("end_all", TensorProto.INT64, [1], [np.iinfo(np.int64).max]),
        helper.make_tensor("axis_zero", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axis_one", TensorProto.INT64, [1], [1]),
        helper.make_tensor("one_i32", TensorProto.INT32, [], [1]),
    ]
    nodes = [helper.make_node("Gather", ["embeddings", "input_ids"], ["embedded"], axis=0)]
    for layer in range(2):
        hidden = "embedded" if layer == 0 else f"attention_{layer - 1}"
        nodes.append(
            helper.make_node(
                "PagedAttention",
                [
                    hidden,
                    hidden,
                    hidden,
                    f"past_key_values.{layer}.key",
                    f"past_key_values.{layer}.value",
                    "cumulative_sequence_lengths",
                    "past_sequence_lengths",
                    f"block_table.{layer}",
                    *([""] * 8),
                    "attention_metadata",
                ],
                [f"attention_{layer}", f"present.{layer}.key", f"present.{layer}.value"],
                domain="com.microsoft",
                num_heads=1,
                kv_num_heads=1,
            )
        )
    nodes += [
        helper.make_node("Slice", ["cumulative_sequence_lengths", "start_one", "end_all", "axis_zero"], ["ends"]),
        helper.make_node("Sub", ["ends", "one_i32"], ["last_indices"]),
        helper.make_node("Gather", ["attention_1", "last_indices"], ["last_hidden"], axis=0),
        helper.make_node("MatMul", ["last_hidden", "projection"], ["projected_fp16"]),
        helper.make_node("Cast", ["projected_fp16"], ["projected"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["projected", "bias"], ["scores"]),
        helper.make_node("Unsqueeze", ["scores", "axis_one"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "paged-k1", inputs, outputs, initializer=initializers)
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
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 64,
                "hidden_size": 64,
                "inputs": {"block_table": "block_table.%d"},
                "session_options": {"intra_op_num_threads": 1},
            },
        },
        "search": {"max_length": 32, "num_beams": 1, "do_sample": False},
        "engine": {
            "dynamic_batching": {
                "block_size": 16,
                "num_blocks": 4,
                "max_batch_size": 2,
                "max_scheduled_tokens": 64,
                "prefix_caching": False,
            }
        },
    }
    (output_dir / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    create_model(Path(__file__).parents[1] / "models" / "paged-k1")
