# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create a tiny decoder-only model for the engine unit tests.

The engine unit tests need to mint real ``Request`` objects. Constructing a
``Request`` builds a ``Search`` from the model's ``GeneratorParams`` and, on
assignment, allocates prompt tokens on the model's device. Most tests drive
the scheduler and Engine with recording doubles, while a small public-API
regression also executes the static path. The graph therefore emits
shape-correct zero logits and zero-filled cache outputs. It remains tiny and
free of execution-provider-specific operators.

Usage:
  python create_dummy_engine_model.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Tiny decoder-only configuration.
VOCAB_SIZE = 32
HEAD_SIZE = 4
NUM_KV_HEADS = 2
NUM_LAYERS = 1
CONTEXT_LENGTH = 128


def create_decoder(output_dir):
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["input_shape"]),
        helper.make_node("Concat", ["input_shape", "vocab_dimension"], ["logits_shape"], axis=0),
        helper.make_node("Expand", ["zero_logit", "logits_shape"], ["logits"]),
        helper.make_node("Shape", ["attention_mask"], ["attention_shape"]),
        helper.make_node(
            "Gather",
            ["attention_shape", "cache_batch_index"],
            ["cache_batch_dimension"],
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["attention_shape", "cache_sequence_index"],
            ["cache_sequence_dimension"],
            axis=0,
        ),
        helper.make_node(
            "Concat",
            [
                "cache_batch_dimension",
                "cache_heads_dimension",
                "cache_sequence_dimension",
                "cache_head_dimension",
            ],
            ["present_cache_shape"],
            axis=0,
        ),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="zero_logit"),
        numpy_helper.from_array(np.asarray([VOCAB_SIZE], dtype=np.int64), name="vocab_dimension"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="cache_batch_index"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="cache_sequence_index"),
        numpy_helper.from_array(
            np.asarray([NUM_KV_HEADS], dtype=np.int64),
            name="cache_heads_dimension",
        ),
        numpy_helper.from_array(np.asarray([HEAD_SIZE], dtype=np.int64), name="cache_head_dimension"),
        numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="zero_cache"),
    ]

    for layer in range(NUM_LAYERS):
        for kv in ("key", "value"):
            # This fixture is shared by conventional static-cache tests and paged-cache unit tests.
            # Keep all cache axes symbolic so each path can validate and bind its own layout without
            # treating the other path's concrete axis order as paged geometry.
            cache_shape = [
                "cache_axis_0",
                "cache_axis_1",
                "cache_axis_2",
                "cache_axis_3",
            ]
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{layer}.{kv}",
                    TensorProto.FLOAT,
                    cache_shape,
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{layer}.{kv}",
                    TensorProto.FLOAT,
                    cache_shape,
                )
            )
            nodes.append(
                helper.make_node(
                    "Expand",
                    ["zero_cache", "present_cache_shape"],
                    [f"present.{layer}.{kv}"],
                )
            )

    graph = helper.make_graph(
        nodes=nodes,
        name="dummy_decoder",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
        value_info=[],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    path = os.path.join(output_dir, "decoder.onnx")
    onnx.save_model(model, path)
    print(f"  Saved decoder -> {path}")


def create_config(output_dir):
    config = {
        "model": {
            "type": "decoder",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder.onnx",
                "num_attention_heads": NUM_KV_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
                "head_size": HEAD_SIZE,
                "hidden_size": NUM_KV_HEADS * HEAD_SIZE,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "search": {"max_length": CONTEXT_LENGTH},
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "engine", "dummy-decoder"),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating dummy engine test model in {output_dir}")
    create_decoder(output_dir)
    create_config(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
