# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create a tiny decoder-only model for the engine unit tests.

The engine unit tests need to mint real ``Request`` objects. Constructing a
``Request`` builds a ``Search`` from the model's ``GeneratorParams`` and, on
assignment, allocates prompt tokens on the model's device. None of that runs
the ONNX graph: the tests drive the scheduler and engine with recording test
doubles that stand in for the cache manager and the model executor, so the
model never performs a forward pass.

Because the graph is never executed, it only has to *load* as a decoder-only
``Generators::Model`` on CPU. This script therefore emits a minimal graph that
declares the decoder inputs and outputs (and zero-filled output initializers)
but contains no compute nodes, following the same "contents don't matter"
approach as ``create_dummy_model.py``. This keeps the checked-in artifact tiny
and free of any execution-provider-specific operators.

Usage:
  python create_dummy_engine_model.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Tiny decoder-only configuration. The values only need to be internally
# consistent so the model loads; they are never used to compute anything.
VOCAB_SIZE = 32
HEAD_SIZE = 4
NUM_KV_HEADS = 2
NUM_LAYERS = 1
CONTEXT_LENGTH = 128


def _zeros(name, shape, dtype=np.float32):
    tensor = numpy_helper.from_array(np.zeros(shape, dtype=dtype))
    tensor.name = name
    return tensor


def create_decoder(output_dir):
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE]),
    ]
    initializers = [
        _zeros("logits", (2, 2, VOCAB_SIZE)),
    ]

    for layer in range(NUM_LAYERS):
        for kv in ("key", "value"):
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{layer}.{kv}",
                    TensorProto.FLOAT,
                    ["batch_size", NUM_KV_HEADS, "past_sequence_length", HEAD_SIZE],
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{layer}.{kv}",
                    TensorProto.FLOAT,
                    ["batch_size", NUM_KV_HEADS, "total_sequence_length", HEAD_SIZE],
                )
            )
            initializers.append(_zeros(f"present.{layer}.{kv}", (2, NUM_KV_HEADS, 2, HEAD_SIZE)))

    graph = helper.make_graph(
        nodes=[],
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
            "type": "gpt2",
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
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "engine", "dummy-decoder"
        ),
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
