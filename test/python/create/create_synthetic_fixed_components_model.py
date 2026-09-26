# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Create the fixed PLE/indexer model used by Engine state-pool tests."""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB_SIZE = 16
NUM_LAYERS = 2
PLE_TOKEN_PAD_ID = 7


def create_decoder(output_dir):
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB_SIZE]),
    ]
    initializers = [numpy_helper.from_array(np.zeros((2, 2, VOCAB_SIZE), dtype=np.float32), "logits")]
    nodes = []

    state_specs = [
        ("ple_tokens", 0, TensorProto.INT64, [2]),
        ("ple_conv", 0, TensorProto.FLOAT, [4, 3]),
        ("indexer_key", 1, TensorProto.FLOAT, [8, 2]),
        ("indexer_kv_buffer", 1, TensorProto.FLOAT, [3, 2]),
        ("indexer_state_lengths", 1, TensorProto.INT32, [2]),
    ]
    for semantic, layer_id, data_type, row_shape in state_specs:
        input_name = f"past.{layer_id}.{semantic}"
        output_name = f"present.{layer_id}.{semantic}"
        shape = ["batch_size", *row_shape]
        inputs.append(helper.make_tensor_value_info(input_name, data_type, shape))
        outputs.append(helper.make_tensor_value_info(output_name, data_type, shape))
        nodes.append(helper.make_node("Identity", [input_name], [output_name]))

    graph = helper.make_graph(
        nodes,
        "synthetic_fixed_components_decoder",
        inputs,
        outputs,
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, os.path.join(output_dir, "decoder.onnx"))


def create_config(output_dir):
    config = {
        "model": {
            "type": "gpt2",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": 128,
            "decoder": {
                "filename": "decoder.onnx",
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "hidden_size": 1,
                "num_hidden_layers": NUM_LAYERS,
                "ple_token_pad_id": PLE_TOKEN_PAD_ID,
                "session_options": {"provider_options": []},
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_ple_token_names": "past.%d.ple_tokens",
                    "past_ple_conv_names": "past.%d.ple_conv",
                    "past_indexer_names": "past.%d.indexer_key",
                    "past_indexer_kv_buffer_names": "past.%d.indexer_kv_buffer",
                    "past_indexer_state_lengths_names": "past.%d.indexer_state_lengths",
                },
                "outputs": {
                    "logits": "logits",
                    "present_ple_token_names": "present.%d.ple_tokens",
                    "present_ple_conv_names": "present.%d.ple_conv",
                    "present_indexer_names": "present.%d.indexer_key",
                    "present_indexer_kv_buffer_names": "present.%d.indexer_kv_buffer",
                    "present_indexer_state_lengths_names": "present.%d.indexer_state_lengths",
                },
                "state_groups": [
                    {"kind": "fixed_ple", "layer_ids": [0]},
                    {"kind": "fixed_indexer", "layer_ids": [1]},
                ],
            },
        },
        "search": {"max_length": 128, "do_sample": False},
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as file:
        json.dump(config, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "engine", "synthetic-fixed-components"
        ),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    create_decoder(output_dir)
    create_config(output_dir)


if __name__ == "__main__":
    main()
