# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create the deterministic paged model used by the Engine tests.

The graph writes packed tokens through the supplied block table and emits one
FP16 logits row per request for:

    (first_prompt_token + current_token + current_length) % vocab_size
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB_SIZE = 64
NUM_LAYERS = 6
PAGED_LAYERS = [1, 4]
WRITE_LAYER, PASSTHROUGH_LAYER = PAGED_LAYERS
BLOCK_SIZE = 4
NUM_BLOCKS = 128
MAX_BATCH_SIZE = 8
CONTEXT_LENGTH = 128
EOS_TOKEN_ID = 1


def _const(name, array):
    tensor = numpy_helper.from_array(np.asarray(array))
    tensor.name = name
    return tensor


def _decoder_graph():
    i64 = lambda v: np.asarray(v, dtype=np.int64)  # noqa: E731
    initializers = [
        _const("c0", i64(0)),
        _const("c1", i64(1)),
        _const("cB", i64(BLOCK_SIZE)),
        _const("cV", i64(VOCAB_SIZE)),
        _const("axis0", i64([0])),
        _const("axis1", i64([1])),
        _const("start1", i64([1])),
        _const("end_all", i64([np.iinfo(np.int64).max])),
        _const("flat", i64([-1])),
        _const("cache_shape", i64([NUM_BLOCKS, BLOCK_SIZE, 1, 1])),
        _const("vocab_range", np.arange(VOCAB_SIZE, dtype=np.int64).reshape(1, VOCAB_SIZE)),
    ]

    nodes = []

    def node(op_type, inputs, outputs, **attrs):
        nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))

    # Derive each packed token's request row from the cumulative boundaries.
    node("Shape", ["input_ids"], ["ids_shape"])
    node("Squeeze", ["ids_shape"], ["num_tokens"])
    node("Range", ["c0", "num_tokens", "c1"], ["token_index"])
    node("Slice", ["cumulative_sequence_lengths", "start1", "end_all", "axis0"], ["boundaries_i32"])
    node("Cast", ["boundaries_i32"], ["boundaries"], to=TensorProto.INT64)
    node("Unsqueeze", ["token_index", "axis1"], ["token_index_col"])
    node("Unsqueeze", ["boundaries", "axis0"], ["boundaries_row"])
    node("GreaterOrEqual", ["token_index_col", "boundaries_row"], ["at_or_past"])
    node("Cast", ["at_or_past"], ["at_or_past_i64"], to=TensorProto.INT64)
    node("ReduceSum", ["at_or_past_i64", "axis1"], ["row_id"], keepdims=0)

    # Compute the token's absolute position within its request.
    node("Cast", ["cumulative_sequence_lengths"], ["cum_i64"], to=TensorProto.INT64)
    node("Gather", ["cum_i64", "row_id"], ["row_start"], axis=0)
    node("Sub", ["token_index", "row_start"], ["offset_in_row"])
    node("Cast", ["past_sequence_lengths"], ["past_i64"], to=TensorProto.INT64)
    node("Gather", ["past_i64", "row_id"], ["past_of_row"], axis=0)
    node("Add", ["past_of_row", "offset_in_row"], ["pos"])

    # Map each request position to its physical cache slot.
    node("Div", ["pos", "cB"], ["block_col"])
    node("Mul", ["block_col", "cB"], ["block_col_base"])
    node("Sub", ["pos", "block_col_base"], ["slot_in_block"])
    node("Cast", ["block_table"], ["block_table_i64"], to=TensorProto.INT64)
    node("Unsqueeze", ["row_id", "axis1"], ["row_id_col"])
    node("Unsqueeze", ["block_col", "axis1"], ["block_col_col"])
    node("Concat", ["row_id_col", "block_col_col"], ["block_gather_index"], axis=1)
    node("GatherND", ["block_table_i64", "block_gather_index"], ["block_id"])
    node("Mul", ["block_id", "cB"], ["block_base"])
    node("Add", ["block_base", "slot_in_block"], ["phys"])

    # Write tokens to the key and value caches.
    node("Cast", ["input_ids"], ["token_f"], to=TensorProto.FLOAT)
    node("Reshape", [f"past_key_values.{WRITE_LAYER}.key", "flat"], ["past_key_flat"])
    node("Reshape", [f"past_key_values.{WRITE_LAYER}.value", "flat"], ["past_value_flat"])
    node("Unsqueeze", ["phys", "axis1"], ["scatter_index"])
    node("ScatterND", ["past_key_flat", "scatter_index", "token_f"], ["present_key_flat"])
    node("ScatterND", ["past_value_flat", "scatter_index", "token_f"], ["present_value_flat"])
    node("Reshape", ["present_key_flat", "cache_shape"], [f"present.{WRITE_LAYER}.key"])
    node("Reshape", ["present_value_flat", "cache_shape"], [f"present.{WRITE_LAYER}.value"])
    node("Identity", [f"past_key_values.{PASSTHROUGH_LAYER}.key"], [f"present.{PASSTHROUGH_LAYER}.key"])
    node("Identity", [f"past_key_values.{PASSTHROUGH_LAYER}.value"], [f"present.{PASSTHROUGH_LAYER}.value"])

    # Read the request's first token and the current token through the cache.
    node("Gather", ["block_table_i64", "c0"], ["first_block_id"], axis=1)
    node("Mul", ["first_block_id", "cB"], ["first_slot_per_row"])
    node("Gather", ["first_slot_per_row", "row_id"], ["first_slot"], axis=0)
    node("Gather", ["present_key_flat", "first_slot"], ["first_key"], axis=0)
    node("Gather", ["present_value_flat", "phys"], ["cur_value"], axis=0)

    node("Add", ["pos", "c1"], ["current_length"])
    node("Cast", ["current_length"], ["current_length_f"], to=TensorProto.FLOAT)
    node("Add", ["first_key", "cur_value"], ["first_plus_cur"])
    node("Add", ["first_plus_cur", "current_length_f"], ["score_f"])
    node("Cast", ["score_f"], ["score"], to=TensorProto.INT64)
    node("Div", ["score", "cV"], ["score_div"])
    node("Mul", ["score_div", "cV"], ["score_floor"])
    node("Sub", ["score", "score_floor"], ["next_token"])

    node("Unsqueeze", ["next_token", "axis1"], ["next_token_col"])
    node("Equal", ["next_token_col", "vocab_range"], ["is_next_per_token"])

    # Select each request's final packed token so the Engine exercises its
    # [batch_size, vocab_size] output allocation and row mapping. Each boundary
    # is an exclusive end offset, so subtracting one gives the final token index.
    node("Sub", ["boundaries", "c1"], ["last_token_index"])
    node("Gather", ["is_next_per_token", "last_token_index"], ["is_next_per_request"], axis=0)
    node("Cast", ["is_next_per_request"], ["logits"], to=TensorProto.FLOAT16)

    cache_shape = [NUM_BLOCKS, BLOCK_SIZE, 1, 1]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_1"]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch"]),
        helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch", "max_blocks"]),
        helper.make_tensor_value_info("attention_metadata", TensorProto.INT32, [3]),
        *[
            helper.make_tensor_value_info(f"past_key_values.{layer}.key", TensorProto.FLOAT, cache_shape)
            for layer in PAGED_LAYERS
        ],
        *[
            helper.make_tensor_value_info(f"past_key_values.{layer}.value", TensorProto.FLOAT, cache_shape)
            for layer in PAGED_LAYERS
        ],
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch_size", VOCAB_SIZE]),
        *[
            helper.make_tensor_value_info(f"present.{layer}.key", TensorProto.FLOAT, cache_shape)
            for layer in PAGED_LAYERS
        ],
        *[
            helper.make_tensor_value_info(f"present.{layer}.value", TensorProto.FLOAT, cache_shape)
            for layer in PAGED_LAYERS
        ],
    ]
    return helper.make_graph(nodes, "synthetic_paged_decoder", inputs, outputs, initializer=initializers)


def create_decoder(output_dir):
    model = helper.make_model(
        _decoder_graph(),
        opset_imports=[helper.make_operatorsetid("", 17)],
        ir_version=9,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    onnx.checker.check_model(model)
    path = os.path.join(output_dir, "decoder.onnx")
    onnx.save_model(model, path)


def create_config(output_dir):
    config = {
        "model": {
            "type": "decoder",
            "bos_token_id": 0,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder.onnx",
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_size": 1,
                "hidden_size": 1,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "block_table": "block_table",
                    "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                    "past_sequence_lengths": "past_sequence_lengths",
                    "attention_metadata": "attention_metadata",
                    "past_key_names": "legacy_past.%d.key",
                    "past_value_names": "legacy_past.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "legacy_present.%d.key",
                    "present_value_names": "legacy_present.%d.value",
                },
                "state_groups": [
                    {
                        "kind": "paged_kv",
                        "layer_ids": PAGED_LAYERS,
                        "bindings": {
                            "key": {
                                "input": "past_key_values.%d.key",
                                "output": "present.%d.key",
                            },
                            "value": {
                                "input": "past_key_values.%d.value",
                                "output": "present.%d.value",
                            },
                        },
                    }
                ],
            },
        },
        "search": {"max_length": CONTEXT_LENGTH, "do_sample": False},
        "engine": {
            "dynamic_batching": {
                "block_size": BLOCK_SIZE,
                "num_blocks": NUM_BLOCKS,
                "max_batch_size": MAX_BATCH_SIZE,
            },
        },
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "engine", "synthetic-paged"),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    create_decoder(output_dir)
    create_config(output_dir)


if __name__ == "__main__":
    main()
