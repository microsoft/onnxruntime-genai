# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Create a tiny deterministic paged-attention model for the engine tests.

Unlike ``create_dummy_engine_model.py`` (whose graph never runs), this model is
executed end to end by the continuous-batching engine on the paged-attention
path. It is *synthetic*: instead of real attention it computes next-token logits
with a closed-form rule that is easy to predict in Python, yet depends on the
same paged inputs a real ``PagedAttention`` model consumes. That lets the tests
assert exact tokens and catch any row, sequence-length, or block-table mixing.

I/O contract (the engine's variable-length paged decoder, one KV layer):

  inputs
    input_ids                     int64  [num_tokens]        packed 1-D tokens
    cumulative_sequence_lengths   int32  [batch + 1]         per-request offsets
    past_sequence_lengths         int32  [batch]             KV write base / row
    block_table                   int32  [batch, max_blocks] physical block ids
    past_key_values.0.key/value   float  [num_blocks, block_size, 1, 1]
  outputs
    logits                        float  [num_tokens, vocab_size]
    present.0.key/value           float  [num_blocks, block_size, 1, 1]

The key/value caches are bound to the same buffers as their ``past`` inputs, so
the graph writes each token into its slot in place, exactly like the real op.

The rule (see ``predicted_tokens`` in the test helper, which mirrors it):

    slot(pos)  = block_table[row, pos // block_size] * block_size + pos % block_size
    write      key[slot] = value[slot] = token            for every packed token
    first_key  = key  read at the row's first slot   (block_table[row, 0])
    cur_value  = value read at the token's own slot   (its own block/column)
    logits row = one-hot( (first_key + cur_value + (pos + 1)) % vocab_size )

For the row the engine samples (a request's last token) this reduces to

    next_token = (first_prompt_token + current_token + current_length) % vocab_size

so a request's output depends on its first token (only reachable through its
block table after prefill), its current token, and its length. Swap any of those
between requests and the sampled token changes.

Usage:
  python create_synthetic_paged_model.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Tiny synthetic configuration. Small blocks make short prompts span several
# blocks, which is what stresses the block-table addressing.
VOCAB_SIZE = 64
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

    # Row index of every packed token: the number of request boundaries at or
    # below its flat position. boundaries = cumulative_sequence_lengths[1:].
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

    # Absolute sequence position of every packed token:
    #   pos = past_sequence_lengths[row] + (flat_index - row_start)
    node("Cast", ["cumulative_sequence_lengths"], ["cum_i64"], to=TensorProto.INT64)
    node("Gather", ["cum_i64", "row_id"], ["row_start"], axis=0)
    node("Sub", ["token_index", "row_start"], ["offset_in_row"])
    node("Cast", ["past_sequence_lengths"], ["past_i64"], to=TensorProto.INT64)
    node("Gather", ["past_i64", "row_id"], ["past_of_row"], axis=0)
    node("Add", ["past_of_row", "offset_in_row"], ["pos"])

    # Physical slot = block_table[row, pos // block_size] * block_size + pos % block_size.
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

    # Write the token into its slot in both caches, in place.
    node("Cast", ["input_ids"], ["token_f"], to=TensorProto.FLOAT)
    node("Reshape", ["past_key_values.0.key", "flat"], ["past_key_flat"])
    node("Reshape", ["past_key_values.0.value", "flat"], ["past_value_flat"])
    node("Unsqueeze", ["phys", "axis1"], ["scatter_index"])
    node("ScatterND", ["past_key_flat", "scatter_index", "token_f"], ["present_key_flat"])
    node("ScatterND", ["past_value_flat", "scatter_index", "token_f"], ["present_value_flat"])
    node("Reshape", ["present_key_flat", "cache_shape"], ["present.0.key"])
    node("Reshape", ["present_value_flat", "cache_shape"], ["present.0.value"])

    # first_key: the row's first token, reachable only through block_table[row, 0]
    # after prefill. cur_value: this token read back from its own (possibly
    # higher) block-table column.
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

    # One-hot logits so the argmax is exactly next_token.
    node("Unsqueeze", ["next_token", "axis1"], ["next_token_col"])
    node("Equal", ["next_token_col", "vocab_range"], ["is_next"])
    node("Cast", ["is_next"], ["logits"], to=TensorProto.FLOAT)

    cache_shape = [NUM_BLOCKS, BLOCK_SIZE, 1, 1]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_1"]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch"]),
        helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch", "max_blocks"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, cache_shape),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, cache_shape),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["num_tokens", VOCAB_SIZE]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, cache_shape),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, cache_shape),
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
    print(f"  Saved decoder -> {path}")


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
                "num_hidden_layers": 1,
                "inputs": {
                    "input_ids": "input_ids",
                    "block_table": "block_table",
                    "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                    "past_sequence_lengths": "past_sequence_lengths",
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
    print(f"  Saved config -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "engine", "synthetic-paged"
        ),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating synthetic paged engine test model in {output_dir}")
    create_decoder(output_dir)
    create_config(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
