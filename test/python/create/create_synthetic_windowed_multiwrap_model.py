# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Create the deterministic sliding-window Engine continuation fixture.

The graph models two paged KV-cache layers:

* layer 0 keeps the full sequence;
* layer 1 uses the runtime's repeated sliding-window block table.

Both layers store token-and-position encodings. Logits read those encodings back
through both block tables, and an invariant-failure token is selected if the
window table does not repeat, its blocks change across continuation, or its
live cache values disagree with the full cache.
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB_SIZE = 32
BLOCK_SIZE = 2
WINDOW_SIZE = 3
CHUNK_SIZE = 2
MAX_BATCH_SIZE = 1
RING_BLOCKS = (CHUNK_SIZE + WINDOW_SIZE - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
NUM_FULL_BLOCKS = 8
NUM_WINDOW_BLOCKS = RING_BLOCKS * MAX_BATCH_SIZE
CONTEXT_LENGTH = 16
EOS_TOKEN_ID = 1
INVARIANT_FAILURE_TOKEN_ID = 31


def _const(name, array):
    tensor = numpy_helper.from_array(np.asarray(array))
    tensor.name = name
    return tensor


def _decoder_graph():
    def i64(value):
        return np.asarray(value, dtype=np.int64)

    full_cache_shape = [NUM_FULL_BLOCKS, BLOCK_SIZE, 1, 1]
    window_cache_shape = [NUM_WINDOW_BLOCKS, BLOCK_SIZE, 1, 1]
    initializers = [
        _const("c0", i64(0)),
        _const("c1", i64(1)),
        _const("c2", i64(2)),
        _const("c3", i64(3)),
        _const("c5", i64(5)),
        _const("c7", i64(7)),
        _const("c13", i64(13)),
        _const("c28", i64(28)),
        _const("cB", i64(BLOCK_SIZE)),
        _const("cR", i64(RING_BLOCKS)),
        _const("cEOS", i64(EOS_TOKEN_ID)),
        _const("cInvariantFailure", i64(INVARIANT_FAILURE_TOKEN_ID)),
        _const("axis0", i64([0])),
        _const("axis1", i64([1])),
        _const("start1", i64([1])),
        _const("end_all", i64([np.iinfo(np.int64).max])),
        _const("flat", i64([-1])),
        _const("full_cache_shape", i64(full_cache_shape)),
        _const("window_cache_shape", i64(window_cache_shape)),
        _const(
            "vocab_range",
            np.arange(VOCAB_SIZE, dtype=np.int64).reshape(1, VOCAB_SIZE),
        ),
    ]

    nodes = []

    def node(op_type, inputs, outputs, *, name=None, **attrs):
        if name is not None:
            attrs["name"] = name
        nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))

    # Resolve each packed token to a request row and an absolute request position.
    node("Shape", ["input_ids"], ["ids_shape"])
    node("Squeeze", ["ids_shape"], ["num_tokens"])
    node("Range", ["c0", "num_tokens", "c1"], ["token_index"])
    node(
        "Slice",
        ["cumulative_sequence_lengths", "start1", "end_all", "axis0"],
        ["boundaries_i32"],
    )
    node("Cast", ["boundaries_i32"], ["boundaries"], to=TensorProto.INT64)
    node("Unsqueeze", ["token_index", "axis1"], ["token_index_col"])
    node("Unsqueeze", ["boundaries", "axis0"], ["boundaries_row"])
    node("GreaterOrEqual", ["token_index_col", "boundaries_row"], ["at_or_past"])
    node("Cast", ["at_or_past"], ["at_or_past_i64"], to=TensorProto.INT64)
    node("ReduceSum", ["at_or_past_i64", "axis1"], ["row_id"], keepdims=0)

    node(
        "Cast",
        ["cumulative_sequence_lengths"],
        ["cumulative_sequence_lengths_i64"],
        to=TensorProto.INT64,
    )
    node(
        "Gather",
        ["cumulative_sequence_lengths_i64", "row_id"],
        ["row_start"],
        axis=0,
    )
    node("Sub", ["token_index", "row_start"], ["offset_in_row"])
    node(
        "Cast",
        ["past_sequence_lengths"],
        ["past_sequence_lengths_i64"],
        to=TensorProto.INT64,
    )
    node(
        "Gather",
        ["past_sequence_lengths_i64", "row_id"],
        ["past_of_row"],
        axis=0,
    )
    node("Add", ["past_of_row", "offset_in_row"], ["pos"])
    node("Sub", ["pos", "c1"], ["previous_pos_unclamped"])
    node("Max", ["previous_pos_unclamped", "c0"], ["previous_pos"])

    node("Cast", ["block_table"], ["block_table_i64"], to=TensorProto.INT64)
    node(
        "Cast",
        ["block_table_windowed"],
        ["block_table_windowed_i64"],
        to=TensorProto.INT64,
    )
    node("Unsqueeze", ["row_id", "axis1"], ["row_id_col"])

    def map_positions(prefix, positions, table):
        block_col = f"{prefix}_block_col"
        block_col_base = f"{prefix}_block_col_base"
        slot = f"{prefix}_slot_in_block"
        block_col_col = f"{prefix}_block_col_col"
        gather_index = f"{prefix}_block_gather_index"
        block_id = f"{prefix}_block_id"
        block_base = f"{prefix}_block_base"
        physical = f"{prefix}_physical"

        node("Div", [positions, "cB"], [block_col])
        node("Mul", [block_col, "cB"], [block_col_base])
        node("Sub", [positions, block_col_base], [slot])
        node("Unsqueeze", [block_col, "axis1"], [block_col_col])
        node("Concat", ["row_id_col", block_col_col], [gather_index], axis=1)
        node("GatherND", [table, gather_index], [block_id])
        node("Mul", [block_id, "cB"], [block_base])
        node("Add", [block_base, slot], [physical])
        return block_col, block_id, physical

    current_block_col, _, current_full_physical = map_positions("current_full", "pos", "block_table_i64")
    _, current_window_block, current_window_physical = map_positions(
        "current_window", "pos", "block_table_windowed_i64"
    )
    _, _, previous_full_physical = map_positions("previous_full", "previous_pos", "block_table_i64")
    _, _, previous_window_physical = map_positions("previous_window", "previous_pos", "block_table_windowed_i64")

    # Position zero remains in the full cache. Its value records the first
    # window block id so continuation can prove that the ring stayed resident.
    node("Gather", ["block_table_i64", "c0"], ["first_full_block_per_row"], axis=1)
    node(
        "Gather",
        ["first_full_block_per_row", "row_id"],
        ["first_full_block"],
        axis=0,
    )
    node("Mul", ["first_full_block", "cB"], ["first_full_physical"])
    node(
        "Gather",
        ["block_table_windowed_i64", "c0"],
        ["first_window_block_per_row"],
        axis=1,
    )
    node(
        "Gather",
        ["first_window_block_per_row", "row_id"],
        ["first_window_block"],
        axis=0,
    )

    # Store exact integer-valued float encodings. Including the absolute
    # position makes a continuation that resets past_sequence_lengths diverge.
    node("Mul", ["input_ids", "c7"], ["key_token_term"])
    node("Mul", ["pos", "c3"], ["key_position_term"])
    node("Add", ["key_token_term", "key_position_term"], ["key_without_bias"])
    node("Add", ["key_without_bias", "c1"], ["key_encoding_i64"])
    node("Cast", ["key_encoding_i64"], ["key_encoding"], to=TensorProto.FLOAT)

    node("Mul", ["input_ids", "c5"], ["value_token_term"])
    node("Mul", ["pos", "c2"], ["value_position_term"])
    node(
        "Add",
        ["value_token_term", "value_position_term"],
        ["value_without_bias"],
    )
    node("Add", ["value_without_bias", "c2"], ["value_encoding_i64"])
    node(
        "Cast",
        ["value_encoding_i64"],
        ["window_value_encoding"],
        to=TensorProto.FLOAT,
    )
    node(
        "Cast",
        [current_window_block],
        ["window_owner_encoding"],
        to=TensorProto.FLOAT,
    )

    node(
        "Reshape",
        ["past_key_values.0.key", "flat"],
        ["past_full_key_flat"],
    )
    node(
        "Reshape",
        ["past_key_values.0.value", "flat"],
        ["past_full_value_flat"],
    )
    node(
        "Reshape",
        ["past_key_values.1.key", "flat"],
        ["past_window_key_flat"],
    )
    node(
        "Reshape",
        ["past_key_values.1.value", "flat"],
        ["past_window_value_flat"],
    )
    node(
        "Unsqueeze",
        [current_full_physical, "axis1"],
        ["current_full_scatter_index"],
    )
    node(
        "Unsqueeze",
        [current_window_physical, "axis1"],
        ["current_window_scatter_index"],
    )

    node(
        "ScatterND",
        ["past_full_key_flat", "current_full_scatter_index", "key_encoding"],
        ["present_full_key_flat"],
    )
    node(
        "ScatterND",
        [
            "past_full_value_flat",
            "current_full_scatter_index",
            "window_owner_encoding",
        ],
        ["present_full_value_flat"],
    )
    node(
        "ScatterND",
        ["past_window_key_flat", "current_window_scatter_index", "key_encoding"],
        ["present_window_key_flat"],
    )
    node(
        "ScatterND",
        [
            "past_window_value_flat",
            "current_window_scatter_index",
            "window_value_encoding",
        ],
        ["present_window_value_flat"],
    )

    node(
        "Reshape",
        ["present_full_key_flat", "full_cache_shape"],
        ["present.0.key"],
    )
    node(
        "Reshape",
        ["present_full_value_flat", "full_cache_shape"],
        ["present.0.value"],
    )
    node(
        "Reshape",
        ["present_window_key_flat", "window_cache_shape"],
        ["present.1.key"],
    )
    node(
        "Reshape",
        ["present_window_value_flat", "window_cache_shape"],
        ["present.1.value"],
    )

    # Logits consume values read through both cache layers, including the
    # previous live window position after the ring has wrapped.
    node(
        "Gather",
        ["present_full_key_flat", "first_full_physical"],
        ["read_full_first_key"],
        axis=0,
        name="read_full_first_key",
    )
    node(
        "Gather",
        ["present_full_key_flat", current_full_physical],
        ["read_full_current_key"],
        axis=0,
        name="read_full_current_key",
    )
    node(
        "Gather",
        ["present_full_key_flat", previous_full_physical],
        ["read_full_previous_key"],
        axis=0,
        name="read_full_previous_key",
    )
    node(
        "Gather",
        ["present_full_value_flat", "first_full_physical"],
        ["read_first_window_owner"],
        axis=0,
        name="read_first_window_owner",
    )
    node(
        "Gather",
        ["present_window_key_flat", previous_window_physical],
        ["read_window_previous_key"],
        axis=0,
        name="read_window_previous_key",
    )
    node(
        "Gather",
        ["present_window_key_flat", current_window_physical],
        ["read_window_current_key"],
        axis=0,
        name="read_window_current_key",
    )
    node(
        "Gather",
        ["present_window_value_flat", previous_window_physical],
        ["read_window_previous_value"],
        axis=0,
        name="read_window_previous_value",
    )
    node(
        "Gather",
        ["present_window_value_flat", current_window_physical],
        ["read_window_current_value"],
        axis=0,
        name="read_window_current_value",
    )

    # The current and prior-cycle columns must name the same physical block.
    node("Sub", [current_block_col, "cR"], ["prior_cycle_col_unclamped"])
    node("Max", ["prior_cycle_col_unclamped", "c0"], ["prior_cycle_col"])
    node("Unsqueeze", ["prior_cycle_col", "axis1"], ["prior_cycle_col_col"])
    node(
        "Concat",
        ["row_id_col", "prior_cycle_col_col"],
        ["prior_cycle_gather_index"],
        axis=1,
    )
    node(
        "GatherND",
        ["block_table_windowed_i64", "prior_cycle_gather_index"],
        ["prior_cycle_window_block"],
    )
    node(
        "GreaterOrEqual",
        [current_block_col, "cR"],
        ["has_prior_block_cycle"],
    )
    node(
        "Equal",
        [current_window_block, "prior_cycle_window_block"],
        ["window_block_repeats"],
    )
    node("Not", ["has_prior_block_cycle"], ["before_first_block_cycle"])
    node(
        "Or",
        ["before_first_block_cycle", "window_block_repeats"],
        ["repeated_window_block_valid"],
        name="guard_repeated_window_block",
    )

    node(
        "Equal",
        ["read_full_previous_key", "read_window_previous_key"],
        ["previous_cache_values_match"],
    )
    node(
        "Equal",
        ["read_full_current_key", "read_window_current_key"],
        ["current_cache_values_match"],
    )
    node(
        "Cast",
        ["first_window_block"],
        ["first_window_block_f"],
        to=TensorProto.FLOAT,
    )
    node(
        "Equal",
        ["read_first_window_owner", "first_window_block_f"],
        ["window_owner_stable"],
        name="guard_stable_window_owner",
    )
    node(
        "And",
        ["previous_cache_values_match", "current_cache_values_match"],
        ["cache_values_match"],
    )
    node(
        "And",
        ["cache_values_match", "window_owner_stable"],
        ["cache_and_owner_valid"],
    )
    node(
        "And",
        ["cache_and_owner_valid", "repeated_window_block_valid"],
        ["window_invariants_valid"],
    )

    score_terms = [
        "read_full_first_key",
        "read_full_current_key",
        "read_window_previous_key",
        "read_window_current_key",
        "read_window_previous_value",
        "read_window_current_value",
    ]
    score = score_terms[0]
    for index, term in enumerate(score_terms[1:], start=1):
        output = f"score_sum_{index}"
        node("Add", [score, term], [output])
        score = output
    node("Cast", [score], ["score_i64"], to=TensorProto.INT64)
    node("Div", ["score_i64", "c28"], ["score_div"])
    node("Mul", ["score_div", "c28"], ["score_floor"])
    node("Sub", ["score_i64", "score_floor"], ["score_mod"])
    node("Add", ["score_mod", "c2"], ["normal_next_token"])
    node(
        "Where",
        ["window_invariants_valid", "normal_next_token", "cInvariantFailure"],
        ["guarded_next_token"],
    )

    # EOS at absolute positions 2 and 13 creates two short turns while leaving
    # max_length headroom. The continuation spans positions 3..11; normal
    # outputs at 11 and 12 validate both columns of the two-block ring.
    node("Equal", ["pos", "c2"], ["is_first_turn_eos_position"])
    node("Equal", ["pos", "c13"], ["is_second_turn_eos_position"])
    node(
        "Or",
        ["is_first_turn_eos_position", "is_second_turn_eos_position"],
        ["is_eos_position"],
    )
    node(
        "Where",
        ["is_eos_position", "cEOS", "guarded_next_token"],
        ["next_token"],
    )

    node("Unsqueeze", ["next_token", "axis1"], ["next_token_col"])
    node("Equal", ["next_token_col", "vocab_range"], ["is_next_per_token"])
    node("Sub", ["boundaries", "c1"], ["last_token_index"])
    node(
        "Gather",
        ["is_next_per_token", "last_token_index"],
        ["is_next_per_request"],
        axis=0,
    )
    node(
        "Cast",
        ["is_next_per_request"],
        ["logits"],
        to=TensorProto.FLOAT16,
    )

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"]),
        helper.make_tensor_value_info(
            "cumulative_sequence_lengths",
            TensorProto.INT32,
            ["batch_plus_1"],
        ),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch"]),
        helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch", "max_blocks"]),
        helper.make_tensor_value_info(
            "block_table_windowed",
            TensorProto.INT32,
            ["batch", "max_blocks"],
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.key",
            TensorProto.FLOAT,
            full_cache_shape,
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.value",
            TensorProto.FLOAT,
            full_cache_shape,
        ),
        helper.make_tensor_value_info(
            "past_key_values.1.key",
            TensorProto.FLOAT,
            window_cache_shape,
        ),
        helper.make_tensor_value_info(
            "past_key_values.1.value",
            TensorProto.FLOAT,
            window_cache_shape,
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch_size", VOCAB_SIZE]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, full_cache_shape),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, full_cache_shape),
        helper.make_tensor_value_info("present.1.key", TensorProto.FLOAT, window_cache_shape),
        helper.make_tensor_value_info("present.1.value", TensorProto.FLOAT, window_cache_shape),
    ]
    return helper.make_graph(
        nodes,
        "synthetic_windowed_multiwrap_decoder",
        inputs,
        outputs,
        initializer=initializers,
    )


def create_decoder(output_dir):
    model = helper.make_model(
        _decoder_graph(),
        opset_imports=[helper.make_operatorsetid("", 17)],
        ir_version=9,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
    )
    metadata = model.metadata_props.add()
    metadata.key = "fixture"
    metadata.value = "engine-windowed-multiwrap-continuation"
    onnx.checker.check_model(model)
    onnx.save_model(model, os.path.join(output_dir, "decoder.onnx"))


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
                "num_hidden_layers": 2,
                "sliding_window": {
                    "window_size": WINDOW_SIZE,
                    "slide_key_value_cache": False,
                    "slide_inputs": False,
                    "layers": [1],
                },
                "inputs": {
                    "input_ids": "input_ids",
                    "block_table": "block_table",
                    "block_table_windowed": "block_table_windowed",
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
        "search": {
            "max_length": CONTEXT_LENGTH,
            "chunk_size": CHUNK_SIZE,
            "do_sample": False,
        },
        "engine": {
            "dynamic_batching": {
                "block_size": BLOCK_SIZE,
                "num_blocks": NUM_FULL_BLOCKS,
                "max_batch_size": MAX_BATCH_SIZE,
            },
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "engine",
            "synthetic-windowed-multiwrap",
        ),
    )
    args = parser.parse_args()
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    create_decoder(output_dir)
    create_config(output_dir)


if __name__ == "__main__":
    main()
