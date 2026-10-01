# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def value(name, data_type, shape):
    return helper.make_tensor_value_info(name, data_type, shape)


inputs = [
    value("aux_hidden_states", TensorProto.FLOAT, ["context_rows", 1]),
    value("input_ids", TensorProto.INT64, ["block_rows"]),
    value("q_row_map", TensorProto.INT32, ["packed_rows"]),
    value("qkv_row_map", TensorProto.INT32, ["packed_rows"]),
    value("block_row_index", TensorProto.INT32, ["block_rows"]),
    value("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_1"]),
    value("past_sequence_lengths", TensorProto.INT32, ["batch"]),
    value("block_table", TensorProto.INT32, ["batch", "max_blocks"]),
    value("attention_metadata", TensorProto.INT32, [3]),
    value("past_key_values.0.key", TensorProto.FLOAT, ["num_blocks", 4, 1, 1]),
    value("past_key_values.0.value", TensorProto.FLOAT, ["num_blocks", 4, 1, 1]),
]

outputs = [
    value("draft_candidate_ids", TensorProto.INT32, ["batch", 4, 2]),
    value("draft_scores", TensorProto.FLOAT, ["batch", 4, 2, 2]),
    value("present.0.key", TensorProto.FLOAT, ["num_blocks", 4, 1, 1]),
    value("present.0.value", TensorProto.FLOAT, ["num_blocks", 4, 1, 1]),
]

initializers = [
    helper.make_tensor("axis0", TensorProto.INT64, [1], [0]),
    helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
    helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("start0", TensorProto.INT64, [1], [0]),
    helper.make_tensor("start1", TensorProto.INT64, [1], [1]),
    helper.make_tensor("end_minus1", TensorProto.INT64, [1], [-1]),
    helper.make_tensor("end_all", TensorProto.INT64, [1], [9223372036854775807]),
    helper.make_tensor("candidate_tail", TensorProto.INT64, [2], [4, 2]),
    helper.make_tensor("score_tail", TensorProto.INT64, [3], [4, 2, 2]),
    helper.make_tensor("one", TensorProto.INT64, [1], [1]),
    helper.make_tensor("zero_score", TensorProto.FLOAT, [], [0.0]),
]

nodes = [
    helper.make_node("Shape", ["past_sequence_lengths"], ["past_shape"]),
    helper.make_node("Gather", ["past_shape", "start0"], ["batch"], axis=0),
    helper.make_node("Concat", ["batch", "candidate_tail"], ["candidate_shape"], axis=0),
    helper.make_node("Concat", ["batch", "score_tail"], ["score_shape"], axis=0),
    helper.make_node("ReduceSum", ["block_table", "axis1"], ["block_sum"], keepdims=0),
    helper.make_node("Unsqueeze", ["block_sum", "axis1"], ["block_sum_col"]),
    helper.make_node("Unsqueeze", ["past_sequence_lengths", "axis1"], ["past_col"]),
    helper.make_node("Slice", ["cumulative_sequence_lengths", "start0", "end_minus1", "axis0"], ["row_begin"]),
    helper.make_node("Slice", ["cumulative_sequence_lengths", "start1", "end_all", "axis0"], ["row_end"]),
    helper.make_node("Sub", ["row_end", "row_begin"], ["row_length"]),
    helper.make_node("Unsqueeze", ["row_length", "axis1"], ["row_length_col"]),
    helper.make_node("ReduceSum", ["q_row_map", "axis0"], ["q_checksum"], keepdims=0),
    helper.make_node("Concat", ["batch", "one"], ["batch_column_shape"], axis=0),
    helper.make_node("Expand", ["q_checksum", "batch_column_shape"], ["q_checksum_col"]),
    helper.make_node(
        "Concat",
        ["block_sum_col", "past_col", "row_length_col", "q_checksum_col"],
        ["draft_values"],
        axis=1,
    ),
    helper.make_node("Unsqueeze", ["draft_values", "axis2"], ["draft_values_col"]),
    helper.make_node("Expand", ["draft_values_col", "candidate_shape"], ["draft_candidate_ids"]),
    helper.make_node("Expand", ["zero_score", "score_shape"], ["draft_scores"]),
    helper.make_node("Identity", ["past_key_values.0.key"], ["present.0.key"]),
    helper.make_node("Identity", ["past_key_values.0.value"], ["present.0.value"]),
]

graph = helper.make_graph(nodes, "synthetic-dspark", inputs, outputs, initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("dspark.onnx"))

# A packed-QKV drafter has no q_row_map input; its checksum column reports qkv_row_map instead.
packed_qkv = onnx.ModelProto()
packed_qkv.CopyFrom(model)
for index, graph_input in enumerate(packed_qkv.graph.input):
    if graph_input.name == "q_row_map":
        del packed_qkv.graph.input[index]
        break
for node in packed_qkv.graph.node:
    if "q_checksum" in node.output:
        node.input[0] = "qkv_row_map"
onnx.checker.check_model(packed_qkv)
onnx.save(packed_qkv, Path(__file__).with_name("dspark_packed_qkv.onnx"))

# The windowed variant uses DFlash2's anchor row plus three draft rows.
windowed = onnx.ModelProto()
windowed.CopyFrom(model)
for initializer in windowed.graph.initializer:
    if initializer.name == "candidate_tail":
        initializer.CopyFrom(helper.make_tensor("candidate_tail", TensorProto.INT64, [2], [3, 2]))
    elif initializer.name == "score_tail":
        initializer.CopyFrom(helper.make_tensor("score_tail", TensorProto.INT64, [3], [3, 2, 2]))
for output in windowed.graph.output:
    if output.name in ("draft_candidate_ids", "draft_scores"):
        output.type.tensor_type.shape.dim[1].dim_value = 3
windowed.graph.initializer.extend(
    [
        helper.make_tensor("zero_i32", TensorProto.INT32, [], [0]),
        helper.make_tensor("one_i32", TensorProto.INT32, [], [1]),
        helper.make_tensor("four_i32", TensorProto.INT32, [], [4]),
        helper.make_tensor("five_i32", TensorProto.INT32, [], [5]),
        helper.make_tensor("one_float", TensorProto.FLOAT, [], [1.0]),
    ]
)
for index, node in enumerate(windowed.graph.node):
    if "draft_values_col" in node.output:
        node.input[0] = "draft_values_with_cache"
        cache_nodes = [
            helper.make_node("Cast", ["batch"], ["batch_i32"], to=TensorProto.INT32),
            helper.make_node("Range", ["zero_i32", "batch_i32", "one_i32"], ["batch_rows"]),
            helper.make_node("Sub", ["row_length", "five_i32"], ["last_context_offset"]),
            helper.make_node("Add", ["past_sequence_lengths", "last_context_offset"], ["last_position"]),
            helper.make_node("Div", ["last_position", "four_i32"], ["last_column"]),
            helper.make_node("Mod", ["last_position", "four_i32"], ["last_slot"]),
            helper.make_node("Sub", ["past_sequence_lengths", "one_i32"], ["previous_unclamped"]),
            helper.make_node("Max", ["previous_unclamped", "zero_i32"], ["previous_position"]),
            helper.make_node("Div", ["previous_position", "four_i32"], ["previous_column"]),
            helper.make_node("Mod", ["previous_position", "four_i32"], ["previous_slot"]),
            helper.make_node("Unsqueeze", ["batch_rows", "axis1"], ["batch_rows_col"]),
            helper.make_node("Unsqueeze", ["last_column", "axis1"], ["last_column_col"]),
            helper.make_node("Unsqueeze", ["previous_column", "axis1"], ["previous_column_col"]),
            helper.make_node("Concat", ["batch_rows_col", "previous_column_col"], ["read_table_indices"], axis=1),
            helper.make_node("Cast", ["read_table_indices"], ["read_table_indices_i64"], to=TensorProto.INT64),
            helper.make_node("GatherND", ["block_table", "read_table_indices_i64"], ["previous_block"]),
            helper.make_node("Unsqueeze", ["previous_block", "axis1"], ["previous_block_col"]),
            helper.make_node("Unsqueeze", ["previous_slot", "axis1"], ["previous_slot_col"]),
            helper.make_node("Concat", ["batch_rows_col", "last_column_col"], ["write_table_indices"], axis=1),
            helper.make_node("Cast", ["write_table_indices"], ["write_table_indices_i64"], to=TensorProto.INT64),
            helper.make_node("GatherND", ["block_table", "write_table_indices_i64"], ["last_block"]),
            helper.make_node("Unsqueeze", ["last_block", "axis1"], ["last_block_col"]),
            helper.make_node("Unsqueeze", ["last_slot", "axis1"], ["last_slot_col"]),
            helper.make_node("Sub", ["past_col", "past_col"], ["zero_col"]),
            helper.make_node(
                "Concat",
                ["previous_block_col", "previous_slot_col", "zero_col", "zero_col"],
                ["read_cache_indices"],
                axis=1,
            ),
            helper.make_node("Cast", ["read_cache_indices"], ["read_cache_indices_i64"], to=TensorProto.INT64),
            helper.make_node("GatherND", ["past_key_values.0.key", "read_cache_indices_i64"], ["cached_key"]),
            helper.make_node(
                "Concat",
                ["last_block_col", "last_slot_col", "zero_col", "zero_col"],
                ["write_cache_indices"],
                axis=1,
            ),
            helper.make_node("Cast", ["write_cache_indices"], ["write_cache_indices_i64"], to=TensorProto.INT64),
            helper.make_node("Cast", ["cached_key"], ["cached_key_int"], to=TensorProto.INT32),
            helper.make_node("Unsqueeze", ["cached_key_int", "axis1"], ["cached_key_col"]),
            helper.make_node(
                "Concat", ["cached_key_col", "past_col", "row_length_col"], ["draft_values_with_cache"], axis=1
            ),
            helper.make_node("Cast", ["last_position"], ["last_position_float"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["last_position_float", "one_float"], ["cache_updates"]),
            helper.make_node(
                "ScatterND",
                ["past_key_values.0.key", "write_cache_indices_i64", "cache_updates"],
                ["present.0.key"],
            ),
        ]
        for cache_node in reversed(cache_nodes):
            windowed.graph.node.insert(index, cache_node)
        break
for index, node in enumerate(windowed.graph.node):
    if "present.0.key" in node.output and node.op_type == "Identity":
        del windowed.graph.node[index]
        break
onnx.checker.check_model(windowed)
onnx.save(windowed, Path(__file__).with_name("dflash2.onnx"))
