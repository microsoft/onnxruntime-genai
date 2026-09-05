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
