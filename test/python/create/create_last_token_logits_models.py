# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate tiny CPU fixtures for the native Logits component and Generator API."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def create_model(path, *, pruned, indices_type=None):
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch", "sequence"])]
    if indices_type is not None:
        inputs.append(helper.make_tensor_value_info("last_token_indices", indices_type, ["batch", 1, 1]))
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["tokens"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["tokens", "axes"], ["hidden"]),
    ]
    selected = "hidden"
    if pruned:
        nodes.append(helper.make_node("GatherElements", ["hidden", "last_token_indices"], ["selected"], axis=1))
        selected = "selected"
    nodes.append(helper.make_node("Concat", [selected] * 4, ["logits"], axis=2))
    graph = helper.make_graph(
        nodes,
        "last_token_logits",
        inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1 if pruned else "sequence", 4])],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
        producer_name="onnxruntime-genai",
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def create_generator_model(path, *, pruned):
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch", 1, "past", 1]),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, ["batch", 1, "past", 1]),
    ]
    if pruned:
        inputs.append(helper.make_tensor_value_info("last_token_indices", TensorProto.INT32, ["batch", 1, 1]))
    constants = [
        helper.make_tensor("cache_axes", TensorProto.INT64, [2], [1, 3]),
        helper.make_tensor("reduce_axes", TensorProto.INT64, [3], [1, 2, 3]),
        helper.make_tensor("sum_axis", TensorProto.INT64, [1], [1]),
        helper.make_tensor("sequence_axis", TensorProto.INT64, [], [1]),
        helper.make_tensor("logits_axis", TensorProto.INT64, [1], [2]),
        helper.make_tensor("zero", TensorProto.FLOAT, [], [0]),
    ]
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["tokens"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["tokens", "cache_axes"], ["new_keys"]),
        helper.make_node("Concat", ["past_key_values.0.key", "new_keys"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "new_keys"], ["present.0.value"], axis=2),
        helper.make_node("ReduceSum", ["past_key_values.0.key", "reduce_axes"], ["past_sum"], keepdims=0),
        helper.make_node("Unsqueeze", ["past_sum", "sum_axis"], ["past_sum_column"]),
        helper.make_node("CumSum", ["tokens", "sequence_axis"], ["token_sums"]),
        helper.make_node("Add", ["past_sum_column", "token_sums"], ["sum"]),
        helper.make_node("Neg", ["sum"], ["negative"]),
        helper.make_node("Mul", ["sum", "zero"], ["zeros"]),
        helper.make_node("Unsqueeze", ["sum", "logits_axis"], ["positive_logit"]),
        helper.make_node("Unsqueeze", ["negative", "logits_axis"], ["negative_logit"]),
        helper.make_node("Unsqueeze", ["zeros", "logits_axis"], ["zero_logit"]),
        helper.make_node(
            "Concat",
            ["negative_logit", "positive_logit", "zero_logit"],
            ["all_logits" if pruned else "logits"],
            axis=2,
        ),
    ]
    if pruned:
        constants.append(helper.make_tensor("indices_shape", TensorProto.INT64, [3], [1, 1, 3]))
        nodes.extend(
            [
                helper.make_node("Expand", ["last_token_indices", "indices_shape"], ["expanded_indices"]),
                helper.make_node("GatherElements", ["all_logits", "expanded_indices"], ["logits"], axis=1),
            ]
        )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "last_token_generation",
            inputs,
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1 if pruned else "sequence", 3]),
                helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch", 1, "total", 1]),
                helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, ["batch", 1, "total", 1]),
            ],
            initializer=constants,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
        producer_name="onnxruntime-genai",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models/engine/last-token-logits",
    )
    parser.add_argument(
        "--generation_output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models/last-token-generation",
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    create_model(output / "pruned.onnx", pruned=True, indices_type=TensorProto.INT32)
    create_model(output / "invalid-index-type.onnx", pruned=True, indices_type=TensorProto.INT64)
    create_model(output / "unpruned-with-indices.onnx", pruned=False, indices_type=TensorProto.INT32)
    create_model(output / "full.onnx", pruned=False)
    args.generation_output_dir.mkdir(parents=True, exist_ok=True)
    create_generator_model(args.generation_output_dir / "full.onnx", pruned=False)
    create_generator_model(args.generation_output_dir / "pruned.onnx", pruned=True)


if __name__ == "__main__":
    main()
