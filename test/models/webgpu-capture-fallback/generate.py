# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regenerate tiny decoders with real GQA supporting shared KV."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(control_flow=False, cpu_compute=False, fixed=False):
    def value(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    def constant(name, values, dtype=np.int64):
        return numpy_helper.from_array(np.array(values, dtype=dtype), name)

    table = np.full((16, 16), -10, dtype=np.float32)
    for token in range(16):
        table[token, (token + 1) % 16] = 10
    cache_size = 16 if fixed else "capacity"
    # All variants exercise sparse layer discovery and custom cache names.
    past = ("kv.3.k", "kv.3.v")
    present = ("out.3.k", "out.3.v")
    inputs = [
        value("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        value("attention_mask", TensorProto.INT64, ["batch", "total"]),
        *[value(name, TensorProto.FLOAT, ["batch", 1, cache_size, 64]) for name in past],
    ]
    outputs = [
        value("logits", TensorProto.FLOAT, ["batch", "sequence", 16]),
        *[value(name, TensorProto.FLOAT, ["batch", 1, cache_size, 64]) for name in present],
    ]
    constants = [
        numpy_helper.from_array(table, "table"),
        constant("last_axis", [-1]),
        constant("sequence_axis", [1]),
        constant("batch_axis", [0]),
        constant("tile", [1, 1, 64]),
        constant("one", 1, np.int32),
        constant("zero_float", 0, np.float32),
    ]
    nodes = [helper.make_node("Gather", ["table", "input_ids"], ["base_logits"])]
    value_ids = "input_ids"
    if cpu_compute:
        constants.append(constant("flat_shape", [-1]))
        nodes.extend(
            [
                helper.make_node("Reshape", ["input_ids", "flat_shape"], ["flat_ids"]),
                helper.make_node("Unique", ["flat_ids"], ["unique_ids", "indices", "inverse"], sorted=0),
                helper.make_node("Gather", ["unique_ids", "inverse"], ["restored_flat"]),
                helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
                helper.make_node("Reshape", ["restored_flat", "ids_shape"], ["restored_ids"]),
            ]
        )
        value_ids = "restored_ids"
    nodes.extend(
        [
            helper.make_node("Cast", [value_ids], ["float_ids"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["float_ids", "last_axis"], ["ids_3d"]),
            helper.make_node("Tile", ["ids_3d", "tile"], ["values"]),
            helper.make_node("Mul", ["values", "zero_float"], ["zeros"]),
            helper.make_node("Cast", ["attention_mask"], ["mask_int32"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["mask_int32", "sequence_axis"], ["lengths"], keepdims=0),
            helper.make_node("Sub", ["lengths", "one"], ["last_positions"]),
            helper.make_node("Cast", ["last_positions"], ["seqlens_k"], to=TensorProto.INT32),
            helper.make_node("ReduceMax", ["lengths", "batch_axis"], ["total_length"], keepdims=0),
            helper.make_node("Cast", ["total_length"], ["total_sequence_length"], to=TensorProto.INT32),
            helper.make_node(
                "GroupQueryAttention",
                ["zeros", "zeros", "values", *past, "seqlens_k", "total_sequence_length"],
                ["attention", *present],
                domain="com.microsoft",
                num_heads=1,
                kv_num_heads=1,
            ),
            helper.make_node("ReduceMean", ["attention", "last_axis"], ["attention_bias"], keepdims=1),
            helper.make_node("Add", ["base_logits", "attention_bias"], ["raw_logits"]),
        ]
    )
    if control_flow:
        constants.extend([constant("first_index", [0, 0]), constant("zero", 0)])
        nodes.extend(
            [
                helper.make_node("GatherND", ["input_ids", "first_index"], ["first_token"]),
                helper.make_node("Greater", ["first_token", "zero"], ["condition"]),
            ]
        )
        branches = {}
        for name, op in (("then_branch", "Identity"), ("else_branch", "Neg")):
            branches[name] = helper.make_graph(
                [helper.make_node(op, ["raw_logits"], ["branch_logits"])],
                name,
                [],
                [value("branch_logits", TensorProto.FLOAT, ["batch", "sequence", 16])],
            )
        nodes.append(helper.make_node("If", ["condition"], ["logits"], **branches))
    else:
        nodes.append(helper.make_node("Identity", ["raw_logits"], ["logits"]))
    graph = helper.make_graph(nodes, "shared_capture_fallback", inputs, outputs, constants)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=9
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    for name, options in (
        ("shared.onnx", {}),
        ("control-flow.onnx", {"control_flow": True}),
        ("cpu-compute.onnx", {"cpu_compute": True}),
        ("fixed-control-flow.onnx", {"control_flow": True, "fixed": True}),
    ):
        onnx.save(build(**options), Path(__file__).parent / name)
