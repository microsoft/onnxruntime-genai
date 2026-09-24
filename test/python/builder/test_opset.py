# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Schema and numerical regressions for the shared ONNX opset-24 default."""

import copy

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from transformers import LlamaConfig

from _builder_test_utils import load_builder_module

base = load_builder_module("base")
BlockDrafterBuilder = load_builder_module("block_drafter").BlockDrafterBuilder

# Standard-domain ops emitted by the builders whose schema changes after opset 22.
# QuantizeLinear is also included because quantization passes may insert it.
# Attention and RotaryEmbedding use com.microsoft:1, not their new ONNX schemas.
UPDATED_OPS = (
    "Cast",
    "Constant",
    "ConstantOfShape",
    "DequantizeLinear",
    "Identity",
    "If",
    "QuantizeLinear",
    "Reshape",
    "Shape",
    "Squeeze",
    "TopK",
    "Transpose",
    "Unsqueeze",
)


def _builder(tmp_path):
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32,
        architectures=["LlamaForCausalLM"],
    )
    return base.Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path), {})


def test_all_graph_builders_default_to_opset_24(tmp_path):
    model = _builder(tmp_path)
    drafter = BlockDrafterBuilder()
    drafter.make_graph("test_drafter", "/constants")
    assert base.DEFAULT_OPSET == 24
    for built in (model.model, drafter.model):
        assert dict(built.graph.opset_imports) == {"": 24, "com.microsoft": 1}
        assert built.ir_version == 10


def test_tensor_scatter_uses_default_without_changing_imports(tmp_path):
    builder = _builder(tmp_path)
    for name, dtype, shape in (
        ("past", ir.DataType.FLOAT, [1, 1, 8, 2]),
        ("updates", ir.DataType.FLOAT, [1, 1, 2, 2]),
        ("indices", ir.DataType.INT64, [1]),
    ):
        builder.make_value(name, dtype, shape)
        builder.graph.inputs.append(builder.values[name])
    imports = dict(builder.graph.opset_imports)
    output = builder.make_tensor_scatter("/cache", "past", "updates", "indices", ir.DataType.FLOAT, [1, 1, 8, 2])
    builder.graph.outputs.append(builder.values[output])
    assert dict(builder.graph.opset_imports) == imports
    model = ir.serde.serialize_model(builder.model)
    past = np.zeros((1, 1, 8, 2), dtype=np.float32)
    updates = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    expected = past.copy()
    expected[:, :, 3:5] = updates
    actual = _run(model, {"past": past, "updates": updates, "indices": np.array([3], dtype=np.int64)})
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize("old_version", [21, 22])
@pytest.mark.parametrize("op", UPDATED_OPS)
def test_updated_schemas_preserve_existing_signatures(op, old_version):
    old = onnx.defs.get_schema(op, old_version, "")
    new = onnx.defs.get_schema(op, base.DEFAULT_OPSET, "")
    assert (old.min_input, old.max_input, old.min_output, old.max_output) == (
        new.min_input,
        new.max_input,
        new.min_output,
        new.max_output,
    )
    old_types = {constraint.type_param_str: set(constraint.allowed_type_strs) for constraint in old.type_constraints}
    new_types = {constraint.type_param_str: set(constraint.allowed_type_strs) for constraint in new.type_constraints}
    for before, after in zip((*old.inputs, *old.outputs), (*new.inputs, *new.outputs), strict=True):
        assert (before.name, before.option, before.is_homogeneous, before.min_arity) == (
            after.name,
            after.option,
            after.is_homogeneous,
            after.min_arity,
        )
        # DequantizeLinear's output changes from T2 to T3, but still supports all old types.
        assert old_types.get(before.type_str, {before.type_str}) <= new_types.get(after.type_str, {after.type_str})
    for name, before in old.attributes.items():
        after = new.attributes[name]
        assert (before.type, before.required, before.default_value) == (after.type, after.required, after.default_value)
    assert all(not attr.required for name, attr in new.attributes.items() if name not in old.attributes)


def _run(model, inputs):
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, inputs)


def _compare_with_previous_opsets(model, inputs):
    actual = _run(model, inputs)
    for version in (21, 22):
        previous = copy.deepcopy(model)
        previous.opset_import[0].version = version
        expected = _run(previous, inputs)
        for value, reference in zip(actual, expected, strict=True):
            assert value.dtype == reference.dtype
            np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("condition", [True, False])
def test_shape_cast_topk_and_control_flow_match_previous_opsets(condition):
    def constant(name, values):
        return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(values, dtype=np.int64)))

    nodes = [
        constant("flat_shape", [-1]),
        constant("axes", [0]),
        constant("k", [1]),
        helper.make_node("Cast", ["x"], ["half"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["half"], ["float"], to=TensorProto.FLOAT),
        helper.make_node("Shape", ["float"], ["shape"]),
        helper.make_node("Reshape", ["float", "flat_shape"], ["flat"]),
        helper.make_node("Unsqueeze", ["flat", "axes"], ["expanded"]),
        helper.make_node("Squeeze", ["expanded", "axes"], ["squeezed"]),
        helper.make_node("Reshape", ["squeezed", "shape"], ["restored"]),
        helper.make_node("Transpose", ["restored"], ["transposed"], perm=[1, 0]),
        helper.make_node("Shape", ["transposed"], ["transposed_shape"]),
        helper.make_node(
            "ConstantOfShape", ["transposed_shape"], ["offset"],
            value=numpy_helper.from_array(np.array([0.25], dtype=np.float32)),
        ),
        helper.make_node("Add", ["transposed", "offset"], ["shifted"]),
        helper.make_node("TopK", ["shifted", "k"], ["values", "indices"], axis=-1, largest=1),
    ]
    branches = {}
    for name, op in (("then_branch", "Identity"), ("else_branch", "Neg")):
        branches[name] = helper.make_graph(
            [helper.make_node(op, ["values"], ["result"])],
            name,
            [],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, [3, 1])],
        )
    nodes.append(helper.make_node("If", ["condition"], ["output"], **branches))
    graph = helper.make_graph(
        nodes,
        "opset_regression",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 1]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", base.DEFAULT_OPSET)], ir_version=10)
    _compare_with_previous_opsets(
        model,
        {
            "x": np.array([[1.1, -2.2, 3.3], [4.4, 5.5, -6.6]], dtype=np.float32),
            "condition": np.array(condition),
        },
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("blocked", [False, True])
def test_qdq_default_dtypes_and_values_match_previous_opsets(dtype, blocked):
    tensor_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    scales = np.array([[0.25, 0.5], [0.5, 0.25]] if blocked else 0.25, dtype=dtype)
    zeros = np.zeros(scales.shape, dtype=np.int8)
    attrs = {"axis": -1, "block_size": 2} if blocked else {}
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["quantized"], **attrs),
        helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["output"], **attrs),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_regression",
        [helper.make_tensor_value_info("x", tensor_type, [2, 4])],
        [
            helper.make_tensor_value_info("output", tensor_type, [2, 4]),
            helper.make_tensor_value_info("quantized", TensorProto.INT8, [2, 4]),
        ],
        [numpy_helper.from_array(scales, "scale"), numpy_helper.from_array(zeros, "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", base.DEFAULT_OPSET)], ir_version=10)
    _compare_with_previous_opsets(
        model,
        {"x": np.array([[-100, -1.125, 0.375, 100], [1.25, -0.125, 0, 2.75]], dtype=dtype)},
    )
