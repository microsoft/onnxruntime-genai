# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Per-channel storage, packing and default-quantizer isolation checks."""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, QuantFormat

_MODULE_PATH = Path(__file__).parents[3] / "src/python/py/models/quantization/per_channel.py"
_spec = importlib.util.spec_from_file_location("per_channel", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _gather_model(table):
    vocab, hidden = table.shape
    dtype = onnx.TensorProto.FLOAT if table.dtype == np.float32 else onnx.TensorProto.FLOAT16
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Gather", ["E", "I"], ["Y"], axis=0)],
            "test",
            [onnx.helper.make_tensor_value_info("I", onnx.TensorProto.INT64, [2])],
            [onnx.helper.make_tensor_value_info("Y", dtype, [2, hidden])],
            [onnx.numpy_helper.from_array(table, "E")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 21)],
        ir_version=10,
    )


def _model(weights):
    k, n = weights.shape
    dtype = onnx.TensorProto.FLOAT if weights.dtype == np.float32 else onnx.TensorProto.FLOAT16
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["A", "W"], ["Y"])],
            "test",
            [onnx.helper.make_tensor_value_info("A", dtype, [2, k])],
            [onnx.helper.make_tensor_value_info("Y", dtype, [2, n])],
            [onnx.numpy_helper.from_array(weights, "W")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 21)],
        ir_version=10,
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("k", [17, 96, 1024, 3584])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_single_channel_scale_and_packing(bits, k, dtype):
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(k, 5)).astype(dtype)
    weights[:, 0] = 0
    kwargs = dict(
        bits=bits,
        block_size=32,
        is_symmetric=True,
        accuracy_level=1,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        algo_config=None,
    )
    quantizer = _module.make_per_channel_quantizer(model=_model(weights), **kwargs)
    quantizer.process()
    model = quantizer.model.model
    tensors = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    packed, scales = tensors[f"W_Q{bits}"], tensors["W_scales"]
    block = max(16, 1 << (k - 1).bit_length())
    kpack = 8 // bits
    qmax = (1 << (bits - 1)) - 1
    zero = 1 << (bits - 1)
    assert packed.shape == (5, 1, block // kpack)
    assert scales.shape == (5, 1)
    assert scales.dtype == dtype
    node = model.graph.node[0]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    assert attrs["K"] == k and attrs["block_size"] == block and attrs["bits"] == bits
    assert len(node.input) == 3  # Implicit zero point.
    if bits == 4:
        values = np.stack((packed[:, 0] & 15, packed[:, 0] >> 4), axis=-1).reshape(5, -1).astype(np.float32)
    else:
        values = packed[:, 0].astype(np.float32)
    scale = np.max(np.abs(weights.astype(np.float32)), axis=0) / qmax
    scale = np.where(scale == 0, 1, scale).astype(dtype)
    np.testing.assert_array_equal(scales[:, 0], scale)
    expected = np.clip(np.rint(weights.astype(np.float32) / scale), -qmax, qmax) * scale
    np.testing.assert_array_equal((values[:, :k].T - zero) * scale, expected)
    assert np.all(values[:, k:] == zero)
    if k <= 256:
        inputs = rng.normal(size=(2, k)).astype(dtype)
        result = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"A": inputs}
        )[0]
        np.testing.assert_allclose(result, inputs.astype(np.float32) @ expected, rtol=0.01, atol=0.03)
    # The adapter must not mutate ORT's default quantizer globally.
    default = MatMulNBitsQuantizer(model=_model(weights), **kwargs)
    default.process()
    assert next(t for t in default.model.model.graph.initializer if t.name == "W_scales").dims[1] == (k + 31) // 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 2, "is_symmetric": True},
        {"bits": 4, "is_symmetric": False},
        {"bits": 8, "is_symmetric": False},
        {"bits": 4, "is_symmetric": True, "quant_format": QuantFormat.QDQ},
        {"bits": 8, "is_symmetric": True, "quant_format": QuantFormat.QDQ},
    ],
)
def test_unsupported_modes(kwargs):
    options = dict(model=_model(np.ones((32, 4), dtype=np.float32)), quant_format=QuantFormat.QOperator)
    options.update(kwargs)
    with pytest.raises(ValueError):
        _module.make_per_channel_quantizer(**options)


def test_matmul_does_not_change_following_gather_block_size():
    weights = np.ones((96, 5), dtype=np.float32)
    model = _model(weights)
    model.graph.node.append(onnx.helper.make_node("Gather", ["E", "I"], ["Z"], axis=0))
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(np.ones((8, 96), dtype=np.float32), "E"),
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "I"),
        ]
    )
    quantizer = _module.make_per_channel_quantizer(
        model=model,
        bits=4,
        block_size=32,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
    )
    quantizer.process()
    gather = next(n for n in quantizer.model.model.graph.node if n.op_type == "GatherBlockQuantized")
    assert next(a.i for a in gather.attribute if a.name == "block_size") == 32
    assert quantizer.algo_config.block_size == 32


@pytest.mark.parametrize("hidden", [17, 96, 1024, 3584])
def test_gather_per_channel_one_scale_per_row(hidden):
    # Mirrors how base.py sizes the shared quantizer block_size for per-channel mode:
    # MatMul recomputes its own per-node block internally, but Gather reads this value as-is.
    block = _module.per_channel_block_size(hidden)
    rng = np.random.default_rng(7)
    table = rng.normal(size=(6, hidden)).astype(np.float32)
    quantizer = _module.make_per_channel_quantizer(
        model=_gather_model(table),
        bits=4,
        block_size=block,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("Gather",),
        algo_config=None,
    )
    quantizer.process()
    model = quantizer.model.model
    gather = next(n for n in model.graph.node if n.op_type == "GatherBlockQuantized")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gather.attribute}
    assert attrs["block_size"] == block
    assert len(gather.input) == 3  # Symmetric: no zero-point input.

    tensors = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    scales = tensors["E_scales"]
    assert scales.shape == (6, 1)  # One scale per vocab row: a single block spans the whole hidden dim.

    # ORT's own (unmodified) symmetric Gather quantization: scale = abs_max / -8.
    max_val, min_val = table.max(axis=1), table.min(axis=1)
    abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)
    expected_scale = abs_max / -8.0
    np.testing.assert_allclose(scales[:, 0], expected_scale, rtol=1e-6)
