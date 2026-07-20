from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_base_module():
    sys.modules.setdefault("models", types.ModuleType("models"))
    builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
    builders_package.__path__ = [str(BUILDERS_DIR)]

    spec = importlib.util.spec_from_file_location("models.builders.base", BUILDERS_DIR / "base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.builders.base"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    # `builder.py` imports the concrete model classes via `from builders import (...)`.
    # Provide a stub `builders` module so we can import the lightweight precision helpers
    # (`set_onnx_dtype` / `set_io_dtype`) without pulling in every model builder.
    builders_stub = types.ModuleType("builders")

    def __getattr__(name):  # PEP 562: satisfies `from builders import <ModelClass>`
        return type(name, (), {})

    builders_stub.__getattr__ = __getattr__
    sys.modules["builders"] = builders_stub

    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_module = _load_base_module()
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model


# ---------------------------------------------------------------------------
# int8 precision maps onnx_dtype to INT8/UINT8 (like int4 -> INT4/UINT4).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "is_symmetric, expected",
    [
        (True, ir.DataType.INT8),
        (False, ir.DataType.UINT8),
    ],
)
def test_int8_onnx_dtype_is_int8(is_symmetric, expected):
    assert builder_module.set_onnx_dtype("int8", {"is_symmetric": is_symmetric}) == expected


@pytest.mark.parametrize(
    "is_symmetric, expected",
    [
        (True, ir.DataType.INT4),
        (False, ir.DataType.UINT4),
    ],
)
def test_int4_onnx_dtype_is_still_int4(is_symmetric, expected):
    assert builder_module.set_onnx_dtype("int4", {"is_symmetric": is_symmetric}) == expected


@pytest.mark.parametrize(
    "execution_provider, expected",
    [
        ("cpu", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16),
        ("webgpu", ir.DataType.FLOAT16),
    ],
)
def test_int8_io_dtype_is_not_forced_to_fp32(execution_provider, expected):
    # int8 must not assume FP32 I/O: GPU/WebGPU use FP16, only CPU uses FP32.
    assert builder_module.set_io_dtype("int8", execution_provider, {}) == expected


# ---------------------------------------------------------------------------
# int8's INT8/UINT8 onnx_dtype routes through the MatMulNBits builders, which
# fall back to a float MatMul when the source model is not already quantized.
# ---------------------------------------------------------------------------


def _make_bare_model(onnx_dtype, quant_attrs=None):
    model = Model.__new__(Model)
    model.onnx_dtype = onnx_dtype
    model.quant_attrs = quant_attrs if quant_attrs is not None else {"use_qdq": False}
    return model


@pytest.mark.parametrize("onnx_dtype", [ir.DataType.INT8, ir.DataType.UINT8])
def test_make_matmul_op_int8_falls_back_to_float_when_not_quantized(monkeypatch, onnx_dtype):
    model = _make_bare_model(onnx_dtype)
    sentinel = object()
    monkeypatch.setattr(model, "make_matmul_float", lambda *a, **k: sentinel)

    assert model.make_matmul_op(object(), "/lm_head/MatMul", "root") is sentinel


@pytest.mark.parametrize("onnx_dtype", [ir.DataType.INT8, ir.DataType.UINT8])
def test_make_packed_matmul_int8_falls_back_to_float_when_not_quantized(monkeypatch, onnx_dtype):
    model = _make_bare_model(onnx_dtype)
    sentinel = object()
    monkeypatch.setattr(model, "make_packed_matmul_float", lambda *a, **k: sentinel)

    assert model.make_packed_matmul(object(), object(), object(), "/attn/qkv/MatMul", "root") is sentinel


# ---------------------------------------------------------------------------
# `to_nbits` forwards the requested weight bit width to `MatMulNBitsQuantizer`.
# ---------------------------------------------------------------------------


def _make_quant_model(bits):
    model = Model.__new__(Model)
    model.model = object()
    model.quant_attrs = {
        "bits": bits,
        "qdq_block_size": 32,
        "is_symmetric": True,
        "accuracy_level": 4,
        "nodes_to_exclude": [],
        "use_qdq": False,
        "op_types_to_quantize": ("MatMul",),
        "algo_config": None,
    }
    return model


class _FakeQuantizer:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.model = types.SimpleNamespace(model="quantized-proto")

    def process(self):
        pass


@pytest.mark.parametrize("bits", [4, 8])
def test_to_nbits_forwards_requested_bits(monkeypatch, bits):
    _FakeQuantizer.captured = None
    monkeypatch.setattr(base_module, "MatMulNBitsQuantizer", _FakeQuantizer)
    monkeypatch.setattr(base_module.ir, "to_proto", lambda m: m)
    monkeypatch.setattr(base_module.ir, "from_proto", lambda p: p)

    model = _make_quant_model(bits)
    result = model.to_nbits()

    assert _FakeQuantizer.captured is not None
    assert _FakeQuantizer.captured["bits"] == bits
    assert result == "quantized-proto"


# ---------------------------------------------------------------------------
# int8 rejects the unsupported QDQ format (8-bit MatMulNBits is QOperator-only).
# ---------------------------------------------------------------------------


def test_int8_with_qdq_is_rejected():
    with pytest.raises(NotImplementedError, match="QDQ"):
        builder_module.check_extra_options({"use_qdq": "true"}, "int8", "cpu")


def test_int4_with_qdq_is_allowed():
    # QDQ is only rejected for int8; int4 still supports it.
    builder_module.check_extra_options({"use_qdq": "true"}, "int4", "cpu")


# ---------------------------------------------------------------------------
# Deprecated `int4_*` extra_option names still work as soft-deprecated aliases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_name, new_name",
    [
        ("int4_accuracy_level", "accuracy_level"),
        ("int4_block_size", "block_size"),
        ("int4_is_symmetric", "is_symmetric"),
        ("int4_op_types_to_quantize", "op_types_to_quantize"),
        ("int4_nodes_to_exclude", "nodes_to_exclude"),
        ("int4_algo_config", "algo_config"),
    ],
)
def test_deprecated_alias_is_renamed_to_new_name(old_name, new_name):
    kv_pairs = {old_name: "value"}
    builder_module.apply_deprecated_extra_option_aliases(kv_pairs)
    assert old_name not in kv_pairs
    assert kv_pairs[new_name] == "value"


def test_new_name_wins_when_both_provided():
    kv_pairs = {"int4_algo_config": "old", "algo_config": "new"}
    builder_module.apply_deprecated_extra_option_aliases(kv_pairs)
    assert kv_pairs == {"algo_config": "new"}


def test_check_extra_options_applies_deprecated_aliases():
    # Old name flows through check_extra_options and is parsed like the new one.
    kv_pairs = {"int4_is_symmetric": "false", "int4_op_types_to_quantize": "MatMul/Gather"}
    builder_module.check_extra_options(kv_pairs, "int4", "cpu")
    assert "int4_is_symmetric" not in kv_pairs
    assert kv_pairs["is_symmetric"] is False
    assert kv_pairs["op_types_to_quantize"] == ("MatMul", "Gather")


