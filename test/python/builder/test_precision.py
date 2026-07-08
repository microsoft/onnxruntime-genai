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
# int8 keeps FP32 weights + I/O so graph construction avoids the INT8/UINT8
# `NotImplementedError` branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_symmetric", [True, False, None])
def test_int8_onnx_dtype_stays_float(is_symmetric):
    extra_options = {} if is_symmetric is None else {"is_symmetric": is_symmetric}
    assert builder_module.set_onnx_dtype("int8", extra_options) == ir.DataType.FLOAT


@pytest.mark.parametrize(
    "is_symmetric, expected",
    [
        (True, ir.DataType.INT4),
        (False, ir.DataType.UINT4),
    ],
)
def test_int4_onnx_dtype_is_still_int4(is_symmetric, expected):
    assert builder_module.set_onnx_dtype("int4", {"is_symmetric": is_symmetric}) == expected


def test_int8_io_dtype_is_fp32():
    assert builder_module.set_io_dtype("int8", "cpu", {}) == ir.DataType.FLOAT


# ---------------------------------------------------------------------------
# int8's FP32 onnx_dtype routes to the float MatMul builders (no NotImplementedError).
# ---------------------------------------------------------------------------


def _make_bare_model(onnx_dtype, quant_attrs=None):
    model = Model.__new__(Model)
    model.onnx_dtype = onnx_dtype
    model.quant_attrs = quant_attrs if quant_attrs is not None else {"use_qdq": False}
    return model


def test_make_matmul_op_float_path_for_int8_dtype(monkeypatch):
    model = _make_bare_model(ir.DataType.FLOAT)
    sentinel = object()
    monkeypatch.setattr(model, "make_matmul_float", lambda *a, **k: sentinel)

    assert model.make_matmul_op(object(), "/lm_head/MatMul", "root") is sentinel


def test_make_packed_matmul_float_path_for_int8_dtype(monkeypatch):
    model = _make_bare_model(ir.DataType.FLOAT)
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


def test_int8_with_qdq_is_rejected(monkeypatch):
    config = types.SimpleNamespace(architectures=["LlamaForCausalLM"])
    monkeypatch.setattr(builder_module.AutoConfig, "from_pretrained", lambda *a, **k: config)

    with pytest.raises(NotImplementedError, match="QDQ"):
        builder_module.create_model(
            model_name="dummy",
            input_path="",
            output_dir=".",
            precision="int8",
            execution_provider="cpu",
            cache_dir=".",
            use_qdq=True,
        )

