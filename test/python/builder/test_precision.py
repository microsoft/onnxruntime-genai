# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for `--precision int8` support in the model builder.

int8 precision maps `onnx_dtype` to INT8/UINT8 (mirroring int4 -> INT4/UINT4), builds a
float graph, and quantizes the dense weights to 8-bit `MatMulNBits` at save time via
`to_nbits`. These tests exercise the precision plumbing standalone (no model download).
"""

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
    # (`set_onnx_dtype` / `set_io_dtype` / `check_extra_options`) without pulling in every
    # model builder.
    builders_stub = types.ModuleType("builders")

    def _stub_getattr(name):  # PEP 562: satisfies `from builders import <ModelClass>`
        return type(name, (), {})

    builders_stub.__getattr__ = _stub_getattr
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
    # int8 must not assume FP32 I/O everywhere: GPU/WebGPU use FP16, only CPU uses FP32.
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
    model.ep = "cpu"  # keep the CUDA prepack post-pass a no-op
    model.matmulnbits_weights_prepacked = 0
    model.quantization_algo = "default"
    model.int4_customized_weight_config = {}
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


def _run_check_extra_options(
    monkeypatch,
    extra_options,
    *,
    precision="int4",
    execution_provider="cpu",
    tie_word_embeddings=True,
):
    # Avoid Hugging Face network/config loading and provide only the config fields needed.
    fake_config = types.SimpleNamespace(tie_word_embeddings=tie_word_embeddings)

    def _fake_get_hf_details(*_args, **_kwargs):
        return {
            "extra_kwargs": {},
            "hf_name": "fake-model",
            "hf_config": fake_config,
        }

    monkeypatch.setattr(builder_module, "get_hf_details", _fake_get_hf_details)
    builder_module.check_extra_options(
        model_name="fake-model",
        input_path="/tmp/fake-model",
        output_dir="/tmp/fake-output",
        precision=precision,
        execution_provider=execution_provider,
        cache_dir="/tmp/fake-cache",
        extra_options=extra_options,
    )


# ---------------------------------------------------------------------------
# int8 rejects the unsupported QDQ format (8-bit MatMulNBits is QOperator-only).
# ---------------------------------------------------------------------------


def test_int8_with_qdq_is_rejected(monkeypatch):
    with pytest.raises(NotImplementedError, match="QDQ"):
        _run_check_extra_options(monkeypatch, {"use_qdq": "true"}, precision="int8")


def test_int4_with_qdq_is_allowed(monkeypatch):
    # QDQ is only rejected for int8; int4 still supports it.
    _run_check_extra_options(monkeypatch, {"use_qdq": "true"}, precision="int4")


# ---------------------------------------------------------------------------
# Deprecated int4_* extra_option names still map to the generalized names.
# ---------------------------------------------------------------------------


def test_deprecated_int4_aliases_are_renamed():
    kv = {
        "int4_accuracy_level": "2",
        "int4_block_size": "64",
        "int4_is_symmetric": "false",
        "int4_op_types_to_quantize": "MatMul/Gather",
        "int4_nodes_to_exclude": "/lm_head/MatMul",
        "int4_algo_config": "k_quant",
    }
    builder_module.apply_deprecated_extra_option_aliases(kv)
    assert kv == {
        "accuracy_level": "2",
        "block_size": "64",
        "is_symmetric": "false",
        "op_types_to_quantize": "MatMul/Gather",
        "nodes_to_exclude": "/lm_head/MatMul",
        "algo_config": "k_quant",
    }


def test_deprecated_alias_does_not_override_new_name():
    # If both the old and new names are provided, the new name wins and the old key is dropped.
    kv = {"int4_algo_config": "rtn", "algo_config": "k_quant"}
    builder_module.apply_deprecated_extra_option_aliases(kv)
    assert kv == {"algo_config": "k_quant"}


def test_check_extra_options_accepts_deprecated_int4_names(monkeypatch):
    # End-to-end through check_extra_options: deprecated names are normalized in-place.
    kv = {"int4_algo_config": "k_quant", "int4_op_types_to_quantize": "MatMul/Gather"}
    _run_check_extra_options(monkeypatch, kv, precision="int4")
    assert kv["algo_config"] == "k_quant"
    assert kv["op_types_to_quantize"] == ("MatMul", "Gather")
    assert "int4_algo_config" not in kv and "int4_op_types_to_quantize" not in kv


def test_shared_embeddings_with_untied_weights_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="tie_word_embeddings=false"):
        _run_check_extra_options(
            monkeypatch,
            {"shared_embeddings": "true"},
            precision="int4",
            tie_word_embeddings=False,
        )


def test_shared_embeddings_with_tied_weights_is_accepted(monkeypatch):
    # Should not raise when tie_word_embeddings=True
    _run_check_extra_options(
        monkeypatch,
        {"shared_embeddings": "true"},
        precision="int4",
        tie_word_embeddings=True,
    )


def test_shared_embeddings_defaults_to_tied_when_config_ties_embeddings(monkeypatch):
    # When shared_embeddings is not specified, it defaults to tie_word_embeddings value
    # Should not raise because shared_embeddings will default to True when tie_word_embeddings=True
    _run_check_extra_options(
        monkeypatch,
        {},
        precision="int4",
        tie_word_embeddings=True,
    )


def test_shared_embeddings_defaults_to_false_when_config_doesnt_tie_embeddings(monkeypatch):
    # When shared_embeddings is not specified and tie_word_embeddings=False,
    # shared_embeddings will default to False
    _run_check_extra_options(
        monkeypatch,
        {},
        precision="int4",
        tie_word_embeddings=False,
    )


def test_shared_embeddings_handles_none_tie_word_embeddings(monkeypatch):
    # When tie_word_embeddings is None, it should default to False
    with pytest.raises(ValueError, match="tie_word_embeddings=false"):
        _run_check_extra_options(
            monkeypatch,
            {"shared_embeddings": "true"},
            precision="int4",
            tie_word_embeddings=None,
        )
