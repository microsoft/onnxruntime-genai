# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for CUDA QMoE expert-weight quantization in the model builder.

These guard the INT4 QMoE encoding fix for Qwen3.5/3.6 MoE models. The CUDA path
must:
  1. dispatch to the CUTLASS-prepack quantizer for ``weights_prepacked`` -1/1
     (both mean the op reads prepacked weights) and to the raw MatMulNBits
     quantizer for ``weights_prepacked`` 0,
  2. only do so on the CUDA EP,
  3. reject unsupported block sizes with a real exception (not a ``python -O``
     strippable ``assert``), and
  4. keep the SIGNED blockwise scales (taking ``abs()`` reintroduces the
     garbage-output bug).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _stub_missing_builder_dependencies():
    if not _module_available("onnx_ir"):
        onnx_ir = types.ModuleType("onnx_ir")
        onnx_ir.DataType = types.SimpleNamespace(INT4=object(), FLOAT=object(), FLOAT16=object(), BFLOAT16=object())
        tensor_adapters = types.ModuleType("onnx_ir.tensor_adapters")
        tensor_adapters.TorchTensor = object
        tensor_adapters.to_torch_dtype = lambda dtype: dtype
        sys.modules["onnx_ir"] = onnx_ir
        sys.modules["onnx_ir.tensor_adapters"] = tensor_adapters

    if not _module_available("onnxruntime.quantization.matmul_nbits_quantizer"):
        onnxruntime = sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
        quantization = types.ModuleType("onnxruntime.quantization")
        matmul_nbits_quantizer = types.ModuleType("onnxruntime.quantization.matmul_nbits_quantizer")
        for class_name in (
            "KQuantWeightOnlyQuantConfig",
            "MatMulNBitsQuantizer",
            "QuantFormat",
            "RTNWeightOnlyQuantConfig",
        ):
            setattr(matmul_nbits_quantizer, class_name, type(class_name, (), {}))
        onnxruntime.quantization = quantization
        quantization.matmul_nbits_quantizer = matmul_nbits_quantizer
        sys.modules["onnxruntime.quantization"] = quantization
        sys.modules["onnxruntime.quantization.matmul_nbits_quantizer"] = matmul_nbits_quantizer

    if not _module_available("tqdm"):
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = tqdm_module

    if not _module_available("transformers"):
        transformers = types.ModuleType("transformers")
        for class_name in (
            "AutoConfig",
            "AutoModelForCausalLM",
            "AutoModelForSpeechSeq2Seq",
            "AutoTokenizer",
            "GenerationConfig",
        ):
            setattr(transformers, class_name, type(class_name, (), {}))
        sys.modules["transformers"] = transformers


_stub_missing_builder_dependencies()

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
Model = base_module.Model


class _FakeMoEModel:
    """Minimal stand-in exposing ``make_qmoe_weights`` and recording which
    quantization path it dispatched to."""

    make_qmoe_weights = Model.make_qmoe_weights

    def __init__(self, ep, block_size, weights_prepacked):
        self.ep = ep
        self.qmoe_block_size = block_size
        self.moe_attrs = {"expert_weight_bits": 4, "weights_prepacked": weights_prepacked}
        self.quant_attrs = {"qmoe_block_size": block_size}
        self.calls = []

    def _cutlass_prepacked_blockwise_quantize(self, weights):
        self.calls.append(("cutlass", self.qmoe_block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _matmulnbits_blockwise_quantize(self, weights):
        self.calls.append(("matmulnbits", self.qmoe_block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _symmetric_blockwise_quantize(self, weights, block_size):
        self.calls.append(("symmetric", block_size))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)


_W = torch.zeros(8, 128)  # dummy expert weight [N, K]


@pytest.mark.parametrize("weights_prepacked", [-1, 1])
def test_cuda_prepacked_path_for_auto_and_one(weights_prepacked):
    """-1 (auto=prepacked) and 1 (explicitly prepacked) both produce
    CUTLASS-prepacked weights so the emitted layout matches the op attribute."""
    model = _FakeMoEModel("cuda", 128, weights_prepacked)
    model.make_qmoe_weights(_W)
    assert model.calls == [("cutlass", 128)]


def test_cuda_raw_path_for_zero():
    """weights_prepacked=0 ships raw weights for the runtime PrePack hook."""
    model = _FakeMoEModel("cuda", 128, 0)
    model.make_qmoe_weights(_W)
    assert model.calls == [("matmulnbits", 128)]


def test_non_cuda_does_not_use_cuda_only_paths():
    """The CUDA-only encodings must not be used on other EPs, even when
    weights_prepacked is set."""
    model = _FakeMoEModel("cpu", 128, 0)
    model.make_qmoe_weights(_W)
    assert ("cutlass", 128) not in model.calls
    assert ("matmulnbits", 128) not in model.calls
    assert model.calls == [("symmetric", 128)]


@pytest.mark.parametrize("weights_prepacked", [-1, 0, 1])
@pytest.mark.parametrize("bad_block", [16, 256])
def test_cuda_rejects_unsupported_block_size(weights_prepacked, bad_block):
    """Unsupported block sizes must raise a real exception (not an assert that
    ``python -O`` would strip)."""
    model = _FakeMoEModel("cuda", bad_block, weights_prepacked)
    with pytest.raises(ValueError, match="block_size 32, 64, or 128"):
        model.make_qmoe_weights(_W)


def _ort_cuda_available():
    try:
        import onnxruntime as ort  # noqa: PLC0415
        from onnxruntime.capi import _pybind_state as _pyb  # noqa: PLC0415

        return (
            hasattr(_pyb, "quantize_matmul_4bits")
            and hasattr(_pyb, "pack_weights_for_cuda_mixed_gemm")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        )
    except Exception:
        return False


@pytest.mark.skipif(not _ort_cuda_available(), reason="onnxruntime CUDA pybind not available")
def test_cutlass_prepacked_scales_are_signed():
    """Regression guard for the abs(scales) bug: blockwise scales must keep their
    sign, and the encoded shapes must match the QMoE op's prepacked layout."""
    model = _FakeMoEModel("cuda", 128, -1)
    torch.manual_seed(0)
    weights = torch.randn(256, 256) * 0.05  # [N, K]
    qweight, scales = Model._cutlass_prepacked_blockwise_quantize(model, weights)

    assert qweight.dtype == torch.uint8
    assert tuple(qweight.shape) == (256, 128)  # [K, N/2] for INT4
    assert tuple(scales.shape) == (256, 2)  # [N, K/block]
    # With zero-mean weights about half the blocks have a negative anchor; the
    # scale must carry that sign. abs() would make every scale non-negative.
    assert (scales < 0).any(), "blockwise scales should be signed (abs() regression)"
