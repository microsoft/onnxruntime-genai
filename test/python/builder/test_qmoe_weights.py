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
  4. restore per-channel quantization when ``qmoe_block_size <= 0``, and
  5. keep the SIGNED blockwise scales (taking ``abs()`` reintroduces the
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
    except (ModuleNotFoundError, ValueError):
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
        # Prefer the real onnxruntime package when it is installed; only fabricate a
        # top-level stub when the package truly isn't available. This avoids shadowing a
        # real onnxruntime wheel (which would break other tests in the session) and only
        # supplies the specific submodule the builder needs.
        if _module_available("onnxruntime"):
            import onnxruntime  # noqa: PLC0415
        else:
            onnxruntime = sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
        quantization = getattr(onnxruntime, "quantization", None)
        if quantization is None:
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
cuda_quantizer_module = sys.modules["models.builders.cuda_quantizer"]
qmoe_symmetric_per_channel_quantize = cuda_quantizer_module.CudaQuantizer.qmoe_symmetric_per_channel_quantize
gptoss_module = _load_builder_module("gptoss")
GPTOSSModel = gptoss_module.GPTOSSModel


def _load_builder_cli_module(monkeypatch):
    builders_module = types.ModuleType("builders")
    for class_name in (
        "ChatGLMModel",
        "ErnieModel",
        "Gemma2Model",
        "Gemma3Model",
        "GemmaModel",
        "GPTOSSModel",
        "GraniteModel",
        "HunyuanDenseV1Model",
        "InternLM2Model",
        "LFM2Model",
        "LlamaModel",
        "Mistral3TextModel",
        "MistralModel",
        "Model",
        "NemotronModel",
        "OLMoModel",
        "Phi3MiniLongRoPEModel",
        "Phi3MiniModel",
        "Phi3MoELongRoPEModel",
        "Phi3SmallLongRoPEModel",
        "Phi3SmallModel",
        "Phi3VModel",
        "Phi4MMModel",
        "PhiModel",
        "Qwen25VLTextModel",
        "Qwen35MoeTextModel",
        "Qwen35TextModel",
        "Qwen3Model",
        "Qwen3VLTextModel",
        "QwenModel",
        "SmolLM3Model",
        "VideoChatFlashQwenModel",
        "WhisperModel",
    ):
        setattr(builders_module, class_name, type(class_name, (), {}))
    monkeypatch.setitem(sys.modules, "builders", builders_module)

    module_name = "builder_cli_under_test"
    spec = importlib.util.spec_from_file_location(module_name, BUILDERS_DIR.parent / "builder.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_moe_quant_type_mxfp4_is_accepted(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = builder.parse_extra_options(["moe_quant_type=mxfp4"], "int4", "cuda")
    assert options["moe_quant_type"] == "mxfp4"


def test_moe_quant_type_rejects_invalid_value(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match="moe_quant_type must be one of"):
        builder.parse_extra_options(["moe_quant_type=fp8"], "int4", "cuda")


def test_use_fp4_moe_is_removed(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match="'use_fp4_moe' has been removed"):
        builder.parse_extra_options(["use_fp4_moe=true"], "int4", "cuda")


def test_use_8bits_moe_maps_to_moe_quant_type(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    options = builder.parse_extra_options(["use_8bits_moe=true"], "int4", "cuda")
    assert options["moe_quant_type"] == "int8"


def test_moe_quant_type_mxfp4_requires_qmoe_precision(monkeypatch):
    builder = _load_builder_cli_module(monkeypatch)
    with pytest.raises(ValueError, match="moe_quant_type=mxfp4 requires precision=int4"):
        builder.parse_extra_options(["moe_quant_type=mxfp4"], "fp16", "cuda")


def test_gptoss_fp4_rejects_quark_experts_before_emitting_nodes():
    model = types.SimpleNamespace(
        moe_attrs={"op_type": "QMoE", "quant_type": "fp4"},
        has_quark_experts=lambda experts: True,
    )
    mlp = types.SimpleNamespace(experts=types.SimpleNamespace(fc1_weights=torch.empty(0), fc2_weights=torch.empty(0)))

    with pytest.raises(ValueError, match="pre-quantized Quark GPT-OSS experts"):
        GPTOSSModel.make_moe_fused(model, 0, mlp, "root")


def test_gptoss_original_mxfp4_blocks_pack_to_qmoe_layout():
    blocks = torch.arange(2 * 4 * 2 * 16, dtype=torch.uint8).reshape(2, 4, 2, 16)
    packed = GPTOSSModel.pack_original_mxfp4_blocks_for_qmoe(blocks)

    low_codes = blocks & 0x0F
    high_codes = blocks >> 4
    codes = torch.empty(2, 4, 2, 32, dtype=torch.uint8)
    codes[..., 0::2] = low_codes
    codes[..., 1::2] = high_codes
    codes = codes.reshape(2, 4, 64)
    codes_kn = codes.permute(0, 2, 1).contiguous()
    expected = (codes_kn[..., 1::2] << 4) | codes_kn[..., 0::2]

    assert packed.shape == (2, 64, 2)
    assert torch.equal(packed, expected)


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

    def _cuda_per_channel_quantize(self, weights, prepack):
        self.calls.append(("cuda_per_channel", prepack))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)

    def _symmetric_per_channel_quantize(self, weights):
        self.calls.append(("per_channel",))
        return torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.float32)


class _RealMoEModel:
    make_qmoe_weights = Model.make_qmoe_weights
    _symmetric_per_channel_quantize = Model._symmetric_per_channel_quantize
    _cuda_per_channel_quantize = Model._cuda_per_channel_quantize

    def __init__(self, ep, block_size, weights_prepacked, bits=4, extra_options=None):
        self.ep = ep
        self.qmoe_block_size = block_size
        self.moe_attrs = {"expert_weight_bits": bits, "weights_prepacked": weights_prepacked}
        self.quant_attrs = {"qmoe_block_size": block_size}
        self.extra_options = extra_options or {}


class _FakeGPTOSSModel:
    make_qmoe_weight_initializer_shapes = GPTOSSModel.make_qmoe_weight_initializer_shapes

    def __init__(self, ep, weights_prepacked):
        self.ep = ep
        self.hidden_size = 96
        self.intermediate_size = 128
        self.moe_attrs = {"expert_weight_bits": 4, "num_experts": 2, "weights_prepacked": weights_prepacked}


_W = torch.zeros(8, 128)  # dummy expert weight [N, K]


@pytest.mark.parametrize(
    "weights_prepacked,gate_shape,down_shape,expected_gate_shape,expected_down_shape",
    [
        (-1, (96, 128), (128, 48), (2, 96, 128), (2, 128, 48)),
        (0, (256, 48), (96, 64), (2, 256, 48), (2, 96, 64)),
        (1, (96, 128), (128, 48), (2, 96, 128), (2, 128, 48)),
    ],
)
def test_gptoss_qmoe_initializer_shapes_match_schema(
    weights_prepacked,
    gate_shape,
    down_shape,
    expected_gate_shape,
    expected_down_shape,
):
    model = _FakeGPTOSSModel("cuda", weights_prepacked)
    gate_up_qweights = [torch.zeros(gate_shape, dtype=torch.uint8) for _ in range(model.moe_attrs["num_experts"])]
    down_qweights = [torch.zeros(down_shape, dtype=torch.uint8) for _ in range(model.moe_attrs["num_experts"])]

    gate_up_initializer_shape, down_initializer_shape = model.make_qmoe_weight_initializer_shapes(
        gate_up_qweights,
        down_qweights,
        has_quark_experts=False,
    )

    assert tuple(torch.stack(gate_up_qweights, dim=0).view(gate_up_initializer_shape).shape) == expected_gate_shape
    assert tuple(torch.stack(down_qweights, dim=0).view(down_initializer_shape).shape) == expected_down_shape


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


@pytest.mark.parametrize(
    "weights_prepacked,expected_prepack",
    [(-1, True), (0, False), (1, True)],
)
@pytest.mark.parametrize("block_size", [-1, 0])
def test_cuda_per_channel_path_for_non_positive_block_size(block_size, weights_prepacked, expected_prepack):
    """qmoe_block_size <= 0 means per-channel QMoE scales and must not emit a
    block_size attribute."""
    model = _FakeMoEModel("cuda", block_size, weights_prepacked)
    model.moe_attrs["block_size"] = 128
    model.make_qmoe_weights(_W)
    assert model.calls == [("cuda_per_channel", expected_prepack)]
    assert "block_size" not in model.moe_attrs


def test_non_cuda_per_channel_path_for_non_positive_block_size():
    model = _FakeMoEModel("cpu", 0, -1)
    model.moe_attrs["block_size"] = 128
    model.make_qmoe_weights(_W)
    assert model.calls == [("per_channel",)]
    assert "block_size" not in model.moe_attrs


def test_per_channel_int4_uses_qmoe_full_range_unsigned_offset_storage_by_default():
    model = _RealMoEModel("cpu", 0, -1, bits=4)
    weights = torch.tensor([[-8.0, -7.0, 0.0, 7.0]], dtype=torch.float32)

    qweight, scales = model._symmetric_per_channel_quantize(weights)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, torch.tensor([[0x10, 0xF8]], dtype=torch.uint8))


def test_per_channel_int8_uses_qmoe_full_range_unsigned_offset_storage_by_default():
    model = _RealMoEModel("cpu", 0, -1, bits=8)
    weights = torch.tensor([[-128.0, -127.0, 0.0, 127.0]], dtype=torch.float32)

    qweight, scales = model._symmetric_per_channel_quantize(weights)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, torch.tensor([[0x00, 0x01, 0x80, 0xFF]], dtype=torch.uint8))


@pytest.mark.parametrize(
    "bits,weights,expected_qweight",
    [
        (
            4,
            torch.tensor([[-8.0, -7.0, 0.0, 7.0]], dtype=torch.float32),
            torch.tensor([[0x10, 0xF8]], dtype=torch.uint8),
        ),
        (
            8,
            torch.tensor([[-128.0, -127.0, 0.0, 127.0]], dtype=torch.float32),
            torch.tensor([[0x00, 0x01, 0x80, 0xFF]], dtype=torch.uint8),
        ),
    ],
)
def test_per_channel_can_emit_full_range_unsigned_offset_storage(bits, weights, expected_qweight):
    qweight, scales = qmoe_symmetric_per_channel_quantize(weights, bits)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, expected_qweight)


@pytest.mark.parametrize(
    "bits,weights,expected_qweight",
    [
        (
            4,
            torch.tensor([[-7.0, -3.0, 0.0, 7.0]], dtype=torch.float32),
            torch.tensor([[0x51, 0xF8]], dtype=torch.uint8),
        ),
        (
            8,
            torch.tensor([[-127.0, -1.0, 0.0, 127.0]], dtype=torch.float32),
            torch.tensor([[0x01, 0x7F, 0x80, 0xFF]], dtype=torch.uint8),
        ),
    ],
)
def test_per_channel_can_emit_legacy_15_bucket_unsigned_offset_storage(bits, weights, expected_qweight):
    qweight, scales = qmoe_symmetric_per_channel_quantize(weights, bits, unsigned_full_range=False)

    assert torch.equal(scales, torch.tensor([1.0]))
    assert torch.equal(qweight, expected_qweight)


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


def _ort_qmoe_pack_available():
    try:
        from onnxruntime.capi import _pybind_state as _pyb  # noqa: PLC0415

        return hasattr(_pyb, "pack_weights_for_cuda_mixed_gemm")
    except Exception:
        return False


@pytest.mark.skipif(
    not _ort_qmoe_pack_available(),
    reason="onnxruntime QMoE pack pybind not available",
)
@pytest.mark.parametrize(
    "bits,raw_shape,packed_shape",
    [(4, (64, 64), (128, 32)), (8, (64, 128), (128, 64))],
)
def test_cuda_per_channel_quantization_shapes(bits, raw_shape, packed_shape):
    torch.manual_seed(0)
    weights = torch.randn(64, 128) * 0.05

    raw_model = _RealMoEModel("cuda", 0, 0, bits=bits)
    raw_qweight, raw_scales = raw_model.make_qmoe_weights(weights)
    assert raw_qweight.dtype == torch.uint8
    assert tuple(raw_qweight.shape) == raw_shape
    assert tuple(raw_scales.shape) == (64,)
    assert (raw_scales > 0).all()
    assert "block_size" not in raw_model.moe_attrs

    prepacked_model = _RealMoEModel("cuda", 0, -1, bits=bits)
    prepacked_qweight, prepacked_scales = prepacked_model.make_qmoe_weights(weights)
    assert prepacked_qweight.dtype == torch.uint8
    assert tuple(prepacked_qweight.shape) == packed_shape
    assert tuple(prepacked_scales.shape) == (64,)
    assert (prepacked_scales > 0).all()
    assert "block_size" not in prepacked_model.moe_attrs


@pytest.mark.parametrize("bits", [4, 8])
def test_cuda_per_channel_quantization_uses_qmoe_symmetric_storage(bits):
    torch.manual_seed(0)
    n = 17
    k = 16
    weights = torch.randn(n, k, dtype=torch.float32) * 0.05
    model = _RealMoEModel("cuda", 0, 0, bits=bits)

    qweight, scales = model.make_qmoe_weights(weights)

    if bits == 4:
        qmin, qmax, scale_divisor, zero_point = -8, 7, 8, 8
    else:
        qmin, qmax, scale_divisor, zero_point = -128, 127, 128, 128

    expected_scales = torch.clamp(
        weights.abs().amax(dim=1, keepdim=True) / float(scale_divisor), min=torch.finfo(torch.float32).eps
    )
    quantized = torch.clamp(torch.round(weights / expected_scales), qmin, qmax).to(torch.int16)
    quantized = (quantized + zero_point).to(torch.uint8)
    if bits == 4:
        expected_qweight = quantized[:, 0::2] | (quantized[:, 1::2] << 4)
    else:
        expected_qweight = quantized

    assert torch.equal(qweight, expected_qweight)
    assert torch.allclose(scales.float(), expected_scales.squeeze(-1), atol=1e-6, rtol=1e-3)
    assert quantized.min() == 0
    assert quantized.max() == (15 if bits == 4 else 255)


def test_cuda_raw_per_channel_quantization_does_not_require_qmoe_pack_pybind(monkeypatch):
    torch.manual_seed(0)
    weights = torch.randn(17, 16, dtype=torch.float32) * 0.05
    model = _RealMoEModel("cuda", 0, 0, bits=4)

    qweight, scales = model.make_qmoe_weights(weights)

    assert qweight.dtype == torch.uint8
    assert tuple(qweight.shape) == (17, 8)
    assert tuple(scales.shape) == (17,)
    assert "block_size" not in model.moe_attrs


@pytest.mark.skipif(not _ort_cuda_available(), reason="onnxruntime CUDA pybind not available")
def test_cutlass_prepacked_scales_are_positive_with_full_range_symmetric_quantization():
    """The synced CudaQuantizer full-range symmetric path returns positive
    scales, and the encoded shapes must match the QMoE op's prepacked layout."""
    model = _FakeMoEModel("cuda", 128, -1)
    torch.manual_seed(0)
    weights = torch.randn(256, 256) * 0.05  # [N, K]
    qweight, scales = Model._cutlass_prepacked_blockwise_quantize(model, weights)

    assert qweight.dtype == torch.uint8
    assert tuple(qweight.shape) == (256, 128)  # [K, N/2] for INT4
    assert tuple(scales.shape) == (256, 2)  # [N, K/block]
    assert (scales > 0).all()
