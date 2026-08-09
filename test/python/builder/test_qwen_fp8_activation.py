# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import MethodType

import onnx_ir as ir
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.qwen import Qwen35MoeTextModel


def _make_model(scales):
    """Build a minimal Qwen35MoeTextModel stub that records emitted initializers.

    ``MatMulBlockQuantizedFp8Weight`` takes the FP16/BF16 activation directly plus an
    optional fp32 *scalar* ``a_scale``, so the builder no longer emits an FP8 quantization
    subgraph -- only a scalar initializer per distinct calibrated scale.
    """
    model = object.__new__(Qwen35MoeTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model._fp8_attention_activation_cache = {}
    model._fp8_weight_key_for_matmul = MethodType(lambda self, basename: basename, model)

    def load_scale(self, key):
        basename = key.removesuffix(".input_scale")
        if basename not in scales:
            raise RuntimeError(f"missing {key}")
        return torch.tensor(scales[basename])

    model._load_nvfp4_tensor = MethodType(load_scale, model)

    initializers = []

    def record_initializer(self, tensor, name, **kwargs):
        initializers.append((name, tensor))
        return name

    model.make_initializer = MethodType(record_initializer, model)
    return model, initializers


def test_static_input_scale_is_read_from_checkpoint():
    model, _ = _make_model({"q": 0.125})

    assert model._fp8_attention_input_scale("q") == 0.125


def test_static_input_scale_rejects_missing_checkpoint_scale():
    model, _ = _make_model({})

    with pytest.raises(RuntimeError, match="missing q.input_scale"):
        model._fp8_attention_input_scale("q")


@pytest.mark.parametrize("scale", [0.0, -0.125, float("inf"), float("nan")])
def test_static_input_scale_rejects_non_positive_or_non_finite_values(scale):
    model, _ = _make_model({"q": scale})

    with pytest.raises(ValueError, match="finite and positive"):
        model._fp8_attention_input_scale("q")


def test_static_input_scale_rejects_non_scalar_tensor():
    model, _ = _make_model({"q": [0.125, 0.25]})

    with pytest.raises(ValueError, match="must be a scalar"):
        model._fp8_attention_input_scale("q")


def test_scale_initializer_is_shared_for_matching_qkv_scales():
    model, initializers = _make_model({"q": 0.125, "k": 0.125, "v": 0.125})

    q_name = model._make_fp8_activation_scale_initializer("/q", 0.125)
    k_name = model._make_fp8_activation_scale_initializer("/k", 0.125)
    v_name = model._make_fp8_activation_scale_initializer("/v", 0.125)

    assert k_name == q_name
    assert v_name == q_name
    assert len(initializers) == 1
    assert initializers[0][1].dtype == torch.float32
    assert initializers[0][1].numel() == 1


def test_scale_initializer_is_not_shared_when_scale_differs():
    model, initializers = _make_model({"q": 0.125, "o": 0.25})

    q_name = model._make_fp8_activation_scale_initializer("/q", 0.125)
    o_name = model._make_fp8_activation_scale_initializer("/o", 0.25)

    assert o_name != q_name
    assert len(initializers) == 2
