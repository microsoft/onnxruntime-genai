# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.base import Model
from models.builders.qwen import Qwen35MoETextModel


@pytest.fixture
def model():
    return object.__new__(Model)


def test_modelopt_e4m3_bytes_accepts_float8_and_preserves_shape(model):
    scales = torch.ones((4, 2), dtype=torch.float8_e4m3fn)

    raw = model.modelopt_e4m3_bytes(scales, "scales", (4, 2))

    assert raw.dtype == torch.uint8
    assert raw.shape == scales.shape


def test_modelopt_e4m3_bytes_rejects_wrong_dtype(model):
    with pytest.raises(ValueError, match="must contain E4M3 bytes"):
        model.modelopt_e4m3_bytes(torch.ones((4, 2)), "scales", (4, 2))


def test_modelopt_e4m3_bytes_rejects_wrong_shape(model):
    with pytest.raises(ValueError, match=r"expected \(4, 2\)"):
        model.modelopt_e4m3_bytes(torch.ones((2, 4), dtype=torch.uint8), "scales", (4, 2))


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_modelopt_positive_scalar_rejects_invalid_values(model, value):
    with pytest.raises(ValueError, match="finite and positive"):
        model.modelopt_positive_scalar(torch.tensor(value), "global_scale")


def test_native_fp8_rejects_non_fp8_weight():
    model = object.__new__(Model)

    with pytest.raises(ValueError, match="must be float8_e4m3fn"):
        model.make_matmul_block_quantized_fp8_weight(
            "/attention/MatMul", "input", torch.ones((4, 4), dtype=torch.bfloat16), torch.tensor(1.0)
        )


def test_native_nvfp4_rejects_non_nvfp4_weight():
    model = object.__new__(Model)

    with pytest.raises(ValueError, match="packed uint8 codes"):
        model.make_matmul_block_quantized_nvfp4_weight(
            "/lm_head/MatMul", "input", torch.ones((4, 4)), torch.ones((4, 1)), 1.0
        )


def test_nvfp4_qmoe_rejects_mismatched_gate_up_global_scales():
    model = object.__new__(Qwen35MoETextModel)

    def projection(scale):
        return SimpleNamespace(
            weight=torch.zeros((16, 8), dtype=torch.uint8),
            weight_scale=torch.ones((16, 1), dtype=torch.float8_e4m3fn),
            weight_scale_2=torch.tensor(scale),
        )

    experts = [SimpleNamespace(gate_proj=projection(0.5), up_proj=projection(0.25), down_proj=projection(0.5))]

    with pytest.raises(ValueError, match="gate/up global scales must match"):
        model.make_nvfp4_moe_initializers(experts, "gw", "gs", "gg", "dw", "ds", "dg")
