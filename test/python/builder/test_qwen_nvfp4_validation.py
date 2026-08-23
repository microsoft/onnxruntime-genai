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


def test_native_nvfp4_moe_recognizes_preprocessed_experts():
    model = object.__new__(Qwen35MoETextModel)
    model.moe_attrs = {"quant_type": "nvfp4"}
    moe = SimpleNamespace(experts=SimpleNamespace(quant_type="nvfp4"))

    assert model.is_native_nvfp4_moe(moe)
