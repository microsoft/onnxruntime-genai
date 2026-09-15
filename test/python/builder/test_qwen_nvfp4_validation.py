# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path

import onnx_ir as ir
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.base import Model


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


def _recording_native_matmul_model(use_paged_attention):
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 8
    model.use_paged_attention = use_paged_attention
    model.values = []
    model.make_initializer = lambda *args, **kwargs: None
    model.make_node = lambda *args, **kwargs: None
    model.make_value = lambda name, dtype, shape: model.values.append((name, shape))
    return model


@pytest.mark.parametrize("paged,expected", [(False, ["batch_size", "sequence_length", 4]), (True, ["num_tokens", 4])])
def test_native_fp8_matmul_value_shape_follows_layout(paged, expected):
    model = _recording_native_matmul_model(paged)

    model.make_matmul_block_quantized_fp8_weight(
        "/attention/MatMul",
        "input",
        torch.ones((4, 8), dtype=torch.float8_e4m3fn),
        torch.ones((4, 1)),
    )

    assert model.values[-1][1] == expected


@pytest.mark.parametrize("paged,expected", [(False, ["batch_size", "sequence_length", 4]), (True, ["num_tokens", 4])])
def test_native_nvfp4_matmul_value_shape_follows_layout(paged, expected):
    model = _recording_native_matmul_model(paged)

    model.make_matmul_block_quantized_nvfp4_weight(
        "/lm_head/MatMul",
        "input",
        torch.ones((4, 8), dtype=torch.uint8),
        torch.ones((4, 1), dtype=torch.float8_e4m3fn),
        1.0,
    )

    assert model.values[-1][1] == expected
