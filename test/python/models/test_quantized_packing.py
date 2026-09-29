# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Packing layout and memory regression tests using small CPU tensors."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders.base import QuantizedModel


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "bits,packed_dtype,expected",
    [
        (2, torch.uint8, [[0xE4, 0xE4, 0], [0x1B, 0x1B, 3]]),
        (2, torch.int32, [[0xE4E4], [0x31B1B]]),
        (4, torch.uint8, [[0x10, 0x32, 0x54, 0x76, 8], [0xEF, 0xCD, 0xAB, 0x89, 7]]),
        (4, torch.int32, [[0x76543210, 8], [0x89ABCDEF - 2**32, 7]]),
        (8, torch.uint8, [list(range(9)), list(range(255, 246, -1))]),
        (8, torch.int32, [[0x03020100, 0x07060504, 8], [0xFCFDFEFF - 2**32, 0xF8F9FAFB - 2**32, 0xF7]]),
    ],
)
def test_pack_on_row_known_bit_layout(bits, packed_dtype, expected, transpose):
    tensor = torch.tensor([list(range(9)), list(range(-1, -10, -1))], dtype=torch.int32)
    expected = torch.tensor(expected, dtype=packed_dtype)
    if transpose:
        tensor = tensor.T
        expected = expected.T

    actual = object.__new__(QuantizedModel).pack_on_row(tensor, bits, transpose, packed_dtype)

    assert torch.equal(actual, expected)
    assert actual.dtype == packed_dtype


@pytest.mark.parametrize("shape", [(3, 19), (19, 3), (7, 32), (1, 1)])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64, torch.uint8])
@pytest.mark.parametrize("packed_dtype", [torch.int32, torch.uint8])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("transpose", [False, True])
def test_pack_on_row_matches_scalar_layout(shape, input_dtype, packed_dtype, bits, transpose):
    # A strided view also checks that packing does not write into the input's backing storage.
    backing = (torch.arange(shape[0] * shape[1] * 2).reshape(shape[0], shape[1] * 2) * 7919 - 100000).to(input_dtype)
    tensor = backing[:, ::2]
    saved = backing.clone()
    rows = (tensor.T if transpose else tensor).tolist()
    width = torch.iinfo(packed_dtype).bits
    values_per_pack = width // bits
    expected = []
    for row in rows:
        packed_row = []
        for start in range(0, len(row), values_per_pack):
            # Express each word as a base-(2**bits) integer, independently of tensor operations.
            word = sum(
                (int(value) % (2**bits)) * (2 ** (index * bits))
                for index, value in enumerate(row[start : start + values_per_pack])
            )
            if packed_dtype == torch.int32 and word >= 2**31:
                word -= 2**32
            packed_row.append(word)
        expected.append(packed_row)
    expected = torch.tensor(expected, dtype=packed_dtype)
    if transpose:
        expected = expected.T

    actual = object.__new__(QuantizedModel).pack_on_row(tensor, bits, transpose, packed_dtype)

    assert torch.equal(actual, expected)
    assert actual.dtype == packed_dtype
    assert actual.device == tensor.device
    assert torch.equal(backing, saved)


def test_pack_on_row_does_not_expand_intermediate_tensors():
    tensor = torch.arange(257 * 19, dtype=torch.int32).reshape(257, 19)
    memory_budget = 2 * tensor.numel() * tensor.element_size()

    class CheckIntermediateSize(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    assert output.numel() * output.element_size() <= memory_budget, str(func)
            return result

    # The previous per-bit int64 expansion exceeds this budget even on this tiny input.
    with CheckIntermediateSize():
        actual = object.__new__(QuantizedModel).pack_on_row(tensor, 4, False, torch.uint8)

    assert actual.shape == (257, 10)
