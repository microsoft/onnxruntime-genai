# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["unpack_uint4", "to_int4"]

from typing import Sequence

import ml_dtypes
import numpy as np
from onnxscript import ir
from onnxruntime import __version__ as ort_version
from packaging import version

if version.parse(ort_version) > version.parse("1.21.1"):
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        QuantFormat,
    )
else:
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        MatMul4BitsQuantizer as MatMulNBitsQuantizer,
        QuantFormat,
    )


def _unpack_uint4_as_uint8(data: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def unpack_uint4(data: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    data = data.view(np.uint8).flatten()
    return _unpack_uint4_as_uint8(data, dims).view(ml_dtypes.uint4)


def to_int4(
    model: ir.Model,
    *,
    block_size: int,
    is_symmetric: bool,
    accuracy_level: int,
    use_qdq: bool,
    op_types_to_quantize: tuple[str, ...],
) -> ir.Model:
    """Quantize the model to int4."""
    ir.external_data.load_to_model(model)
    quant = MatMulNBitsQuantizer(
        model=ir.to_proto(model),
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
        nodes_to_exclude=[],
        quant_format=(QuantFormat.QDQ if use_qdq else QuantFormat.QOperator),
        op_types_to_quantize=op_types_to_quantize,
    )
    quant.process()
    return ir.from_proto(quant.model.model)
