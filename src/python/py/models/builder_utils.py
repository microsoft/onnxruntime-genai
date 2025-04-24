# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["tensor_proto_string_to_dtype", "unpack_uint4", "to_int4"]

from typing import Sequence

import ml_dtypes
import numpy as np
from onnxscript import ir
from onnxruntime.quantization import matmul_4bits_quantizer


_TENSOR_PROTO_STRING_TO_DTYPE = {
    "TensorProto.UNDEFINED": ir.DataType.UNDEFINED,
    "TensorProto.FLOAT": ir.DataType.FLOAT,
    "TensorProto.UINT8": ir.DataType.UINT8,
    "TensorProto.INT8": ir.DataType.INT8,
    "TensorProto.UINT16": ir.DataType.UINT16,
    "TensorProto.INT16": ir.DataType.INT16,
    "TensorProto.INT32": ir.DataType.INT32,
    "TensorProto.INT64": ir.DataType.INT64,
    "TensorProto.STRING": ir.DataType.STRING,
    "TensorProto.BOOL": ir.DataType.BOOL,
    "TensorProto.FLOAT16": ir.DataType.FLOAT16,
    "TensorProto.DOUBLE": ir.DataType.DOUBLE,
    "TensorProto.UINT32": ir.DataType.UINT32,
    "TensorProto.UINT64": ir.DataType.UINT64,
    "TensorProto.COMPLEX64": ir.DataType.COMPLEX64,
    "TensorProto.COMPLEX128": ir.DataType.COMPLEX128,
    "TensorProto.BFLOAT16": ir.DataType.BFLOAT16,
    "TensorProto.FLOAT8E4M3FN": ir.DataType.FLOAT8E4M3FN,
    "TensorProto.FLOAT8E4M3FNUZ": ir.DataType.FLOAT8E4M3FNUZ,
    "TensorProto.FLOAT8E5M2": ir.DataType.FLOAT8E5M2,
    "TensorProto.FLOAT8E5M2FNUZ": ir.DataType.FLOAT8E5M2FNUZ,
    "TensorProto.UINT4": ir.DataType.UINT4,
    "TensorProto.INT4": ir.DataType.INT4,
    "TensorProto.FLOAT4E2M1": ir.DataType.FLOAT4E2M1,
}


def tensor_proto_string_to_dtype(tensor_proto_string: str) -> ir.DataType:
    """Convert a TensorProto string to an ir.DataType.

    Args:
        tensor_proto_string: A string representing the TensorProto type.

    Returns:
        The corresponding ir.DataType.
    """
    if tensor_proto_string not in _TENSOR_PROTO_STRING_TO_DTYPE:
        raise ValueError(f"Unsupported TensorProto string: {tensor_proto_string}")
    return _TENSOR_PROTO_STRING_TO_DTYPE[tensor_proto_string]


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
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model=ir.to_proto(model),
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
        nodes_to_exclude=[],
        quant_format=(
            matmul_4bits_quantizer.QuantFormat.QDQ
            if use_qdq
            else matmul_4bits_quantizer.QuantFormat.QOperator
        ),
        op_types_to_quantize=op_types_to_quantize,
    )
    quant.process()
    return ir.from_proto(quant.model.model)
