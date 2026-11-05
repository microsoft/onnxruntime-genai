# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Per-channel INT4 with one stored scale per output channel."""

import numpy as np
from onnxruntime.quantization import matmul_nbits_quantizer as q


def per_channel_block_size(k: int) -> int:
    """Smallest power of two (>= 16) covering ``k``, i.e. one block spans an entire channel."""
    return max(16, 1 << (k - 1).bit_length())


class PerChannelQuantizer(q.DefaultWeightOnlyQuantizer):
    def quantize_matmul(self, node, graph_stack):
        previous = self.config.block_size
        try:
            return super().quantize_matmul(node, graph_stack)
        finally:
            self.config.block_size = previous

    def qbits_block_quant(self, weights):
        bits = self.config.bits
        # MatMulNBits symmetric INT4/INT8 uses unsigned nibbles/bytes with an implicit
        # zero point at the range midpoint (8 for 4 bits, 128 for 8 bits).
        qmax = (1 << (bits - 1)) - 1
        zero = 1 << (bits - 1)
        kpack = 8 // bits
        fill = zero | (zero << 4) if bits == 4 else zero

        k, n = weights.shape
        if k == 0 or n == 0:
            raise ValueError("Per-channel MatMul requires nonempty weights")
        block = per_channel_block_size(k)
        self.config.block_size = block
        blob_size = block // kpack
        # Work in column chunks to bound temporary memory for large LM heads.
        packed = np.full((n, 1, blob_size), fill, dtype=np.uint8)
        # Store exactly one scale per output channel.
        scales = np.empty((n, 1), dtype=weights.dtype)
        for start in range(0, n, 256):
            end = min(start + 256, n)
            columns = weights[:, start:end].astype(np.float32).T
            if not np.isfinite(columns).all():
                raise ValueError("Per-channel MatMul weights must be finite")
            scale = np.max(np.abs(columns), axis=1) / qmax
            scale = np.where(scale == 0, 1, scale).astype(weights.dtype)
            if np.any(scale == 0) or not np.isfinite(scale).all():
                raise ValueError("Per-channel scale cannot be represented in weight dtype")
            scales[start:end] = scale[:, None]
            values = np.full((end - start, block), zero, dtype=np.uint8)
            values[:, :k] = (np.clip(np.rint(columns / scale[:, None]), -qmax, qmax) + zero).astype(np.uint8)
            if bits == 4:
                packed[start:end] = (values[:, 0::2] | (values[:, 1::2] << 4)).reshape(end - start, 1, blob_size)
            else:
                packed[start:end] = values.reshape(end - start, 1, blob_size)
        return packed, scales, np.full((n, 1), fill, dtype=np.uint8)


def make_per_channel_quantizer(**kwargs):
    """Keep ORT graph traversal/Gather handling without global monkey patching."""
    if kwargs.get("bits") not in (4, 8) or not kwargs.get("is_symmetric", False):
        raise ValueError("Per-channel MatMul requires symmetric INT4 or INT8")
    if kwargs.get("algo_config") is not None or kwargs.get("quant_format") != q.QuantFormat.QOperator:
        raise ValueError("Per-channel MatMul requires default method and QOperator output")
    quantizer = q.MatMulNBitsQuantizer(**kwargs)
    quantizer.node_quantizer = PerChannelQuantizer(quantizer.algo_config)
    return quantizer
