# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
from pathlib import Path

import numpy as np
import torch

QUANTIZATION_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "quantization"
spec = importlib.util.spec_from_file_location("cuda_quantizer_int2_test_module", QUANTIZATION_DIR / "cuda_quantizer.py")
cuda_quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_quantizer)
CudaQuantizer = cuda_quantizer.CudaQuantizer


def test_int2_blockwise_quantization_matches_ort(monkeypatch):
    weights = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 0.5], [1.0, -1.0, 2.0, -2.0, 0.0, 0.5, -0.5, 3.0]],
        dtype=torch.float32,
    )
    expected_qweight = np.arange(4, dtype=np.uint8).reshape(2, 1, 2)
    expected_scales = np.array([[1.0], [2.0]], dtype=np.float32)

    def quantize_2bits(packed, weight, scales, zero_points, block_size, cols, rows, symmetric):
        assert weight.shape == (8, 2)
        assert (block_size, cols, rows, symmetric) == (8, 2, 8, True)
        packed[:] = expected_qweight
        scales[:] = expected_scales
        zero_points[:] = 0xAA

    monkeypatch.setattr(
        cuda_quantizer, "_get_quantize_matmul_nbits", lambda bits: quantize_2bits if bits == 2 else None
    )

    qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
        weights,
        bits=2,
        block_size=8,
        flatten_qweight=False,
        use_ort_quantizer=True,
    )

    assert torch.equal(qweight, torch.from_numpy(expected_qweight))
    assert torch.equal(scales, torch.from_numpy(expected_scales))
