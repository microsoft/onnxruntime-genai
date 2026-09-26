# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

QUANTIZATION_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "quantization"
spec = importlib.util.spec_from_file_location("cuda_quantizer_int2_test_module", QUANTIZATION_DIR / "cuda_quantizer.py")
cuda_quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_quantizer)
CudaQuantizer = cuda_quantizer.CudaQuantizer


def test_int2_blockwise_quantization_matches_ort():
    weights = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -2.5, 0.5] * 4, [1.0, -1.0, 2.0, -2.0, 0.0, 0.5, -0.5, 3.0] * 4],
        dtype=torch.float32,
    )
    ort_pybind = pytest.importorskip("onnxruntime.capi._pybind_state")
    quantize_2bits = getattr(ort_pybind, "quantize_matmul_2bits", None)
    if quantize_2bits is None:
        pytest.skip("ORT build does not expose quantize_matmul_2bits")

    expected_qweight = np.zeros((2, 1, 8), dtype=np.uint8)
    expected_scales = np.zeros((2, 1), dtype=np.float32)
    zero_points = np.zeros((2, 1), dtype=np.uint8)
    quantize_2bits(
        expected_qweight, np.ascontiguousarray(weights.numpy().T), expected_scales, zero_points, 32, 2, 32, True
    )

    np.testing.assert_array_equal(expected_qweight, np.array([[[111, 177] * 4], [[221, 42] * 4]], dtype=np.uint8))
    np.testing.assert_array_equal(expected_scales, [[-1.5], [-1.5]])

    qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
        weights,
        bits=2,
        block_size=32,
        flatten_qweight=False,
    )

    assert torch.equal(qweight, torch.from_numpy(expected_qweight))
    assert torch.equal(scales, torch.from_numpy(expected_scales))
