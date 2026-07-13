# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Numerical guard for the NVFP4 model-builder repack helpers.

Model Optimizer stores each NVFP4 expert projection as:
  weight        uint8   [N, K/2]  (E2M1, 2 codes/byte, packed along K; low nibble = even K)
  weight_scale  e4m3    [N, K/16] (one FP8 block scale per 16 K-elements)
  weight_scale_2 f32    scalar    (per-tensor global scale)
and the value is reconstructed as
  w[n, k] = e2m1(code) * e4m3(weight_scale[n, k // 16]) * weight_scale_2.

The ORT CUDA QMoE ``nvfp4`` kernel reads weights as ``[E, K, N/2]`` (N-packed,
even N = low nibble), block scales as ``[E, N, K/16]`` e4m3, and per-expert
float32 global scales, dequantizing as
  w[n, k] = e2m1(code) * e4m3(block_scale[n, k // 16]) * global[e].

This test proves the builder's repack (K-unpack -> N-pack) preserves the exact
code<->(N, K) mapping, so the kernel reconstructs Model Optimizer's weights
bit-for-bit (no re-quantization).
"""

import importlib.util
import os
import sys
import types
from unittest import mock

import numpy as np
import torch


def _load_builder_model_class():
    """Load ``Model`` from the source base.py, stubbing heavy optional imports.

    The two repack helpers under test are pure-torch static methods, so we do not
    need transformers/onnxruntime installed just to exercise them.
    """
    base_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "python", "py", "models", "builders", "base.py"
    )
    base_path = os.path.abspath(base_path)
    builders_dir = os.path.dirname(base_path)
    models_dir = os.path.dirname(builders_dir)
    for name in (
        "transformers",
        "tqdm",
        "peft",
        "onnxruntime",
        "onnxruntime.quantization",
        "onnxruntime.quantization.matmul_nbits_quantizer",
    ):
        sys.modules.setdefault(name, mock.MagicMock())
    # Synthetic parent packages so base.py's `from .cuda_quantizer import ...` resolves
    # without executing the real builders __init__ (which imports every builder).
    pkg_models = types.ModuleType("_genai_models_pkg")
    pkg_models.__path__ = [models_dir]
    pkg_builders = types.ModuleType("_genai_models_pkg.builders")
    pkg_builders.__path__ = [builders_dir]
    sys.modules["_genai_models_pkg"] = pkg_models
    sys.modules["_genai_models_pkg.builders"] = pkg_builders
    sys.modules["_genai_models_pkg.builders.cuda_quantizer"] = mock.MagicMock()
    spec = importlib.util.spec_from_file_location("_genai_models_pkg.builders.base", base_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_genai_models_pkg.builders.base"] = module
    spec.loader.exec_module(module)
    return module.Model


Model = _load_builder_model_class()

_FP4_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _decode_e2m1(code):
    code = int(code) & 0xF
    v = _FP4_E2M1[code & 0x7]
    return -v if (code & 0x8) else v


def _e4m3_byte_to_float(byte):
    return torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()


def _make_modelopt_nvfp4_projection(n, k, seed):
    """Build a random Model-Optimizer-form NVFP4 projection and its reconstruction."""
    rng = np.random.default_rng(seed)
    block = 16
    codes_nk = rng.integers(0, 16, size=(n, k), dtype=np.uint8)  # e2m1 codes [N, K]
    # Random e4m3 block scales (avoid NaN encodings 0x7F / 0xFF).
    scale_bytes = rng.integers(0, 0x7F, size=(n, k // block), dtype=np.uint8)  # [N, K/16]
    global_scale = np.float32(rng.uniform(0.05, 0.5))

    # Model Optimizer weight layout: [N, K/2], low nibble = even K, high = odd K.
    low = codes_nk[:, 0::2]
    high = codes_nk[:, 1::2]
    packed_nk2 = ((high << 4) | low).astype(np.uint8)  # [N, K/2]

    # Reference reconstruction from the modelopt-form tensors.
    ref = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            ref[i, j] = _decode_e2m1(codes_nk[i, j]) * _e4m3_byte_to_float(scale_bytes[i, j // block]) * float(global_scale)
    return packed_nk2, scale_bytes, global_scale, ref


def _kernel_dequant_from_qmoe_layout(packed_kn2, scale_nk16, global_scale, n, k):
    """Mirror the ORT QMoE nvfp4 dequant kernel over a [K, N/2]-packed weight."""
    block = 16
    out = np.empty((n, k), dtype=np.float32)
    packed_kn2 = packed_kn2.numpy() if isinstance(packed_kn2, torch.Tensor) else packed_kn2
    for row_n in range(n):
        for col_k in range(k):
            byte = packed_kn2[col_k, row_n // 2]
            code = (byte & 0x0F) if (row_n % 2 == 0) else (byte >> 4)
            out[row_n, col_k] = (
                _decode_e2m1(code)
                * _e4m3_byte_to_float(scale_nk16[row_n, col_k // block])
                * float(global_scale)
            )
    return out


def test_nvfp4_repack_roundtrip_matches_modelopt():
    for seed, (n, k) in enumerate([(8, 64), (16, 128), (32, 512), (2048, 512)]):
        packed_nk2, scale_bytes, global_scale, ref = _make_modelopt_nvfp4_projection(n, k, seed)

        # Builder repack: K-unpack -> per-element codes [N, K] -> N-pack [K, N/2].
        codes_nk = Model.repack_modelopt_nvfp4_weight_codes(torch.from_numpy(packed_nk2))
        assert tuple(codes_nk.shape) == (n, k)
        qweight_kn2 = Model.pack_nvfp4_codes_for_qmoe(codes_nk)
        assert tuple(qweight_kn2.shape) == (k, n // 2)

        # Block scales are unchanged ([N, K/16]); global scale passed through.
        got = _kernel_dequant_from_qmoe_layout(qweight_kn2, scale_bytes, global_scale, n, k)

        max_abs = float(np.max(np.abs(got - ref)))
        assert max_abs == 0.0, f"shape ({n},{k}): repacked dequant differs from modelopt ref (max_abs={max_abs})"


if __name__ == "__main__":
    test_nvfp4_repack_roundtrip_matches_modelopt()
    print("OK: NVFP4 repack round-trip is bit-exact against the Model Optimizer reconstruction.")
