from __future__ import annotations

import numpy as np
import torch


# TODO: Keep this helper in sync with the ORT-side QMoE quantization test helper.
# Longer term, expose the shared implementation from onnxruntime quantization
# tools so onnxruntime-genai can import it instead of carrying a copy.
def symmetric_per_channel_quantize(
    weights: torch.Tensor,
    bits: int,
    *,
    trtllm_signed_storage: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one QMoE expert with symmetric per-channel storage.

    ``weights`` has logical shape ``[N, K]``. Returns raw QMoE storage
    ``[N, K/pack]`` and scales ``[N]``. By default, this emits the ORT CUDA
    QMoE storage contract: unsigned bytes/nibbles with an implicit zero-point
    offset, so each stored value is ``q + zero_point`` even though the numeric
    quantization is symmetric. Set ``trtllm_signed_storage=True`` for the
    TRT-LLM-style signed two's-complement storage using ``[-8, 7]`` / ``[-128,
    127]``.
    """
    weights = weights.detach().cpu().to(torch.float32).contiguous()
    bits = int(bits)
    n, k = weights.shape
    pack = 8 // bits
    if k % pack != 0:
        raise ValueError(f"K ({k}) must be divisible by {pack} for QMoE per-channel quantization.")

    if bits == 4:
        if trtllm_signed_storage:
            qmin, qmax, scale_divisor, zero_point = -8, 7, 8, None
        else:
            qmin, qmax, scale_divisor, zero_point = -7, 7, 7, 8
    elif bits == 8:
        if trtllm_signed_storage:
            qmin, qmax, scale_divisor, zero_point = -128, 127, 128, None
        else:
            qmin, qmax, scale_divisor, zero_point = -127, 127, 127, 128
    else:
        raise ValueError(f"QMoE per-channel quantization only supports 4 or 8 bits, got {bits}.")

    scales = weights.abs().amax(dim=1, keepdim=True) / float(scale_divisor)
    scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
    quantized = torch.clamp(torch.round(weights / scales), qmin, qmax).to(torch.int16).contiguous()
    if zero_point is None:
        quantized = quantized.to(torch.uint8)
    else:
        quantized = (quantized + zero_point).to(torch.uint8)

    if bits == 4:
        qweight = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)
        qweight = qweight.to(torch.uint8)
    else:
        qweight = quantized

    return qweight.contiguous(), scales.squeeze(-1).contiguous()


def cuda_per_channel_quantize(
    weights: torch.Tensor,
    bits: int,
    prepack: bool,
    pack_weights_for_cuda_mixed_gemm,
    force_arch: int = 80,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize per-channel QMoE weights and optionally CUTLASS-prepack them."""
    qweight, scales = symmetric_per_channel_quantize(weights, bits)
    if not prepack:
        return qweight, scales

    n, k = weights.shape
    pack = 8 // int(bits)
    if n % pack != 0:
        raise ValueError(f"N ({n}) must be divisible by {pack} for CUDA QMoE prepacked weights.")

    packed = pack_weights_for_cuda_mixed_gemm(qweight.numpy(), n, k, int(bits), force_arch)
    packed = np.asarray(packed).view(np.uint8).reshape(k, n // pack)
    return torch.from_numpy(np.ascontiguousarray(packed)), scales
