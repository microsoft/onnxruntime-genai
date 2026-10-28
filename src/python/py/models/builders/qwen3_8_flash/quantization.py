# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bounded-memory checkpoint access and Qwen NVFP4-to-INT2 weight conversion."""

import json
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


class SafetensorsCheckpoint:
    """Keep shard mappings open without loading the full model into memory."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self._stack = ExitStack()
        self._handles = {}
        self._closed = False
        self._lock = threading.Lock()
        index = self.directory / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(self.weight_map, dict) or not self.weight_map:
                raise ValueError(f"Empty or invalid safetensors weight map: {index}")
            for name, shard in self.weight_map.items():
                if not isinstance(name, str) or not isinstance(shard, str):
                    raise ValueError(f"Invalid safetensors weight-map entry: {index}")
                self._shard_path(shard)
        else:
            shards = list(self.directory.glob("*.safetensors"))
            if len(shards) != 1:
                raise ValueError(f"Expected an index or exactly one safetensors shard: {self.directory}")
            handle = self._open(shards[0].name)
            self.weight_map = dict.fromkeys(handle.keys(), shards[0].name)

    def _shard_path(self, shard):
        path = (self.directory / shard).resolve(strict=True)
        if not path.is_relative_to(self.directory) or not path.is_file():
            raise ValueError(f"Safetensors shard escapes checkpoint directory: {shard}")
        return path

    def _open(self, shard):
        with self._lock:
            if self._closed:
                raise RuntimeError("Safetensors checkpoint is closed")
            if shard not in self._handles:
                self._handles[shard] = self._stack.enter_context(
                    safe_open(self._shard_path(shard), framework="pt", device="cpu")
                )
            return self._handles[shard]

    def tensor(self, name):
        if name not in self.weight_map:
            raise KeyError(f"Missing checkpoint tensor: {name}")
        return self._open(self.weight_map[name]).get_tensor(name)

    def close(self):
        self._stack.close()
        self._handles.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Safetensors checkpoint is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def dequantize_nvfp4(packed, block_scales, global_scale):
    """Match the target's two BF16 rounding steps before INT2 requantization."""
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] == 0:
        raise ValueError("NVFP4 weights must be nonempty packed UINT8 [N, K/2]")
    if packed.shape[0] == 0 or packed.shape[1] % 8:
        raise ValueError("NVFP4 weights require N > 0 and K divisible by 16")
    expected = (packed.shape[0], packed.shape[1] // 8)
    if block_scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(block_scales.shape) != expected:
        raise ValueError(f"NVFP4 block scales must be E4M3 or UINT8 with shape {expected}")
    if global_scale.numel() != 1 or not global_scale.is_floating_point():
        raise ValueError("NVFP4 global scale must be a floating-point scalar")
    global_value = global_scale.float()
    if not torch.isfinite(global_value).all() or not (global_value > 0).all():
        raise ValueError("NVFP4 global scale must be finite and positive")
    scales = block_scales.view(torch.float8_e4m3fn).float()
    if not torch.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    scales = (scales * global_value.reshape(())).to(torch.bfloat16).float()
    if not torch.isfinite(scales).all():
        raise ValueError("NVFP4 combined block scales overflow BF16")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    values = lookup[codes].reshape(packed.shape[0], -1, 16)
    weights = (values * scales.unsqueeze(-1)).to(torch.bfloat16).float()
    if not torch.isfinite(weights).all():
        raise ValueError("Dequantized NVFP4 weights overflow BF16")
    return weights.reshape(packed.shape[0], -1)


@dataclass(frozen=True)
class Int2Weights:
    packed: torch.Tensor
    scales: torch.Tensor
    squared_error: float
    squared_norm: float


def quantize_int2(weights):
    """Pack four {-2,-1,0,1} codes per byte; return signed BF16 group scales."""
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("INT2 conversion requires a floating-point [N, K] matrix")
    if weights.shape[0] == 0 or weights.shape[1] == 0 or weights.shape[1] % 64:
        raise ValueError("INT2 conversion requires N > 0 and K divisible by 64")
    values = weights.float()
    if not torch.isfinite(values).all():
        raise ValueError("INT2 conversion requires finite weights")
    grouped = values.reshape(weights.shape[0], -1, 64)
    candidates = []
    for sign in (1, -1):
        scale = grouped.abs().amax(dim=-1).clamp_min(1e-20) / 2 * sign
        for _ in range(6):
            code = torch.round(grouped / scale.unsqueeze(-1)).clamp(-2, 1)
            scale = ((grouped * code).sum(-1) / code.square().sum(-1).clamp_min(1)).to(torch.bfloat16).float()
            scale = torch.where(scale.abs() > 1e-20, scale, torch.full_like(scale, 1e-20 * sign))
        code = torch.round(grouped / scale.unsqueeze(-1)).clamp(-2, 1)
        error = (grouped - code * scale.unsqueeze(-1)).square().sum(-1)
        candidates.append((scale, code, error))
    choose_negative = candidates[1][2] < candidates[0][2]
    scale = torch.where(choose_negative, candidates[1][0], candidates[0][0])
    code = torch.where(choose_negative.unsqueeze(-1), candidates[1][1], candidates[0][1])
    decoded = code * scale.unsqueeze(-1)
    stored = (code.reshape(*weights.shape).to(torch.int16) + 2).to(torch.uint8)
    stored = stored.reshape(weights.shape[0], -1, 4)
    packed = stored[..., 0] | (stored[..., 1] << 2) | (stored[..., 2] << 4) | (stored[..., 3] << 6)
    return Int2Weights(
        packed.contiguous(),
        scale.to(torch.bfloat16),
        float((decoded - grouped).square().sum().item()),
        float(grouped.square().sum().item()),
    )


def prepack_int2(packed):
    """Apply ORT's SM80 W2_A16 layout, also used by the Hopper INT2 path."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("INT2 prepacking requires UINT8 [N, K/4]")
    n, packed_k = packed.shape
    if n == 0 or n % 8 or packed_k == 0 or packed_k % 16:
        raise ValueError("INT2 prepacking requires N divisible by 8 and K divisible by 64")
    k = packed_k * 4
    codes = torch.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1).reshape(n, k)
    permutation = torch.arange(64, device=packed.device).reshape(8, 4, 2).permute(1, 0, 2).reshape(64)
    codes = codes.reshape(n, k // 64, 64)[..., permutation]
    codes = codes.reshape(n // 8, 8, k // 64, 64).permute(0, 2, 1, 3).contiguous()
    # The input is already biased by +2; this commutes with the CUDA layout
    # transforms, so only the final even/odd pair permutation remains.
    pairs = torch.arange(16, device=packed.device).reshape(8, 2).T.reshape(16)
    codes = codes.reshape(-1, 16)[:, pairs].reshape(-1, 4)
    output = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return output.reshape_as(packed).contiguous()


def quantize_expert(checkpoint, layer, expert, projection):
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(f"Unsupported expert projection: {projection}")
    if not 0 <= layer < 48 or not 0 <= expert < 512:
        raise ValueError(f"Unsupported Qwen expert index: layer={layer}, expert={expert}")
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
    weights = dequantize_nvfp4(
        checkpoint.tensor(prefix + ".weight"),
        checkpoint.tensor(prefix + ".weight_scale"),
        checkpoint.tensor(prefix + ".weight_scale_2"),
    )
    expected = (2560, 640) if projection == "down_proj" else (640, 2560)
    if tuple(weights.shape) != expected:
        raise ValueError(f"Unexpected Qwen NVFP4 shape at {prefix}: expected {expected}, got {tuple(weights.shape)}")
    return quantize_int2(weights)
