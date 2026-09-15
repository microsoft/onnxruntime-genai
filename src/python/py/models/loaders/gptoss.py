# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------

import json
from pathlib import Path

import onnx_ir as ir
import torch
from safetensors import safe_open

from .base import QuantizedExperts


class GptOssMXFP4Loader:
    """Load original GPT-OSS MXFP4 expert tensors into QMoE's packed layout."""

    def __init__(self, model_name_or_path, cache_dir=None, token=None):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.token = token
        self._snapshot_dir = None
        self._weight_map = None
        self._weight_map_loaded = False

    def get_snapshot_dir(self):
        if self._snapshot_dir is not None:
            return self._snapshot_dir

        model_path = Path(self.model_name_or_path)
        if model_path.is_dir():
            self._snapshot_dir = model_path
            return self._snapshot_dir

        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to locate original GPT-OSS MXFP4 weights.") from exc

        self._snapshot_dir = Path(
            snapshot_download(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                token=self.token,
                local_files_only=True,
            )
        )
        return self._snapshot_dir

    def get_weight_map(self):
        if self._weight_map_loaded:
            return self._weight_map

        snapshot_dir = self.get_snapshot_dir()
        index_path = snapshot_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as index_file:
                self._weight_map = json.load(index_file)["weight_map"]
        else:
            safetensors_files = sorted(snapshot_dir.glob("*.safetensors"))
            if len(safetensors_files) != 1:
                raise RuntimeError(f"Could not locate original GPT-OSS MXFP4 safetensors index in {snapshot_dir}.")
            self._weight_map = None
        self._weight_map_loaded = True
        return self._weight_map

    def load_tensor(self, tensor_name):
        snapshot_dir = self.get_snapshot_dir()
        weight_map = self.get_weight_map()
        if weight_map is None:
            candidate_files = sorted(snapshot_dir.glob("*.safetensors"))
        else:
            if tensor_name not in weight_map:
                raise RuntimeError(f"Original GPT-OSS MXFP4 tensor '{tensor_name}' was not found in safetensors index.")
            candidate_files = [snapshot_dir / weight_map[tensor_name]]

        for tensor_file in candidate_files:
            with safe_open(tensor_file, framework="pt", device="cpu") as safetensors_file:
                if tensor_name in safetensors_file.keys():  # noqa: SIM118
                    return safetensors_file.get_tensor(tensor_name)

        raise RuntimeError(f"Original GPT-OSS MXFP4 tensor '{tensor_name}' was not found in {snapshot_dir}.")

    def pack_blocks_for_qmoe(self, blocks):
        """Repack ``[E,N,K/32,16]`` checkpoint blocks to QMoE ``[E,K,N/2]``."""
        if blocks.dtype != torch.uint8:
            blocks = blocks.to(torch.uint8)
        if blocks.ndim != 4 or blocks.shape[-1] != 16:
            raise ValueError(f"GPT-OSS MXFP4 blocks must have shape [E, N, K/32, 16], got {tuple(blocks.shape)}.")
        if blocks.shape[1] % 2 != 0:
            raise ValueError(f"GPT-OSS MXFP4 output dimension N must be even, got {blocks.shape[1]}.")

        even_n = blocks[:, 0::2, :, :]
        odd_n = blocks[:, 1::2, :, :]
        packed_even_k = ((odd_n & 0x0F) << 4) | (even_n & 0x0F)
        packed_odd_k = ((odd_n >> 4) << 4) | (even_n >> 4)
        packed = torch.stack((packed_even_k, packed_odd_k), dim=-1)
        return (
            packed.permute(0, 2, 3, 4, 1)
            .reshape(blocks.shape[0], blocks.shape[2] * 32, blocks.shape[1] // 2)
            .contiguous()
        )

    def prepare_experts(self, layer_id):
        prefix = f"model.layers.{layer_id}.moe.experts"
        gate_up_blocks = self.load_tensor(f"{prefix}.gate_up_proj_blocks")
        gate_up_scales = self.load_tensor(f"{prefix}.gate_up_proj_scales")
        down_blocks = self.load_tensor(f"{prefix}.down_proj_blocks")
        down_scales = self.load_tensor(f"{prefix}.down_proj_scales")

        for projection, blocks, scales in (
            ("gate_up_proj", gate_up_blocks, gate_up_scales),
            ("down_proj", down_blocks, down_scales),
        ):
            expected_scale_shape = blocks.shape[:-1]
            if tuple(scales.shape) != tuple(expected_scale_shape):
                raise ValueError(
                    f"GPT-OSS MXFP4 scales for layer {layer_id} {projection} must have shape "
                    f"{tuple(expected_scale_shape)}, got {tuple(scales.shape)}."
                )

        experts = QuantizedExperts()
        experts.quant_type = "fp4"
        experts.block_size = 32
        experts.scale_dtype = ir.DataType.FLOAT8E8M0
        experts.scales_raw = True
        experts.gate_up_qweight = self.pack_blocks_for_qmoe(gate_up_blocks)
        experts.gate_up_scales = gate_up_scales.to(torch.uint8).contiguous()
        experts.gate_up_global_scales = torch.ones(gate_up_blocks.shape[0], dtype=torch.float32)
        experts.down_qweight = self.pack_blocks_for_qmoe(down_blocks)
        experts.down_scales = down_scales.to(torch.uint8).contiguous()
        experts.down_global_scales = torch.ones(down_blocks.shape[0], dtype=torch.float32)
        return experts
