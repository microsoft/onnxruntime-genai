# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json

import onnx_ir as ir
import torch
from safetensors.torch import save_file

from loaders.gptoss import GptOssMXFP4Loader


def _projection_tensors(layer_id=0):
    prefix = f"model.layers.{layer_id}.moe.experts"
    gate_up_blocks = torch.arange(2 * 4 * 2 * 16, dtype=torch.uint8).reshape(2, 4, 2, 16)
    down_blocks = torch.arange(2 * 6 * 2 * 16, dtype=torch.uint8).reshape(2, 6, 2, 16)
    return {
        f"{prefix}.gate_up_proj_blocks": gate_up_blocks,
        f"{prefix}.gate_up_proj_scales": torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2),
        f"{prefix}.down_proj_blocks": down_blocks,
        f"{prefix}.down_proj_scales": torch.arange(24, dtype=torch.uint8).reshape(2, 6, 2),
    }


def test_pack_blocks_for_qmoe_preserves_fp4_codes():
    blocks = _projection_tensors()["model.layers.0.moe.experts.gate_up_proj_blocks"]
    packed = GptOssMXFP4Loader.pack_blocks_for_qmoe(blocks)

    codes = torch.empty(2, 4, 2, 32, dtype=torch.uint8)
    codes[..., 0::2] = blocks & 0x0F
    codes[..., 1::2] = blocks >> 4
    codes_kn = codes.reshape(2, 4, 64).permute(0, 2, 1).contiguous()
    expected = (codes_kn[..., 1::2] << 4) | codes_kn[..., 0::2]

    assert packed.shape == (2, 64, 2)
    assert torch.equal(packed, expected)


def test_prepare_experts_preserves_mxfp4_scale_bytes(tmp_path):
    tensors = _projection_tensors()
    save_file(tensors, tmp_path / "model.safetensors")

    experts = GptOssMXFP4Loader(tmp_path).prepare_experts(0)

    assert experts.quant_type == "fp4"
    assert experts.block_size == 32
    assert experts.scale_dtype == ir.DataType.FLOAT8E8M0
    assert experts.scales_raw
    assert torch.equal(
        experts.gate_up_scales,
        tensors["model.layers.0.moe.experts.gate_up_proj_scales"],
    )
    assert torch.equal(experts.gate_up_global_scales, torch.ones(2))
    assert experts.gate_up_qweight.shape == (2, 64, 2)
    assert experts.down_qweight.shape == (2, 64, 3)


def test_indexed_checkpoint_reads_each_projection_from_its_shard(tmp_path):
    tensors = _projection_tensors()
    gate_tensors = {name: tensor for name, tensor in tensors.items() if "gate_up" in name}
    down_tensors = {name: tensor for name, tensor in tensors.items() if "down_proj" in name}
    save_file(gate_tensors, tmp_path / "gate.safetensors")
    save_file(down_tensors, tmp_path / "down.safetensors")
    weight_map = {
        name: "gate.safetensors" if "gate_up" in name else "down.safetensors" for name in tensors
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    experts = GptOssMXFP4Loader(tmp_path).prepare_experts(0)

    assert experts.gate_up_qweight.shape == (2, 64, 2)
    assert experts.down_qweight.shape == (2, 64, 3)
