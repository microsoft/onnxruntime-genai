# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import MethodType

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

from loaders.modelopt import ModeloptModel


def _make_model(tensors, quant_type="modelopt"):
    model = object.__new__(ModeloptModel)
    model.weight_map = tensors
    model.get_tensor = MethodType(lambda self, key: tensors.get(key), model)
    model.quant_type = quant_type
    return model


LINEAR = "model.language_model.layers.3.linear_attn.in_proj_qkv"
ATTENTION = "model.language_model.layers.3.self_attn.q_proj"


def test_linear_attention_fp8_is_weight_only():
    tensors = {
        f"{LINEAR}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{LINEAR}.weight_scale": torch.tensor(0.125),
        f"{LINEAR}.input_scale": torch.tensor(0.25),
    }
    module = _make_model(tensors).make_linear_module(LINEAR)

    assert module.quant_type == "fp8"
    assert module.weight.dtype == torch.float8_e4m3fn
    assert module.weight_scale.item() == 0.125
    assert module.input_scale is None


def test_self_attention_fp8_keeps_calibrated_input_scale():
    tensors = {
        f"{ATTENTION}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{ATTENTION}.weight_scale": torch.tensor(0.125),
        f"{ATTENTION}.input_scale": torch.tensor(0.25),
    }
    module = _make_model(tensors).make_linear_module(ATTENTION)

    assert module.input_scale.item() == 0.25


def test_modelopt_fp8_rejects_per_channel_weight_scale():
    tensors = {
        f"{ATTENTION}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{ATTENTION}.weight_scale": torch.ones(4),
    }

    with pytest.raises(ValueError, match="must be a scalar"):
        _make_model(tensors).make_linear_module(ATTENTION)


def test_compressed_tensors_fp8_accepts_per_channel_weight_scale():
    tensors = {
        f"{ATTENTION}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{ATTENTION}.weight_scale": torch.ones(4),
    }

    module = _make_model(tensors, quant_type="compressed-tensors").make_linear_module(ATTENTION)

    assert module.weight_scale.shape == (4,)


def test_compressed_tensors_nvfp4_normalizes_packed_weight_and_global_scale():
    tensors = {
        f"{ATTENTION}.weight_packed": torch.zeros((4, 8), dtype=torch.uint8),
        f"{ATTENTION}.weight_scale": torch.ones((4, 1), dtype=torch.float8_e4m3fn),
        f"{ATTENTION}.weight_global_scale": torch.tensor(2.0),
    }

    module = _make_model(tensors, quant_type="compressed-tensors").make_linear_module(ATTENTION)

    assert module.quant_type == "nvfp4"
    assert module.weight is tensors[f"{ATTENTION}.weight_packed"]
    assert module.weight_scale_2.item() == 0.5


def test_compressed_tensors_dense_mlp_uses_packed_weights():
    layer = "model.language_model.layers.0"
    tensors = {
        f"{layer}.self_attn.q_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{layer}.mlp.{projection}"
        tensors[f"{base}.weight_packed"] = torch.zeros((4, 8), dtype=torch.uint8)
        tensors[f"{base}.weight_scale"] = torch.ones((4, 1), dtype=torch.float8_e4m3fn)
        tensors[f"{base}.weight_global_scale"] = torch.tensor(2.0)

    model = _make_model(tensors, quant_type="compressed-tensors")
    model.num_experts = None

    mlp = model.make_layer(0).mlp

    assert mlp.gate_proj.quant_type == "nvfp4"
    assert mlp.up_proj.weight_scale_2.item() == 0.5
    assert mlp.down_proj.weight is tensors[f"{layer}.mlp.down_proj.weight_packed"]


def test_bf16_linear_attention_projection_stays_unquantized():
    base = "model.language_model.layers.3.linear_attn.in_proj_a"
    weight = torch.ones((4, 4), dtype=torch.bfloat16)
    module = _make_model({f"{base}.weight": weight}).make_linear_module(base)

    assert module.quant_type == "none"
    assert module.weight is weight
    assert module.weight_scale is None


def test_optional_mtp_tensors_are_parsed_into_generic_modules():
    tensors = {
        "mtp.fc.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(2, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(2, dtype=torch.bfloat16),
        "mtp.norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    model = _make_model(tensors)
    layer = object()
    model.make_layer = MethodType(lambda self, layer_id, prefix=None: layer, model)

    mtp = model.make_mtp()

    assert mtp.fc.weight is tensors["mtp.fc.weight"]
    assert mtp.pre_fc_norm_embedding.weight is tensors["mtp.pre_fc_norm_embedding.weight"]
    assert mtp.pre_fc_norm_hidden.weight is tensors["mtp.pre_fc_norm_hidden.weight"]
    assert mtp.norm.weight is tensors["mtp.norm.weight"]
    assert mtp.layers == [layer]


def test_modelopt_nvfp4_dequantization_stays_in_parser():
    packed = torch.full((2, 8), 0x21, dtype=torch.uint8)
    block_scale = torch.ones((2, 1), dtype=torch.float8_e4m3fn)
    model = _make_model({})

    actual = model.dequantize_tensor(
        packed,
        block_scale,
        torch.tensor(0.5),
        "mtp.fc.weight",
    )

    expected_row = torch.tensor([0.25, 0.5] * 8, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected_row.repeat(2, 1), rtol=0, atol=0)


def test_modelopt_mtp_layer_preserves_native_quantized_modules():
    tensors = {}

    def add_bf16(prefix):
        tensors[f"{prefix}.weight"] = torch.ones((2, 2), dtype=torch.bfloat16)

    def add_fp8(prefix):
        tensors[f"{prefix}.weight"] = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        tensors[f"{prefix}.weight_scale"] = torch.tensor(0.25)

    def add_nvfp4(prefix):
        tensors[f"{prefix}.weight"] = torch.zeros((2, 8), dtype=torch.uint8)
        tensors[f"{prefix}.weight_scale"] = torch.zeros((2, 1), dtype=torch.float8_e4m3fn)
        tensors[f"{prefix}.weight_scale_2"] = torch.tensor(0.5)

    layer = "mtp.layers.0"
    for name in (
        f"{layer}.input_layernorm",
        f"{layer}.post_attention_layernorm",
        f"{layer}.self_attn.q_norm",
        f"{layer}.self_attn.k_norm",
        f"{layer}.mlp.gate",
        f"{layer}.mlp.shared_expert_gate",
    ):
        add_bf16(name)
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        add_fp8(f"{layer}.self_attn.{projection}")
    for base in (f"{layer}.mlp.shared_expert", f"{layer}.mlp.experts.0"):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            add_nvfp4(f"{base}.{projection}")
    add_nvfp4("mtp.fc")
    for name in ("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm"):
        add_bf16(name)

    model = _make_model(tensors)
    model.num_experts = 1
    mtp = model.make_mtp()

    assert mtp.layers[0].self_attn.q_proj.weight.dtype == torch.float8_e4m3fn
    assert mtp.layers[0].self_attn.q_proj.weight_scale.item() == 0.25
    assert mtp.layers[0].mlp.experts.quant_type == "nvfp4"
    assert mtp.layers[0].mlp.experts.gate_up_qweight.shape[0] == 1
    assert mtp.fc.weight_scale_2.item() == 0.5
