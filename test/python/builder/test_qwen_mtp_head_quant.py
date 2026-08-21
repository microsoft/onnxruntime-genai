# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""The MTP model resolves quantization independently from the main model."""

import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from models.builders.qwen import Qwen35MoETextModel, Qwen35TextModel
from models.builders.qwen_mtp import Qwen35MtpHead
from quantization import QuantConfig


def _resolve(extra_options, main_onnx_dtype=ir.DataType.INT4):
    """Run the MTP model config resolution on a bare stub of the builder."""
    model = object.__new__(Qwen35MoETextModel)
    model._mtp_onnx_dtype = main_onnx_dtype
    model._mtp_io_dtype = ir.DataType.FLOAT16
    model._mtp_extra_options = dict(extra_options)
    model._mtp_ep = "cuda"
    model._resolve_mtp_model_config(extra_options)
    return model


def test_no_mtp_config_inherits_main_model_settings():
    options = {"enable_mtp": True, "moe_quant_type": "nvfp4", "block_size": 64}
    model = _resolve(options)

    assert model._mtp_onnx_dtype == ir.DataType.INT4
    assert model._mtp_extra_options == options


def test_mtp_quant_config_json_configures_targets_independently():
    model = _resolve(
        {
            "mtp_quant_config": json.dumps(
                {
                    "io_dtype": "bf16",
                    "weights": {"type": "int4", "symmetric": False},
                    "moe": {"type": "none"},
                }
            )
        },
        main_onnx_dtype=ir.DataType.INT8,
    )

    assert model._mtp_io_dtype == ir.DataType.BFLOAT16
    assert model._mtp_onnx_dtype == ir.DataType.UINT4
    quant_config = model._mtp_extra_options["_quant_config"]
    assert quant_config.weights.type == "int4"
    assert quant_config.moe.type == "none"


def test_mtp_quant_config_can_keep_the_entire_head_fp16():
    model = _resolve(
        {
            "mtp_quant_config": json.dumps(
                {
                    "io_dtype": "fp16",
                    "weights": {"type": "none"},
                    "moe": {"type": "none"},
                }
            )
        }
    )

    assert model._mtp_io_dtype == ir.DataType.FLOAT16
    assert model._mtp_onnx_dtype == ir.DataType.FLOAT16
    quant_config = model._mtp_extra_options["_quant_config"]
    assert quant_config.weights.type == "none"
    assert quant_config.moe.type == "none"


def test_mtp_dense_fp4_requires_a_supported_dense_format():
    with pytest.raises(ValueError, match="select mxfp4/nvfp4 independently through moe.type"):
        _resolve({"mtp_quant_config": '{"weights": {"type": "nvfp4", "block_size": 16}}'})


def test_modelopt_defaults_to_native_mtp_but_explicit_type_requantizes():
    assert Qwen35MtpHead._should_preserve_modelopt_mtp("modelopt", {})
    assert not Qwen35MtpHead._should_preserve_modelopt_mtp(
        "modelopt",
        {"_quant_config": QuantConfig.from_dict({})},
    )
    assert not Qwen35MtpHead._should_preserve_modelopt_mtp(None, {})


def test_explicit_override_dequantizes_modelopt_nvfp4_weight():
    packed = torch.full((2, 8), 0x21, dtype=torch.uint8)
    block_scale = torch.ones((2, 1), dtype=torch.float8_e4m3fn)

    actual = Qwen35MtpHead._dequantize_modelopt_weight(
        packed,
        block_scale,
        torch.tensor(0.5),
        "mtp.fc.weight",
    )

    expected_row = torch.tensor([0.25, 0.5] * 8, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected_row.repeat(2, 1), rtol=0, atol=0)


def test_modelopt_mtp_loader_preserves_native_tensor_formats():
    head = object.__new__(Qwen35MtpHead)
    head.moe_attrs = {"num_experts": 1}
    state = {}

    def add_bf16(prefix):
        state[f"{prefix}.weight"] = torch.ones((2, 2), dtype=torch.bfloat16)

    def add_fp8(prefix):
        state[f"{prefix}.weight"] = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        state[f"{prefix}.weight_scale"] = torch.tensor(0.25)

    def add_nvfp4(prefix):
        state[f"{prefix}.weight"] = torch.zeros((2, 8), dtype=torch.uint8)
        state[f"{prefix}.weight_scale"] = torch.zeros((2, 1), dtype=torch.float8_e4m3fn)
        state[f"{prefix}.weight_scale_2"] = torch.tensor(0.5)

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
    add_nvfp4("lm_head")
    for name in ("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm"):
        add_bf16(name)

    embedding = torch.ones((2, 2), dtype=torch.bfloat16)
    head._load_native_modelopt_mtp_weights(state, embedding)

    assert head._mtp_layer.self_attn.q_proj.weight.dtype == torch.float8_e4m3fn
    assert head._mtp_layer.self_attn.q_proj.weight_scale.item() == 0.25
    assert head._mtp_layer.mlp.experts[0].gate_proj.weight.dtype == torch.uint8
    assert head._mtp_layer.mlp.experts[0].gate_proj.weight_scale_2.item() == 0.5
    assert head._fc.weight_scale_2.item() == 0.5
    assert head._lm_head.weight_scale_2.item() == 0.5
