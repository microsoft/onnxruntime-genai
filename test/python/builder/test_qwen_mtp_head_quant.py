# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""`mtp_head_quant_type` must reach the head's dense MatMuls, not just its QMoE experts."""

import onnx_ir as ir
import pytest
import torch

from models.builders.qwen import Qwen35MoeTextModel, Qwen35MtpHead


def _resolve(extra_options, main_onnx_dtype=ir.DataType.INT4):
    """Run the MTP-head precision resolution on a bare stub of the builder."""
    model = object.__new__(Qwen35MoeTextModel)
    model._mtp_onnx_dtype = main_onnx_dtype
    model._mtp_io_dtype = ir.DataType.FLOAT16
    model._mtp_extra_options = dict(extra_options)
    model._resolve_mtp_head_quantization(extra_options)
    return model


@pytest.mark.parametrize(
    "quant_type,expected_onnx_dtype",
    [
        ("int4", ir.DataType.INT4),
        ("int8", ir.DataType.INT8),
        # Microscaling FP4 is QMoE-only; the head's dense MatMuls stay int4.
        ("mxfp4", ir.DataType.INT4),
        ("nvfp4", ir.DataType.INT4),
    ],
)
def test_head_quant_type_sets_dense_weight_dtype(quant_type, expected_onnx_dtype):
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": quant_type})

    assert model._mtp_onnx_dtype == expected_onnx_dtype
    assert model._mtp_io_dtype == ir.DataType.FLOAT16
    assert model._mtp_extra_options["moe_quant_type"] == quant_type


def test_int8_head_is_not_downgraded_by_an_nvfp4_main_model():
    # The main model quantizes its experts to NVFP4; the head must still go fully int8.
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "int8", "moe_quant_type": "nvfp4"})

    assert model._mtp_onnx_dtype == ir.DataType.INT8
    assert model._mtp_extra_options["moe_quant_type"] == "int8"


def test_asymmetric_head_uses_unsigned_weight_dtype():
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "int8", "int4_is_symmetric": False})

    assert model._mtp_onnx_dtype == ir.DataType.UINT8


def test_float_main_model_gets_int4_defaults_for_a_fp4_head():
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "nvfp4"}, main_onnx_dtype=ir.DataType.FLOAT16)

    assert model._mtp_onnx_dtype == ir.DataType.INT4
    assert model._mtp_extra_options["block_size"] == 32
    assert model._mtp_extra_options["algo_config"] == "rtn_last"


@pytest.mark.parametrize("quant_type", ["mxfp4", "nvfp4"])
def test_fp4_head_uses_int4_dense_weights_with_int8_main(quant_type):
    model = _resolve(
        {"enable_mtp": True, "mtp_head_quant_type": quant_type},
        main_onnx_dtype=ir.DataType.INT8,
    )

    assert model._mtp_onnx_dtype == ir.DataType.INT4
    assert model._mtp_extra_options["moe_quant_type"] == quant_type


def test_quantized_main_model_does_not_get_int4_defaults():
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "int8"})

    assert "int4_algo_config" not in model._mtp_extra_options


def test_unknown_head_quant_type_is_rejected():
    with pytest.raises(ValueError, match="mtp_head_quant_type must be one of"):
        _resolve({"enable_mtp": True, "mtp_head_quant_type": "fp8"})


def test_modelopt_defaults_to_native_mtp_but_explicit_type_requantizes():
    assert Qwen35MtpHead._should_preserve_modelopt_mtp("modelopt", {})
    assert not Qwen35MtpHead._should_preserve_modelopt_mtp(
        "modelopt",
        {"mtp_head_quant_type": "nvfp4"},
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
