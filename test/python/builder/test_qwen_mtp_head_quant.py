# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""`mtp_head_quant_type` must reach the head's dense MatMuls, not just its QMoE experts."""

import onnx_ir as ir
import pytest

from models.builders.qwen import Qwen35DenseMtpHead, Qwen35MoeTextModel


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
    assert model._mtp_extra_options["int4_block_size"] == 32
    assert model._mtp_extra_options["int4_algo_config"] == "rtn_last"


def test_quantized_main_model_does_not_get_int4_defaults():
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "int8"})

    assert "int4_algo_config" not in model._mtp_extra_options


def test_unknown_head_quant_type_is_rejected():
    with pytest.raises(ValueError, match="mtp_head_quant_type must be one of"):
        _resolve({"enable_mtp": True, "mtp_head_quant_type": "fp8"})


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("none", []),
        ("fc", ["/model/mtp/fc/MatMul"]),
        (
            "fc_attn",
            [
                "/model/mtp/fc/MatMul",
                "/model/layers.0/attn/q_proj/MatMul",
                "/model/layers.0/attn/k_proj/MatMul",
                "/model/layers.0/attn/v_proj/MatMul",
                "/model/layers.0/attn/o_proj/MatMul",
            ],
        ),
        (
            "all",
            [
                "/model/mtp/fc/MatMul",
                "/model/layers.0/attn/q_proj/MatMul",
                "/model/layers.0/attn/k_proj/MatMul",
                "/model/layers.0/attn/v_proj/MatMul",
                "/model/layers.0/attn/o_proj/MatMul",
                "/model/layers.0/mlp/gate_proj/MatMul",
                "/model/layers.0/mlp/up_proj/MatMul",
                "/model/layers.0/mlp/down_proj/MatMul",
            ],
        ),
    ],
)
def test_dense_mtp_fp16_exclusion_modes(mode, expected):
    model = object.__new__(Qwen35DenseMtpHead)
    model.onnx_dtype = ir.DataType.INT8
    model.quant_attrs = {"nodes_to_exclude": []}

    model._apply_fp16_exclusions({"mtp_head_fp16_exclude": mode})

    assert model.quant_attrs["nodes_to_exclude"] == expected


def test_dense_mtp_fp16_exclusions_preserve_existing_nodes():
    model = object.__new__(Qwen35DenseMtpHead)
    model.onnx_dtype = ir.DataType.INT8
    model.quant_attrs = {"nodes_to_exclude": ["existing"]}

    model._apply_fp16_exclusions({"mtp_head_fp16_exclude": "fc"})
    model._apply_fp16_exclusions({"mtp_head_fp16_exclude": "fc"})

    assert model.quant_attrs["nodes_to_exclude"] == ["existing", "/model/mtp/fc/MatMul"]


def test_unknown_dense_mtp_fp16_exclusion_mode_is_rejected():
    model = object.__new__(Qwen35DenseMtpHead)
    model.onnx_dtype = ir.DataType.INT8
    model.quant_attrs = {"nodes_to_exclude": []}

    with pytest.raises(ValueError, match="mtp_head_fp16_exclude must be one of"):
        model._apply_fp16_exclusions({"mtp_head_fp16_exclude": "attention"})


def test_dense_mtp_fp16_exclusions_require_quantized_head():
    model = object.__new__(Qwen35DenseMtpHead)
    model.onnx_dtype = ir.DataType.FLOAT16
    model.quant_attrs = {"nodes_to_exclude": []}

    with pytest.raises(ValueError, match="requires a quantized MTP head"):
        model._apply_fp16_exclusions({"mtp_head_fp16_exclude": "fc"})


def test_mtp_original_nvfp4_path_is_limited_to_shared_lm_head():
    model = object.__new__(Qwen35MoeTextModel)
    model.is_mtp_head = True

    assert model._nvfp4_dense_key_for_matmul("/lm_head/MatMul") == "lm_head"
    assert model._nvfp4_dense_key_for_matmul(
        "/model/layers.0/shared_expert/gate_proj/MatMul"
    ) is None
