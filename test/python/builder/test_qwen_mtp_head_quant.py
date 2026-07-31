# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""`mtp_head_quant_type` must reach the head's dense MatMuls, not just its QMoE experts."""

import onnx_ir as ir
import pytest

from models.builders.qwen import Qwen35MoeTextModel


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


def test_int4_lmhead_override_is_applied_as_mixed_precision():
    model = _resolve({"enable_mtp": True, "mtp_head_quant_type": "int8", "mtp_head_int4_lmhead": "true"})

    assert model._mtp_extra_options["matmul_mixed_precision"] == "last_matmul:int4"


def test_deprecated_mtp_head_int8_still_builds_an_int8_head():
    model = _resolve({"enable_mtp": True, "mtp_head_int8": "true"})

    assert model._mtp_onnx_dtype == ir.DataType.INT8
    assert model._mtp_extra_options["moe_quant_type"] == "int8"


def test_head_fp16_keeps_the_head_unquantized():
    model = _resolve({"enable_mtp": True, "mtp_head_fp16": "true", "moe_quant_type": "nvfp4"})

    assert model._mtp_onnx_dtype == ir.DataType.FLOAT16
    assert "moe_quant_type" not in model._mtp_extra_options


def test_head_fp16_and_head_quant_type_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve({"enable_mtp": True, "mtp_head_fp16": "true", "mtp_head_quant_type": "int8"})


def test_unknown_head_quant_type_is_rejected():
    with pytest.raises(ValueError, match="mtp_head_quant_type must be one of"):
        _resolve({"enable_mtp": True, "mtp_head_quant_type": "fp8"})
