# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Configuration-only GPT-OSS precision advisory tests, without checkpoint weights or inference."""

from __future__ import annotations

import onnx_ir as ir
import pytest
from _builder_test_utils import load_builder_module
from test_precision import builder_module
from transformers import GptOssConfig


@pytest.fixture
def build_gptoss(monkeypatch, tmp_path):
    model_class = load_builder_module("gptoss").GPTOSSModel
    models = []
    monkeypatch.setattr(builder_module, "GPTOSSModel", model_class)
    monkeypatch.setattr(model_class, "make_genai_config", lambda self, *args: models.append(self))
    monkeypatch.setattr(model_class, "save_processing", lambda *args: None)

    def build(num_layers, num_experts, precision, ep="cuda", **extra_options):
        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            num_hidden_layers=num_layers,
            num_local_experts=num_experts,
        )
        builder_module.create_model(
            "",
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            precision,
            ep,
            str(tmp_path / "cache"),
            config_only=True,
            hf_details={"extra_kwargs": {}, "hf_name": "", "hf_config": config},
            **extra_options,
        )
        return models[-1]

    return build


@pytest.mark.parametrize(
    "num_layers,num_experts,exported_layers",
    [(36, 128, None), (24, 32, None), (36, 32, None), (24, 128, None), (36, 128, 35), (36, 128, 36)],
)
@pytest.mark.parametrize(
    "precision,extra_options,expected_io_dtype",
    [
        ("fp16", {}, ir.DataType.FLOAT16),
        ("int4", {}, ir.DataType.FLOAT16),
        ("int8", {}, ir.DataType.FLOAT16),
        ("bf16", {}, ir.DataType.BFLOAT16),
        ("fp32", {}, ir.DataType.FLOAT),
        ("int4", {"use_cuda_bf16": True}, ir.DataType.BFLOAT16),
    ],
)
def test_gptoss_precision_advisory_scope(
    build_gptoss, capsys, num_layers, num_experts, exported_layers, precision, extra_options, expected_io_dtype
):
    options = dict(extra_options)
    if exported_layers is not None:
        options["num_hidden_layers"] = exported_layers

    model = build_gptoss(num_layers, num_experts, precision, **options)

    expected_layers = num_layers if exported_layers is None else exported_layers
    should_warn = expected_layers == 36 and num_experts == 128 and expected_io_dtype == ir.DataType.FLOAT16
    output = capsys.readouterr().out
    assert ("NaN logits" in output) == should_warn
    if should_warn:
        assert "36-layer, 128-expert GPT-OSS (120B)" in output
        assert "may overflow" in output
    assert model.io_dtype == expected_io_dtype
    assert model.onnx_dtype == builder_module.set_onnx_dtype(precision, options)
    assert model.num_layers == expected_layers


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx", "NvTensorRtRtx", "webgpu", "dml"])
def test_gptoss_precision_advisory_guidance_matches_provider(build_gptoss, capsys, ep):
    build_gptoss(36, 128, "int4", ep)

    output = capsys.readouterr().out
    assert "may overflow" in output
    if ep in {"cuda", "trt-rtx", "NvTensorRtRtx"}:
        assert "--precision bf16" in output
        assert "--precision int4 --extra_options use_cuda_bf16=true" in output
    else:
        assert "FP32 I/O" in output
        assert "--precision bf16" not in output
        assert "use_cuda_bf16=true" not in output


@pytest.mark.parametrize("io_dtype", ["fp16", "bf16", "fp32"])
def test_gptoss_precision_advisory_uses_structured_io_dtype(build_gptoss, capsys, io_dtype):
    model = build_gptoss(
        36,
        128,
        "int4",
        target_options={"quant_config": {"io_dtype": io_dtype, "weights": {"type": "int4"}}},
    )

    assert ("NaN logits" in capsys.readouterr().out) == (io_dtype == "fp16")
    assert (
        model.io_dtype
        == {
            "fp16": ir.DataType.FLOAT16,
            "bf16": ir.DataType.BFLOAT16,
            "fp32": ir.DataType.FLOAT,
        }[io_dtype]
    )
    assert model.onnx_dtype == ir.DataType.INT4


@pytest.mark.parametrize("ep,extra_options", [("cpu", {}), ("webgpu", {"use_webgpu_fp32": True})])
def test_gptoss_fp32_quantized_io_does_not_warn(build_gptoss, capsys, ep, extra_options):
    model = build_gptoss(36, 128, "int4", ep, **extra_options)

    assert "NaN logits" not in capsys.readouterr().out
    assert model.io_dtype == ir.DataType.FLOAT
