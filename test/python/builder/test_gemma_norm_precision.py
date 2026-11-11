# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
"""Gemma's RMSNorm weight offset and embedding scale must not lose precision.

Transformers 5 loads checkpoints in their stored dtype, which is bfloat16 for the Gemma
checkpoints on the Hub. Gemma's RMSNorm scales by ``1 + weight``, and adding 1 in
bfloat16 rounds the result to 8 significant bits before it is written to an FP32 model.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import pytest
import torch
from test_precision import builder_module

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma3TextConfig"):
    pytest.skip("transformers does not provide Gemma3TextConfig", allow_module_level=True)

HIDDEN_SIZE = 72  # sqrt(72) = 8.485..., which rounding to two decimals changes


@pytest.fixture
def bf16_gemma3_checkpoint(tmp_path):
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=8,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        architectures=["Gemma3ForCausalLM"],
    )
    torch.manual_seed(0)
    model = transformers.Gemma3ForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                # Gemma norm weights are small offsets from 0, e.g. 0.37 -> scale 1.37.
                param.uniform_(-0.5, 0.5)
    model.to(torch.bfloat16).save_pretrained(tmp_path / "input")
    return model, tmp_path


def build_fp32(monkeypatch, tmp_path):
    model_class = builder_module.Gemma3Model
    monkeypatch.setattr(model_class, "make_genai_config", lambda self, *args: None)
    monkeypatch.setattr(model_class, "save_processing", lambda *args: None)
    input_path, cache_dir = str(tmp_path / "input"), str(tmp_path / "cache")
    extra_options = {}
    extra_options["hf_details"] = builder_module.get_hf_details("", input_path, cache_dir, extra_options)
    builder_module.create_model("", input_path, str(tmp_path / "output"), "fp32", "cpu", cache_dir, **extra_options)
    onnx_model = ir.load(tmp_path / "output" / "model.onnx")
    constants = {name: value.const_value.numpy() for name, value in onnx_model.graph.initializers.items()}
    for node in onnx_model.graph:
        if node.op_type == "Constant" and "value" in node.attributes:
            constants[node.outputs[0].name] = node.attributes["value"].as_tensor().numpy()
    return constants


def test_rmsnorm_offset_is_added_in_fp32(monkeypatch, bf16_gemma3_checkpoint):
    model, tmp_path = bf16_gemma3_checkpoint
    initializers = build_fp32(monkeypatch, tmp_path)
    hf_weights = {name: param.detach().to(torch.bfloat16) for name, param in model.named_parameters()}

    expected = {
        "model.layers.0.input_layernorm.weight": hf_weights["model.layers.0.input_layernorm.weight"],
        "model.layers.1.post_feedforward_layernorm.weight": hf_weights[
            "model.layers.1.post_feedforward_layernorm.weight"
        ],
        "model.layers.0.attn.q_norm.layernorm.weight": hf_weights["model.layers.0.self_attn.q_norm.weight"],
        "model.layers.0.attn.k_norm.layernorm.weight": hf_weights["model.layers.0.self_attn.k_norm.weight"],
    }
    norm_names = sorted(name for name in initializers if "norm" in name)
    for onnx_name, bf16_weight in expected.items():
        assert onnx_name in initializers, f"{onnx_name} not in {norm_names}"
        actual = initializers[onnx_name]
        assert actual.dtype == np.float32
        # Transformers computes (1 + weight) on the weight upcast to FP32.
        np.testing.assert_array_equal(actual, bf16_weight.float().numpy() + 1, err_msg=onnx_name)


def test_embedding_scale_is_exact_sqrt_of_hidden_size(monkeypatch, bf16_gemma3_checkpoint):
    _, tmp_path = bf16_gemma3_checkpoint
    initializers = build_fp32(monkeypatch, tmp_path)
    scales = [
        value.item() for name, value in initializers.items() if value.size == 1 and abs(value.item() - 8.49) < 0.01
    ]
    assert scales, "embedding scale constant not found"
    for scale in scales:
        assert scale == pytest.approx(math.sqrt(HIDDEN_SIZE), rel=1e-7)
