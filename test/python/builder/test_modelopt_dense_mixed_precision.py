# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.quantized_model import ModeloptModel
from models.builders.qwen import Qwen35DenseMtpHead, Qwen35NativeQuantTextModel, Qwen35TextModel


def _make_loader(tensors):
    model = object.__new__(ModeloptModel)
    model._get = lambda name: tensors.get(name)
    return model


def test_dequant_linear_accepts_compressed_tensors_nvfp4_names():
    model = _make_loader(
        {
            "mlp.weight_packed": torch.full((2, 8), 0x22, dtype=torch.uint8),
            "mlp.weight_scale": torch.ones((2, 1), dtype=torch.float8_e4m3fn),
            "mlp.weight_global_scale": torch.tensor([4.0], dtype=torch.float32),
        }
    )

    weight = model._dequant_linear("mlp")

    assert weight.dtype == torch.bfloat16
    assert weight.shape == (2, 16)
    torch.testing.assert_close(weight, torch.full((2, 16), 0.25, dtype=torch.bfloat16))


def test_dequant_linear_broadcasts_per_channel_fp8_scale():
    model = _make_loader(
        {
            "mlp.weight": torch.ones((2, 4), dtype=torch.float8_e4m3fn),
            "mlp.weight_scale": torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16),
        }
    )

    weight = model._dequant_linear("mlp")

    torch.testing.assert_close(weight, torch.tensor([[0.5] * 4, [0.25] * 4], dtype=torch.bfloat16))


def _make_native_model(tensors, model_type=Qwen35NativeQuantTextModel):
    model = object.__new__(model_type)
    model._load_native_tensor_exact = lambda name, required=False: tensors.get(name)
    return model


def test_native_tensor_alias_inverts_compressed_tensors_global_scale():
    model = _make_native_model({"mlp.weight_global_scale": torch.tensor([6400.0])})

    scale = model._load_nvfp4_tensor("mlp.weight_scale_2")

    torch.testing.assert_close(scale, torch.tensor([1.0 / 6400.0]))


def test_dense_mtp_tensor_alias_inverts_compressed_tensors_global_scale():
    model = _make_native_model(
        {"lm_head.weight_global_scale": torch.tensor([6400.0])},
        Qwen35DenseMtpHead,
    )

    scale = model._load_nvfp4_tensor("lm_head.weight_scale_2")

    torch.testing.assert_close(scale, torch.tensor([1.0 / 6400.0]))


def test_dense_mtp_native_fp8_routing_is_limited_to_lm_head():
    model = _make_native_model(
        {"lm_head.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn)},
        Qwen35DenseMtpHead,
    )

    assert model._fp8_weight_key_for_matmul("/lm_head/MatMul") == "lm_head"
    assert model._fp8_weight_key_for_matmul("/model/layers.0/attn/q_proj/MatMul") is None


def test_dense_native_weight_routing_preserves_mixed_mlp_formats():
    nvfp4_key = "model.language_model.layers.0.mlp.gate_proj"
    fp8_key = "model.language_model.layers.56.mlp.gate_proj"
    model = _make_native_model(
        {
            f"{nvfp4_key}.weight_packed": torch.zeros((4, 2), dtype=torch.uint8),
            f"{fp8_key}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
            "lm_head.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        }
    )

    assert model._nvfp4_dense_key_for_matmul("/model/layers.0/mlp/gate_proj/MatMul") == nvfp4_key
    assert model._fp8_weight_key_for_matmul("/model/layers.0/mlp/gate_proj/MatMul") is None
    assert model._fp8_weight_key_for_matmul("/model/layers.56/mlp/gate_proj/MatMul") == fp8_key
    assert model._nvfp4_dense_key_for_matmul("/model/layers.56/mlp/gate_proj/MatMul") is None
    assert model._fp8_weight_key_for_matmul("/lm_head/MatMul") == "lm_head"


def test_dense_mtp_head_constructs_dense_decoder_layer():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    model = object.__new__(Qwen35DenseMtpHead)
    model._mtp_layer_config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
    )

    layer = model._make_mtp_decoder_layer()

    assert hasattr(layer, "self_attn")
    assert hasattr(layer.mlp, "gate_proj")
    assert not hasattr(layer.mlp, "experts")


def test_dense_mtp_head_dispatches_to_dense_layer_builder(monkeypatch):
    model = object.__new__(Qwen35DenseMtpHead)
    expected = object()
    calls = []

    def fake_make_layer(self, layer_id, layer):
        calls.append((self, layer_id, layer))
        return expected

    monkeypatch.setattr(Qwen35TextModel, "make_layer", fake_make_layer)

    layer = object()
    assert model.make_layer(0, layer) is expected
    assert calls == [(model, 0, layer)]


def test_dense_mtp_head_dispatches_to_dense_save(monkeypatch):
    model = object.__new__(Qwen35DenseMtpHead)
    calls = []

    def fake_save_model(self, out_dir):
        calls.append((self, out_dir))
        return "saved"

    monkeypatch.setattr(Qwen35TextModel, "save_model", fake_save_model)

    assert model.save_model("out") == "saved"
    assert calls == [(model, "out")]