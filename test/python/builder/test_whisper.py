# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the Whisper model builder.

The encoder graph makes the cross-attention KV caches that the decoder reads, one key and one value for each decoder
layer: the runtime's CrossCache reads `model.decoder.num_hidden_layers` pairs by the names in genai_config.json. Some
Whisper checkpoints have fewer decoder layers than encoder layers (whisper-large-v3-turbo: 32 and 4,
distil-large-v3: 32 and 2).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from _builder_test_utils import load_builder_module

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
# `_builder_test_utils` substitutes stubs for onnxruntime and transformers when they are not installed
if not hasattr(ort, "InferenceSession") or not hasattr(transformers, "WhisperForConditionalGeneration"):
    pytest.skip("needs onnxruntime and transformers' Whisper classes", allow_module_level=True)

whisper_module = load_builder_module("whisper")
ir = whisper_module.ir


def _save_tiny_whisper(path, encoder_layers, decoder_layers):
    config = transformers.WhisperConfig(
        vocab_size=128,
        num_mel_bins=80,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        d_model=32,
        max_source_positions=1500,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        begin_suppress_tokens=None,
        suppress_tokens=None,
    )
    torch.manual_seed(0)
    model = transformers.WhisperForConditionalGeneration(config).eval()
    # A new model's biases are zeros, which the builder leaves out of the graph; give them values as in a checkpoint
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".bias"):
                parameter.normal_(std=0.02)
    model.save_pretrained(path)
    return model


def _export(tmp_path, encoder_layers, decoder_layers, extra_options):
    hf_dir, out_dir = tmp_path / "hf", tmp_path / "onnx"
    hf_model = _save_tiny_whisper(hf_dir, encoder_layers, decoder_layers)
    config = transformers.AutoConfig.from_pretrained(hf_dir)
    builder = whisper_module.WhisperModel(
        config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path / "cache"), extra_options
    )
    builder.make_model(str(hf_dir))
    out_dir.mkdir()
    builder.save_model(str(out_dir))
    builder.make_genai_config(str(hf_dir), {}, str(out_dir))
    model_config = json.loads((out_dir / "genai_config.json").read_text())["model"]
    return hf_model, config, out_dir, model_config


def _cross_names(names, keys, num_layers):
    return [names[key] % i for key in keys for i in range(num_layers)]


@pytest.mark.parametrize("encoder_layers, decoder_layers", [(2, 2), (2, 1)])
def test_encoder_makes_cross_attention_kv_caches_for_each_decoder_layer(tmp_path, encoder_layers, decoder_layers):
    hf_model, config, out_dir, model_config = _export(tmp_path, encoder_layers, decoder_layers, {})

    # The names the runtime reads: one key and one value per decoder layer
    num_layers = model_config["decoder"]["num_hidden_layers"]
    assert num_layers == decoder_layers
    encoder_outputs = model_config["encoder"]["outputs"]
    present_names = _cross_names(encoder_outputs, ("cross_present_key_names", "cross_present_value_names"), num_layers)
    past_names = _cross_names(
        model_config["decoder"]["inputs"], ("cross_past_key_names", "cross_past_value_names"), num_layers
    )

    encoder = onnx.load(str(out_dir / "encoder.onnx"))
    decoder = onnx.load(str(out_dir / "decoder.onnx"))
    assert {value.name for value in encoder.graph.output} == {encoder_outputs["encoder_hidden_states"], *present_names}
    assert set(past_names) <= {value.name for value in decoder.graph.input}

    # Each decoder layer's key and value come from that layer's cross-attention projections
    features = np.random.default_rng(0).standard_normal((1, config.num_mel_bins, 3000), dtype=np.float32)
    session = ort.InferenceSession(str(out_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, {"audio_features": features}), strict=True))
    num_heads = config.decoder_attention_heads
    head_size = config.d_model // num_heads
    with torch.no_grad():
        hidden_states = hf_model.model.encoder(torch.from_numpy(features)).last_hidden_state
        np.testing.assert_allclose(
            outputs[encoder_outputs["encoder_hidden_states"]], hidden_states.numpy(), rtol=1e-4, atol=1e-4
        )
        for i, layer in enumerate(hf_model.model.decoder.layers):
            for key, proj in (
                ("cross_present_key_names", layer.encoder_attn.k_proj),
                ("cross_present_value_names", layer.encoder_attn.v_proj),
            ):
                expected = proj(hidden_states).reshape(1, -1, num_heads, head_size).transpose(1, 2)
                np.testing.assert_allclose(outputs[encoder_outputs[key] % i], expected.numpy(), rtol=1e-4, atol=1e-4)


def test_num_hidden_layers_option_sets_the_number_of_cross_attention_kv_caches(tmp_path):
    # `num_hidden_layers` in extra_options sets the layer count of both the encoder and the decoder
    _, _, out_dir, model_config = _export(tmp_path, 2, 2, {"num_hidden_layers": 1})

    assert model_config["decoder"]["num_hidden_layers"] == 1
    encoder_outputs = model_config["encoder"]["outputs"]
    present_names = _cross_names(encoder_outputs, ("cross_present_key_names", "cross_present_value_names"), 1)
    encoder = onnx.load(str(out_dir / "encoder.onnx"), load_external_data=False)
    assert {value.name for value in encoder.graph.output} == {encoder_outputs["encoder_hidden_states"], *present_names}
