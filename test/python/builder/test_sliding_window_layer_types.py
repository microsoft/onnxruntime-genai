# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Configs without ``layer_types`` must keep their ``sliding_window`` on every layer (issue #2832)."""

import json
from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

base = load_builder_module("base")
mistral = load_builder_module("mistral")
phi = load_builder_module("phi")
qwen = load_builder_module("qwen")

SMALL_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}


@pytest.mark.parametrize(
    "config, expected",
    [
        (SimpleNamespace(sliding_window=4096), "sliding_attention"),
        (SimpleNamespace(sliding_window=None), "full_attention"),
        (SimpleNamespace(sliding_window=0), "full_attention"),
        (SimpleNamespace(), "full_attention"),
        (SimpleNamespace(sliding_window=4096, use_sliding_window=False), "full_attention"),
        (SimpleNamespace(sliding_window=4096, use_sliding_window=True), "sliding_attention"),
    ],
)
def test_default_layer_type(config, expected):
    assert base.Model.__new__(base.Model).default_layer_type(config) == expected


def transformers_config(class_name, architecture, **kwargs):
    transformers = pytest.importorskip("transformers")
    # _builder_test_utils installs an importable transformers stub when the package is absent.
    if not hasattr(transformers, class_name):
        pytest.skip(f"transformers does not provide {class_name}")
    config = getattr(transformers, class_name)(**(SMALL_CONFIG | kwargs))
    config.architectures = [architecture]
    return config


def local_window_sizes(model):
    # Run make_attention with its subgraph builders stubbed out and record the window each layer is built with.
    sizes = []
    model.make_attention_input_proj = lambda *args, **kwargs: None
    model.make_attention_qk_subgraph = lambda *args, **kwargs: sizes.append(
        model.get_attention_op_attributes()["local_window_size"]
    )
    model.make_attention_output_proj = lambda *args, **kwargs: None
    for layer_id in range(model.num_layers):
        model.make_attention(layer_id, attention=None, root_input="")
    return sizes


def build(model_class, config, tmp_path):
    return model_class(config, 1, 1, "cpu", str(tmp_path), {})


def test_mistral_sliding_window_applies_to_every_layer(tmp_path):
    config = transformers_config("MistralConfig", "MistralForCausalLM", sliding_window=4)
    model = build(mistral.MistralModel, config, tmp_path)

    assert model.layer_types == ["sliding_attention", "sliding_attention"]
    assert local_window_sizes(model) == [4, 4]


def test_mistral_without_sliding_window_uses_full_attention(tmp_path):
    config = transformers_config("MistralConfig", "MistralForCausalLM", sliding_window=None)
    model = build(mistral.MistralModel, config, tmp_path)

    assert model.layer_types == ["full_attention", "full_attention"]
    assert local_window_sizes(model) == [-1, -1]


def test_phi3_sliding_window_applies_to_every_layer(tmp_path):
    config = transformers_config("Phi3Config", "Phi3ForCausalLM", sliding_window=8)
    model = build(phi.Phi3MiniModel, config, tmp_path)

    assert model.layer_types == ["sliding_attention", "sliding_attention"]
    assert local_window_sizes(model) == [8, 8]


@pytest.mark.parametrize("ep", ["cpu", "cuda"])
@pytest.mark.parametrize("window_size", [32, 64, 128])
def test_phi3_only_compacts_windows_smaller_than_context(tmp_path, ep, window_size):
    config = transformers_config("Phi3Config", "Phi3ForCausalLM", sliding_window=window_size)
    io_dtype = base.ir.DataType.FLOAT16 if ep == "cuda" else base.ir.DataType.FLOAT
    model = phi.Phi3MiniModel(config, io_dtype, io_dtype, ep, str(tmp_path), {})

    attributes = model.get_attention_op_attributes()
    assert attributes["local_window_size"] == window_size
    assert bool(attributes.get("sliding_window_cache")) is (window_size < model.context_length)
    cache_shape = model.make_key_value_cache_shape(0, ["batch_size", 2, "past_sequence_length", 16])
    assert cache_shape[2] == ("past_sliding_length" if window_size < model.context_length else "past_sequence_length")

    model.make_genai_config(config, {}, str(tmp_path))
    runtime_config = json.loads((tmp_path / "genai_config.json").read_text())
    assert ("sliding_window" in runtime_config["model"]["decoder"]) is (window_size < model.context_length)


@pytest.mark.parametrize("ep", ["cpu", "cuda"])
@pytest.mark.parametrize("window_size", [64, 128])
def test_phi3_full_context_window_runs_short_generator_session(tmp_path, ep, window_size):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    og = pytest.importorskip("onnxruntime_genai")
    if ep == "cuda" and not og.is_cuda_available():
        pytest.skip("needs the CUDA build of onnxruntime_genai")

    config = transformers_config(
        "Phi3Config",
        "Phi3ForCausalLM",
        vocab_size=320,
        sliding_window=window_size,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._name_or_path = str(tmp_path)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = transformers.Phi3ForCausalLM(config).eval()

    io_dtype = base.ir.DataType.FLOAT16 if ep == "cuda" else base.ir.DataType.FLOAT
    builder = phi.Phi3MiniModel(config, io_dtype, io_dtype, ep, str(tmp_path), {})
    builder.load_weights = lambda input_path: hf_model
    builder.make_model("")
    builder.save_model(str(tmp_path))
    builder.make_genai_config(config, {}, str(tmp_path))

    model = og.Model(str(tmp_path))
    params = og.GeneratorParams(model)
    params.set_search_options(min_length=8, max_length=8)
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([[1, 3, 4]], dtype=np.int32))
    while not generator.is_done():
        sequence = generator.get_sequence(0)
        with torch.no_grad():
            expected = hf_model(torch.tensor([sequence.tolist()])).logits[0, -1].numpy()
        np.testing.assert_allclose(generator.get_logits().reshape(-1), expected, rtol=0, atol=1e-3)
        generator.generate_next_token()

    assert len(generator.get_sequence(0)) == 8
    assert generator.get_input("past_key_values.0.key").shape[2] == 8


def test_qwen2_with_sliding_window_disabled_uses_full_attention(tmp_path):
    config = transformers_config("Qwen2Config", "Qwen2ForCausalLM", sliding_window=4, use_sliding_window=False)
    model = build(qwen.QwenModel, config, tmp_path)

    assert model.layer_types == ["full_attention", "full_attention"]
    assert local_window_sizes(model) == [-1, -1]
