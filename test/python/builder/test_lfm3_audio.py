# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM3-Audio model builder.

With `exclude_embeds` the builder writes the `lfm3_audio` pipeline: the LFM3 decoder, the speech encoder, the embedding
model that places speech features over the audio tokens, and their genai_config. No transformers release has the
LFM3-Audio classes, so the tests build from a stand-in with the module layout of the checkpoints' remote code. Its speech
encoder runs transformers' Lfm2Model, as the checkpoint's does, and the tests write out the downsampler and projector math.
Real checkpoints are compared outside this suite, since their remote code cannot run in CI.
"""

from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
from _builder_test_utils import check_node_shapes, ort_runs_causal_conv_with_state
from test_lfm2_vl import builder_module, ir
from test_lfm3 import Lfm3DecoderLayer, _stand_in_lfm3

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

LFM3AudioModel = builder_module.LFM3AudioModel
LFM3AudioEncoder = importlib.import_module("builders.lfm3_audio").LFM3AudioEncoder
LFM2VLEmbedding = importlib.import_module("builders.lfm2_vl").LFM2VLEmbedding

FIXTURE = Path(__file__).parents[2] / "models" / "lfm3-audio"
# Every width differs from the vocabulary size: the builder takes a Linear with vocab_size outputs as the LM head.
VOCAB_SIZE = 320
AUDIO_TOKEN = 300
EOS, TEXT_END, AUDIO_START, MIXED_START = 2, 311, 312, 313
TEXT_HIDDEN_SIZE = 72
LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention"]
NUM_LAYERS = len(LAYER_TYPES)
FP16, BF16, FP32, INT4 = ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT, ir.DataType.INT4
NUM_MELS = 128
# 4 heads of 16: GroupQueryAttention on CPU needs a head size that is a multiple of 16
SPEECH_HIDDEN_SIZE = 64
SPEECH_CHANNELS = 8
SPEECH_WINDOW = 4
SPEECH_LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]

# The tests that run the speech encoder in onnxruntime need com.microsoft:CausalConvWithState.
requires_causal_conv_with_state = pytest.mark.skipif(
    not ort_runs_causal_conv_with_state(),
    reason="the installed onnxruntime cannot run com.microsoft:CausalConvWithState",
)


def text_config_kwargs(layer_types=LAYER_TYPES):
    return {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": TEXT_HIDDEN_SIZE,
        "intermediate_size": 96,
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "sliding_window": 4,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        },
        "norm_eps": 1e-3,
        "conv_L_cache": 3,
        "tie_word_embeddings": True,
    }


def speech_lfm_config():
    """The LFM2 model the speech encoder runs over the downsampled frames, at a tiny size. Its FFN width rounds
    2 * 72 / 3 up to a multiple of 16, as the checkpoint's rounds 2 * 2560 / 3 up to a multiple of 256."""
    return transformers.Lfm2Config(
        vocab_size=2,
        hidden_size=SPEECH_HIDDEN_SIZE,
        intermediate_size=72,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=16,
        num_hidden_layers=len(SPEECH_LAYER_TYPES),
        layer_types=SPEECH_LAYER_TYPES,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6},
        norm_eps=0.05,
        conv_L_cache=3,
        eos_token_id=None,
    )


def stand_in_config(tmp_path):
    """The attributes of the remote code's Lfm3AudioConfig that the builder reads, at a tiny size."""
    config = transformers.PretrainedConfig(
        architectures=["Lfm3AudioForConditionalGeneration"],
        bos_token_id=1,
        eos_token_id=[EOS, TEXT_END],
        pad_token_id=0,
        text_config=transformers.PretrainedConfig(**text_config_kwargs()),
        audio_encoder_config=transformers.PretrainedConfig(
            num_mel_bins=NUM_MELS,
            conv_channels=SPEECH_CHANNELS,
            hidden_size=SPEECH_HIDDEN_SIZE,
            sliding_window=SPEECH_WINDOW,
            text_config=speech_lfm_config(),
        ),
        audio_token_id=AUDIO_TOKEN,
        audio_start_token_id=AUDIO_START,
        mixed_start_token_id=MIXED_START,
        text_end_token_id=TEXT_END,
    )
    config.name_or_path = str(tmp_path / "checkpoint")
    return config


def write_extractor(tmp_path, **overrides):
    """The feature extractor settings of a checkpoint's preprocessor_config.json; None leaves a key out."""
    path = tmp_path / "checkpoint"
    path.mkdir(exist_ok=True)
    extractor = {
        "encoder_stride": 8,
        "feature_extractor_type": "Lfm3AudioFeatureExtractor",
        "feature_size": 128,
        "hop_length": 160,
        "log_floor": -12.0,
        "n_fft": 400,
        "normalization_divisor": 4.0,
        "normalization_offset": 4.0,
        "padding_side": "right",
        "padding_value": 0.0,
        "sampling_rate": 16000,
        "win_length": 400,
        **overrides,
    }
    extractor = {key: value for key, value in extractor.items() if value is not None}
    (path / "preprocessor_config.json").write_text(json.dumps(extractor), encoding="utf-8")


def stand_in_downsampler():
    """Modules with the names and shapes of the checkpoint's downsampler: three stride-2 convolutions over the
    (mel, frame) plane, the last two depthwise with a 1x1 convolution after each, then lin_out over mels and channels."""
    channels, stride_2 = SPEECH_CHANNELS, {"kernel_size": 3, "stride": 2, "padding": 1}
    downsample = torch.nn.Module()
    downsample.conv_in = torch.nn.Conv2d(1, channels, **stride_2)
    downsample.conv_mid = torch.nn.Conv2d(channels, channels, groups=channels, **stride_2)
    downsample.lin_mid = torch.nn.Conv2d(channels, channels, 1)
    downsample.conv_end = torch.nn.Conv2d(channels, channels, groups=channels, **stride_2)
    downsample.lin_end = torch.nn.Conv2d(channels, channels, 1)
    downsample.lin_out = torch.nn.Conv1d(NUM_MELS // 8 * channels, SPEECH_HIDDEN_SIZE, 1)
    return downsample


def stacked_layers(layer_class, text, count, use_sandwich_norm=True):
    return torch.nn.ModuleList(layer_class(text, "conv", use_sandwich_norm) for _ in range(count))


def stand_in_model(num_layers=NUM_LAYERS):
    """A module tree in the checkpoint's order. After the language model come the speech encoder (transformers'
    Lfm2Model without its token table, as in the checkpoint) and projector, and a depth decoder with layers of the
    decoder's class and a norm of its own."""
    text = types.SimpleNamespace(**text_config_kwargs(LAYER_TYPES[:num_layers]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        text_model = _stand_in_lfm3(text, use_sandwich_norm=True)

        hf_model = torch.nn.Module()
        hf_model.model = torch.nn.Module()
        hf_model.model.language_model = text_model.model
        hf_model.lm_head = text_model.lm_head

        encoder = hf_model.model.audio_encoder = torch.nn.Module()
        encoder.downsample = stand_in_downsampler()
        encoder.lfm = transformers.Lfm2Model(speech_lfm_config())
        encoder.lfm.embed_tokens = None
        projector = hf_model.model.audio_projector = torch.nn.Module()
        projector.layer_norm = torch.nn.LayerNorm(SPEECH_HIDDEN_SIZE)
        projector.linear_1 = torch.nn.Linear(SPEECH_HIDDEN_SIZE, TEXT_HIDDEN_SIZE)
        projector.linear_2 = torch.nn.Linear(TEXT_HIDDEN_SIZE, TEXT_HIDDEN_SIZE)
        hf_model.model.codebook_embedding = torch.nn.Embedding(24, TEXT_HIDDEN_SIZE)
        depth_decoder = hf_model.model.depth_decoder = torch.nn.Module()
        depth_decoder.model = torch.nn.Module()
        depth_decoder.model.layers = stacked_layers(Lfm3DecoderLayer, text, 6)
        depth_decoder.model.embedding_norm = torch.nn.RMSNorm(TEXT_HIDDEN_SIZE)

        for name, parameter in hf_model.named_parameters():
            is_norm = "norm" in name and name.endswith("weight")
            torch.nn.init.normal_(parameter, mean=1.0 if is_norm else 0.0, std=0.2)
        # The checkpoint ties the logits to the token table
        hf_model.lm_head.weight = text_model.model.embed_tokens.weight
    hf_model.get_input_embeddings = lambda: text_model.model.embed_tokens
    return hf_model


def make_pipeline(tmp_path, ep="cpu", io_dtype=FP32, onnx_dtype=None, **options):
    config = stand_in_config(tmp_path)
    write_extractor(tmp_path)
    return LFM3AudioModel(
        config, io_dtype, onnx_dtype or io_dtype, ep, str(tmp_path / "cache"), {"exclude_embeds": True, **options}
    )


def build_pipeline(tmp_path, ep="cpu", io_dtype=FP32, onnx_dtype=None):
    model = make_pipeline(tmp_path, ep, io_dtype, onnx_dtype)
    hf_model = stand_in_model()
    model.embedding.load_weights = lambda input_path: hf_model
    model.make_model("")
    (tmp_path / "out").mkdir()
    model.save_model(str(tmp_path / "out"))
    return model, hf_model


def text_only_model(hf_model):
    """The same decoder weights in a tree with nothing else."""
    text_model = torch.nn.Module()
    text_model.model = torch.nn.Module()
    text_model.model.language_model = hf_model.model.language_model
    text_model.lm_head = hf_model.lm_head
    return text_model


def graph_of(path):
    return onnx.load(path, load_external_data=True).graph


def initializers(graph):
    return {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in graph.initializer}


def build_text_decoder(tmp_path, hf_model):
    (tmp_path / "text").mkdir()
    decoder = builder_module.LFM3Model(
        stand_in_config(tmp_path), FP32, FP32, "cpu", str(tmp_path / "cache"), {"exclude_embeds": True}
    )
    decoder.load_weights = lambda input_path: text_only_model(hf_model)
    decoder.make_model("")
    decoder.save_model(str(tmp_path / "text"))
    return tmp_path / "text" / "model.onnx"


def test_decoder_takes_the_language_model_and_nothing_else(tmp_path):
    _, hf_model = build_pipeline(tmp_path)
    graph = graph_of(tmp_path / "out" / "model.onnx")

    layers = {node.name.split("/")[2] for node in graph.node if node.name.startswith("/model/layers.")}
    assert layers == {f"layers.{layer_id}" for layer_id in range(NUM_LAYERS + 1)}

    weights = initializers(graph)
    (final_norm,) = [name for name in weights if "final_norm_layernorm" in name and name.endswith("weight")]
    np.testing.assert_array_equal(
        weights[final_norm], hf_model.model.language_model.embedding_norm.weight.detach().numpy()
    )
    assert not np.array_equal(weights[final_norm], hf_model.model.audio_encoder.lfm.embedding_norm.weight.detach())
    assert not np.array_equal(weights[final_norm], hf_model.model.depth_decoder.model.embedding_norm.weight.detach())

    # With the speech encoder and depth decoder gone the export is the same graph with the same weights
    text_graph = graph_of(build_text_decoder(tmp_path, hf_model))
    assert [(node.name, node.op_type) for node in graph.node] == [(node.name, node.op_type) for node in text_graph.node]
    text_weights = initializers(text_graph)
    assert weights.keys() == text_weights.keys()
    for name, value in weights.items():
        np.testing.assert_array_equal(value, text_weights[name], err_msg=name)


@pytest.mark.parametrize(
    "options, loads",
    [
        # The embedding model's load is shared with the decoder
        ({}, [NUM_LAYERS]),
        # The decoder loads its own smaller model, and the embedding model still reads the whole checkpoint
        ({"num_hidden_layers": 1}, [NUM_LAYERS, 1]),
    ],
)
def test_loads_through_the_lfm3_auto_class(monkeypatch, tmp_path, options, loads):
    model = make_pipeline(tmp_path, **options)
    hf_config = stand_in_config(tmp_path)
    loaded = []

    def from_pretrained(*args, config=hf_config, **kwargs):
        loaded.append(config.text_config.num_hidden_layers)
        return stand_in_model(config.text_config.num_hidden_layers)

    # Without the LFM3 loader the embedding model asks for Lfm2VlForConditionalGeneration, which this stub does not have
    stub = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: stand_in_config(tmp_path)),
        AutoModelForMultimodalLM=types.SimpleNamespace(from_pretrained=from_pretrained),
    )
    monkeypatch.setattr(importlib.import_module("builders.base"), "transformers", stub)
    model.make_model("")

    assert loaded == loads
    (tmp_path / "out").mkdir()
    model.save_model(str(tmp_path / "out"))
    layers = {
        node.name.split("/")[2]
        for node in graph_of(tmp_path / "out" / "model.onnx").node
        if node.name.startswith("/model/layers.")
    }
    assert layers == {f"layers.{layer_id}" for layer_id in range(loads[-1] + 1)}
    table = initializers(graph_of(tmp_path / "out" / "embeddings.onnx"))["model.embed_tokens.weight"]
    assert table.shape == (VOCAB_SIZE, TEXT_HIDDEN_SIZE)


def run_embedding(tmp_path, ids, features):
    session = ort.InferenceSession(str(tmp_path / "out" / "embeddings.onnx"), providers=["CPUExecutionProvider"])
    return session.run(None, {"input_ids": np.array(ids, np.int64), "audio_features": features})[0]


@pytest.mark.parametrize("ep, io_dtype", [("cpu", FP32), ("cuda", FP16)])
@pytest.mark.parametrize(
    "ids",
    [
        [[1, AUDIO_TOKEN, 5, AUDIO_TOKEN, AUDIO_TOKEN, 7, AUDIO_TOKEN]],
        [[AUDIO_TOKEN] * 5],
        # Zero rows pass through
        [[7, 8, 9]],
    ],
)
def test_embedding_model_matches_masked_scatter(tmp_path, ep, io_dtype, ids):
    _, hf_model = build_pipeline(tmp_path, ep, io_dtype)
    ids_array = np.array(ids, np.int64)
    features = np.random.default_rng(0).standard_normal(((ids_array == AUDIO_TOKEN).sum(), TEXT_HIDDEN_SIZE))
    # The features come in the speech encoder's type, which is the decoder's here
    features = features.astype(np.float32 if io_dtype == FP32 else np.float16)

    embeds = run_embedding(tmp_path, ids, features)

    torch_dtype = torch.float32 if io_dtype == FP32 else torch.float16
    with torch.no_grad():
        expected = hf_model.get_input_embeddings()(torch.from_numpy(ids_array)).to(torch_dtype)
    mask = torch.from_numpy(ids_array == AUDIO_TOKEN).unsqueeze(-1)
    expected = expected.masked_scatter(mask, torch.from_numpy(features).to(torch_dtype)).numpy()
    np.testing.assert_array_equal(embeds, expected)


@pytest.mark.parametrize("rows", [2, 4])
def test_embedding_model_fails_on_a_wrong_row_count(tmp_path, rows):
    build_pipeline(tmp_path)
    features = np.zeros((rows, TEXT_HIDDEN_SIZE), np.float32)
    with pytest.raises(Exception, match="ScatterND|shape|dimension"):
        run_embedding(tmp_path, [[1, AUDIO_TOKEN, AUDIO_TOKEN, AUDIO_TOKEN]], features)


def test_embedding_model_places_each_modality_over_its_own_token(tmp_path):
    config = stand_in_config(tmp_path)
    hf_model = stand_in_model()
    embedding = LFM2VLEmbedding(config, FP32, FP32, "cpu", str(tmp_path / "cache"), {})
    image_token = AUDIO_TOKEN - 1
    embedding.embed_attrs["features"] += [("image_features", image_token, FP16), ("audio_features", AUDIO_TOKEN, FP32)]
    embedding.load_weights = lambda input_path: hf_model
    embedding.make_model("")
    embedding.save_model(str(tmp_path))

    ids = np.array([[AUDIO_TOKEN, image_token, 5, image_token, AUDIO_TOKEN, AUDIO_TOKEN]], np.int64)
    rng = np.random.default_rng(0)
    images = rng.standard_normal((2, TEXT_HIDDEN_SIZE)).astype(np.float16)
    audios = rng.standard_normal((3, TEXT_HIDDEN_SIZE)).astype(np.float32)
    session = ort.InferenceSession(str(tmp_path / "embeddings.onnx"), providers=["CPUExecutionProvider"])
    (embeds,) = session.run(None, {"input_ids": ids, "image_features": images, "audio_features": audios})

    with torch.no_grad():
        expected = hf_model.get_input_embeddings()(torch.from_numpy(ids))
    for token, features in ((image_token, images), (AUDIO_TOKEN, audios)):
        mask = torch.from_numpy(ids == token).unsqueeze(-1)
        expected = expected.masked_scatter(mask, torch.from_numpy(features.astype(np.float32)))
    np.testing.assert_array_equal(embeds, expected.numpy())
    names = {node.name for node in onnx.load(tmp_path / "embeddings.onnx", load_external_data=False).graph.node}
    assert {"/model/embed_tokens/image_features/ScatterND", "/model/embed_tokens/audio_features/ScatterND"} <= names


def build_speech(tmp_path, hf_model, ep="cpu", dtype=FP32):
    speech = LFM3AudioEncoder(stand_in_config(tmp_path), dtype, dtype, ep, str(tmp_path / "cache"), {})
    speech.load_weights = lambda input_path: hf_model
    speech.make_model("")
    speech.save_model(str(tmp_path))
    return speech


def speech_session(tmp_path, hf_model):
    build_speech(tmp_path, hf_model)
    return ort.InferenceSession(str(tmp_path / "audio_encoder.onnx"), providers=["CPUExecutionProvider"])


def random_mel(frames):
    return np.random.default_rng(frames).standard_normal((NUM_MELS, frames), dtype=np.float32)


def run_speech(session, mel):
    return session.run(None, {"input_features": mel[None]})[0]


def expected_speech_features(hf_model, mel, window):
    """The encoder's math on one clip: the downsampler and projector written out here, and the stand-in's Lfm2Model with
    each token seeing the `window` tokens that end at itself (all tokens up to itself for None)."""
    # Lfm2Model takes a mask per layer type from 5.13.0
    pytest.importorskip("transformers", minversion="5.13.0")
    encoder, projector = hf_model.model.audio_encoder, hf_model.model.audio_projector
    down, silu = encoder.downsample, torch.nn.functional.silu
    with torch.no_grad():
        states = silu(down.conv_in(torch.from_numpy(mel)[None, None]))
        states = silu(down.lin_mid(down.conv_mid(states)))
        states = silu(down.lin_end(down.conv_end(states)))
        tokens = down.lin_out(states.flatten(1, 2)).transpose(1, 2)
        positions = torch.arange(tokens.shape[1])
        distance = positions[:, None] - positions[None]
        visible = (distance >= 0) & (distance < (window or tokens.shape[1]))
        band = torch.zeros(visible.shape).masked_fill(~visible, float("-inf"))[None, None]
        hidden = encoder.lfm(inputs_embeds=tokens, attention_mask={"full_attention": band, "conv": None})
        normed = projector.layer_norm(hidden.last_hidden_state)
        return projector.linear_2(torch.nn.functional.gelu(projector.linear_1(normed)))[0].numpy()


@pytest.fixture(scope="module")
def speech_model(tmp_path_factory):
    """One build of the speech encoder, for the tests that only run it."""
    hf_model = stand_in_model()
    return speech_session(tmp_path_factory.mktemp("speech"), hf_model), hf_model


@requires_causal_conv_with_state
# 1 to 9 frames give 1 or 2 tokens; 63 and 200 frames give 8 and 25, past the window of 4
@pytest.mark.parametrize("frames", [1, 7, 8, 9, 63, 200])
def test_speech_encoder_matches_the_encoder_math(speech_model, frames):
    session, hf_model = speech_model
    mel = random_mel(frames)
    features = run_speech(session, mel)
    assert features.shape == (-(-frames // 8), TEXT_HIDDEN_SIZE)
    expected = expected_speech_features(hf_model, mel, SPEECH_WINDOW)
    np.testing.assert_allclose(features, expected, rtol=1e-4, atol=1e-5)


@requires_causal_conv_with_state
def test_speech_encoder_window_counts_the_frame_itself(speech_model):
    session, hf_model = speech_model
    mel = random_mel(200)
    features = run_speech(session, mel)
    for window in (SPEECH_WINDOW + 1, None):
        assert np.abs(features - expected_speech_features(hf_model, mel, window)).max() > 1e-3


@requires_causal_conv_with_state
@pytest.mark.parametrize("layer_id", range(len(SPEECH_LAYER_TYPES)))
def test_each_speech_encoder_layer_reaches_the_features(speech_model, tmp_path, layer_id):
    session, _ = speech_model
    mel = random_mel(63)
    hf_model = stand_in_model()
    layer = hf_model.model.audio_encoder.lfm.layers[layer_id]
    with torch.no_grad():
        (layer.conv if hasattr(layer, "conv") else layer.self_attn).out_proj.weight.zero_()
        layer.feed_forward.w2.weight.zero_()
    zeroed = speech_session(tmp_path, hf_model)
    assert np.abs(run_speech(session, mel) - run_speech(zeroed, mel)).max() > 1e-3


@pytest.mark.parametrize("ep, dtype", [("cpu", FP32), ("cuda", FP16), ("webgpu", FP32)])
def test_speech_graph_keeps_the_runtime_contract(tmp_path, ep, dtype):
    speech = build_speech(tmp_path, stand_in_model(), ep, dtype)
    model_proto = ir.serde.serialize_model(speech.model)
    check_node_shapes(model_proto)
    graph = model_proto.graph

    def signature(values):
        tensors = [(value.name, value.type.tensor_type) for value in values]
        return [
            (name, tensor.elem_type, [d.dim_param or d.dim_value for d in tensor.shape.dim]) for name, tensor in tensors
        ]

    # One clip per run, no length input, and nothing but the features out
    assert signature(graph.input) == [("input_features", dtype, [1, NUM_MELS, "num_frames"])]
    assert signature(graph.output) == [("audio_features", dtype, ["num_audio_tokens", TEXT_HIDDEN_SIZE])]
    ops = {node.op_type for node in graph.node}
    assert {"Conv", "CausalConvWithState", "GroupQueryAttention"} <= ops
    assert not ops & {"NonZero", "Loop", "If", "Identity", "Resize", "Compress"}
    attentions = [node for node in graph.node if node.op_type == "GroupQueryAttention"]
    assert len(attentions) == SPEECH_LAYER_TYPES.count("full_attention")
    for node in attentions:
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        assert (attributes["local_window_size"], attributes["num_heads"], attributes["kv_num_heads"]) == (
            SPEECH_WINDOW,
            4,
            2,
        )
        assert "sliding_window_cache" not in attributes
        assert attributes["do_rotary"] == 1
        # The q/k norms run inside the op on CUDA and WebGPU, as separate nodes on the CPU
        assert sum(map(bool, node.input[14:16])) == (0 if ep == "cpu" else 2)
        # No past, but named presents: CUDA requires them
        assert list(node.input[3:5]) == ["", ""]
        assert all(node.output[1:3])


@pytest.mark.parametrize("ep, dtype", [("cuda", FP32), ("cpu", FP16)])
def test_speech_encoder_refuses_attention_without_a_window(tmp_path, ep, dtype):
    with pytest.raises(NotImplementedError, match="needs GroupQueryAttention"):
        LFM3AudioEncoder(stand_in_config(tmp_path), dtype, dtype, ep, str(tmp_path / "cache"), {})


@pytest.mark.parametrize(
    "ep, io_dtype, onnx_dtype, speech_dtype",
    [
        ("cpu", FP32, FP32, FP32),
        # int4 quantizes the decoder only
        ("cpu", FP32, INT4, FP32),
        # CUDA has no fp32 GroupQueryAttention
        ("cuda", FP16, FP16, FP16),
        ("cuda", BF16, BF16, FP16),
        ("webgpu", FP16, FP16, FP32),
    ],
)
def test_precision_and_quantization_reach_only_the_decoder(tmp_path, ep, io_dtype, onnx_dtype, speech_dtype):
    model, hf_model = build_pipeline(tmp_path, ep, io_dtype, onnx_dtype)

    # The decoder is the one LFM3Model makes on its own
    decoder = builder_module.LFM3Model(
        stand_in_config(tmp_path), io_dtype, onnx_dtype, ep, str(tmp_path / "cache"), {"exclude_embeds": True}
    )
    decoder.load_weights = lambda input_path: hf_model
    decoder.make_model("")
    assert [(node.name, node.op_type) for node in model.decoder.model.graph] == [
        (node.name, node.op_type) for node in decoder.model.graph
    ]

    # The build writes these models and no vision model or image processor
    assert {path.name for path in (tmp_path / "out").glob("*.onnx")} == {
        "model.onnx",
        "embeddings.onnx",
        "audio_encoder.onnx",
    }
    speech = onnx.load(tmp_path / "out" / "audio_encoder.onnx", load_external_data=False).graph
    assert [value.type.tensor_type.elem_type for value in (*speech.input, *speech.output)] == [speech_dtype] * 2
    assert {tensor.data_type for tensor in speech.initializer} == {speech_dtype}
    assert not any(node.op_type == "MatMulNBits" for node in speech.node)

    embedding_proto = onnx.load(tmp_path / "out" / "embeddings.onnx", load_external_data=False)
    check_node_shapes(embedding_proto)
    embedding = embedding_proto.graph
    # The embedding model reads the features in the encoder's type and casts them to its own
    assert [(value.name, value.type.tensor_type.elem_type) for value in embedding.input] == [
        ("input_ids", ir.DataType.INT64),
        ("audio_features", speech_dtype),
    ]
    assert embedding.output[0].type.tensor_type.elem_type == io_dtype
    assert {tensor.data_type for tensor in embedding.initializer if tensor.name == "model.embed_tokens.weight"} == {
        io_dtype
    }
    assert sum(node.op_type == "Cast" for node in embedding.node) == (speech_dtype != io_dtype)
    assert not any(node.op_type == "MatMulNBits" for node in embedding.node)
    decoder_nodes = onnx.load(tmp_path / "out" / "model.onnx", load_external_data=False).graph.node
    assert any(node.op_type == "MatMulNBits" for node in decoder_nodes) == (onnx_dtype == INT4)


@pytest.mark.parametrize(
    "exclude_embeds, builder_name, model_type",
    [(True, "LFM3AudioModel", "lfm3_audio"), (False, "LFM3Model", "lfm3")],
)
def test_architecture_picks_the_pipeline_or_the_text_model(
    monkeypatch, tmp_path, exclude_embeds, builder_name, model_type
):
    config = stand_in_config(tmp_path)
    write_extractor(tmp_path)
    captured = {}

    def recording(parent):
        class Recording(parent):
            def make_genai_config(self, *args):
                is_pipeline = isinstance(self, LFM3AudioModel)
                parts = (self.embedding, self.decoder) if is_pipeline else (self,)
                for part in parts:
                    part.resolve_transformers_class = lambda class_name: class_name
                captured["builder"] = parent.__name__
                captured["loaders"] = {part.resolve_model_loader() for part in parts}
                captured["model_type"] = (self.decoder if is_pipeline else self).model_type

            def save_processing(self, *args):
                pass

        return Recording

    monkeypatch.setattr(builder_module, "LFM3AudioModel", recording(LFM3AudioModel))
    monkeypatch.setattr(builder_module, "LFM3Model", recording(builder_module.LFM3Model))
    builder_module.create_model(
        config.name_or_path,
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "fp32",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        exclude_embeds=exclude_embeds,
        hf_details={"extra_kwargs": {}, "hf_name": config.name_or_path, "hf_config": config},
    )

    assert captured == {
        "builder": builder_name,
        "loaders": {"AutoModelForMultimodalLM"},
        "model_type": model_type,
    }


def make_genai_config(tmp_path, ep="cpu", io_dtype=FP32, **options):
    model = make_pipeline(tmp_path, ep, io_dtype, **options)
    (tmp_path / "out").mkdir()
    model.make_genai_config(stand_in_config(tmp_path), {}, str(tmp_path / "out"))
    return json.loads((tmp_path / "out" / "genai_config.json").read_text(encoding="utf-8"))["model"]


def test_genai_config_sections_match_the_runtime_fixture(tmp_path):
    genai_config = make_genai_config(tmp_path)
    fixture = json.loads((FIXTURE / "genai_config.json").read_text(encoding="utf-8"))["model"]

    assert genai_config["type"] == fixture["type"] == "lfm3_audio"
    assert genai_config["decoder"]["filename"] == "model.onnx"
    assert genai_config["decoder"]["layer_types"] == LAYER_TYPES
    assert genai_config["embedding"] == dict(fixture["embedding"], filename="embeddings.onnx")
    assert genai_config["speech"] == dict(fixture["speech"], filename="audio_encoder.onnx")
    assert genai_config["audio_token_id"] == AUDIO_TOKEN
    # The settings of the released checkpoint's preprocessor_config.json
    mel = ("sample_rate", "num_mels", "fft_size", "win_length", "hop_length", "subsampling_factor")
    assert [genai_config[key] for key in mel] == [16000, 128, 400, 400, 160, 8]
    # What follows these tokens is speech, which this runtime cannot generate for LFM3
    assert genai_config["eos_token_id"] == [EOS, TEXT_END, AUDIO_START, MIXED_START]
    assert not (tmp_path / "out" / "processor_config.json").exists()
    assert "vision" not in genai_config


@pytest.mark.parametrize("ep, io_dtype", [("cpu", FP32), ("cuda", FP16), ("webgpu", FP32)])
def test_speech_section_names_the_encoder_the_build_writes(tmp_path, ep, io_dtype):
    model, _ = build_pipeline(tmp_path, ep, io_dtype)
    model.make_genai_config(stand_in_config(tmp_path), {}, str(tmp_path / "out"))
    genai_config = json.loads((tmp_path / "out" / "genai_config.json").read_text(encoding="utf-8"))["model"]
    assert genai_config["speech"]["filename"] == "audio_encoder.onnx"
    assert (tmp_path / "out" / "audio_encoder.onnx").exists()


@pytest.mark.parametrize(
    "ep, extra_options, decoder_options, embedding_options, speech_options",
    [
        (
            "cuda",
            {},
            {"cuda": {"enable_cuda_graph": "0"}},
            {"cuda": {}},
            {"cuda": {"cudnn_conv_algo_search": "HEURISTIC"}},
        ),
        (
            "cuda",
            {"enable_cuda_graph": True},
            {"cuda": {"enable_cuda_graph": "1"}},
            {"cuda": {}},
            {"cuda": {"cudnn_conv_algo_search": "HEURISTIC"}},
        ),
        (
            "webgpu",
            {"enable_webgpu_graph": True},
            {"webgpu": {"enableGraphCapture": "1", "validationMode": "disabled"}},
            {"webgpu": {"validationMode": "disabled"}},
            {"webgpu": {"validationMode": "disabled"}},
        ),
    ],
)
def test_sub_models_follow_the_decoder_without_graph_capture(
    tmp_path, ep, extra_options, decoder_options, embedding_options, speech_options
):
    genai_config = make_genai_config(tmp_path, ep, FP16, **extra_options)
    assert genai_config["decoder"]["session_options"]["provider_options"] == [decoder_options]
    assert genai_config["embedding"]["session_options"]["provider_options"] == [embedding_options]
    assert genai_config["speech"]["session_options"]["provider_options"] == [speech_options]


@pytest.mark.parametrize(
    "case, error, message",
    [
        ({"ep": "dml", "dtype": FP16}, NotImplementedError, "not dml"),
        ({"ep": "trt-rtx", "dtype": FP16}, NotImplementedError, "not trt-rtx"),
        ({"ep": "cuda", "dtype": FP32}, NotImplementedError, "selects Attention"),
        ({"dtype": FP16}, NotImplementedError, "selects MultiHeadAttention"),
        ({"ep": "cuda", "dtype": FP16, "options": {"use_paged_attention": True}}, NotImplementedError, "use_paged"),
        ({"options": {"filename": "embeddings.onnx"}}, ValueError, "cannot be named embeddings.onnx"),
        ({"options": {"filename": "audio_encoder.onnx"}}, ValueError, "cannot be named audio_encoder.onnx"),
        ({"extractor": {"feature_extractor_type": "WhisperFeatureExtractor"}}, ValueError, "feature_extractor_type"),
        ({"extractor": {"log_floor": -10.0}}, ValueError, "log_floor"),
        ({"extractor": {"normalization_offset": 3.0}}, ValueError, "normalization_offset"),
        ({"extractor": {"normalization_divisor": 2.0}}, ValueError, "normalization_divisor"),
        ({"extractor": {"padding_side": "left"}}, ValueError, "padding_side"),
        ({"extractor": {"padding_value": -1.0}}, ValueError, "padding_value"),
        ({"extractor": {"win_length": 320}}, ValueError, "win_length"),
        ({"extractor": {"feature_size": 80}}, ValueError, "feature_size"),
        ({"extractor": {"encoder_stride": 4}}, ValueError, "encoder_stride"),
        ({"extractor": {"log_floor": None}}, ValueError, "log_floor"),
        ({"extractor": {"n_fft": None}}, KeyError, "n_fft"),
        ({"extractor": {"hop_length": None}}, KeyError, "hop_length"),
        ({"extractor": {"sampling_rate": None}}, KeyError, "sampling_rate"),
    ],
)
def test_refuses_before_building(tmp_path, case, error, message):
    config = stand_in_config(tmp_path)
    write_extractor(tmp_path, **case.get("extractor", {}))
    dtype = case.get("dtype", FP32)
    options = {"exclude_embeds": True, **case.get("options", {})}
    with pytest.raises(error, match=message):
        LFM3AudioModel(config, dtype, dtype, case.get("ep", "cpu"), str(tmp_path / "cache"), options)
