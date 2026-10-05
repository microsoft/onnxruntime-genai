# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for the Gemma4 "unified" (encoder-free, gemma-4-12B) multimodal model.

Unlike the standard gemma4 model, the unified variant consumes raw 48px merged
pixel patches (``pixel_values`` last dim = 6912) and raw 640-sample waveform
frames (``audio_embeds`` last dim = 640) directly, and its ``model.type`` is
``gemma4_unified``. These tests exercise the ``unified_`` branch of
``Gemma4MultiModalProcessor``: processor creation, and that the produced
pixel_values / audio_embeds / audio_sizes follow the unified contract.
Input-sensitive stand-in graphs also verify native modality-to-decoder handoff
by observing logits before and after controlled processor-tensor mutations.

The tests derive temporary unified fixtures from tracked ``test/models/gemma4``
assets using ``test/python/create/create_dummy_gemma4_unified_models.py``.

Usage:
    pytest test_gemma4_unified_models.py --test_models=/path/to/models
"""

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_genai as og
import pytest
from create.create_dummy_gemma4_unified_models import create_model
from test_gemma4_models import (
    _create_dynamic_decoder_model,
    _create_dynamic_embedding_model,
    _set_vision_position_dtype,
)

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG
)
log = logging.getLogger("gemma4-unified-tests")

GEMMA4_UNIFIED_MODEL_NAME = "gemma4_unified"

# Encoder-free unified contract.
_UNIFIED_PIXEL_DIM = 48 * 48 * 3  # 6912
_UNIFIED_AUDIO_DIM = 640


@pytest.fixture(scope="module")
def unified_model_path(request, tmp_path_factory):
    """Reuse tracked Gemma4 assets, deriving all unified files in a temp dir."""
    test_data_path = request.config.getoption("--test_models")
    assert test_data_path, "--test_models is required for Gemma4 validation"
    source = Path(test_data_path) / "gemma4"
    assert source.is_dir(), f"Missing tracked Gemma4 fixture: {source}"
    destination = tmp_path_factory.mktemp("unified") / GEMMA4_UNIFIED_MODEL_NAME
    create_model(source, destination)
    return destination


def _load_model_and_processor(model_path):
    model = og.Model(os.fspath(model_path))
    return model, model.create_multimodal_processor()


def _media_path(test_data_path, relative_path):
    # Main's media layout is test/{images,audios}, not test/models/{images,audios}.
    candidates = [
        Path(test_data_path).parent / relative_path,
        Path(__file__).resolve().parents[2] / relative_path,
    ]
    for path in candidates:
        if path.is_file():
            return os.fspath(path)
    pytest.fail(f"Missing validation media; checked {candidates}")


def _to_numpy(tensor):
    if hasattr(tensor, "as_numpy"):
        return tensor.as_numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.array(tensor)


def test_gemma4_unified_model_load(unified_model_path):
    """The gemma4_unified model (model.type == 'gemma4_unified') loads."""
    model = og.Model(os.fspath(unified_model_path))
    assert model is not None


def test_gemma4_unified_processor_creation(unified_model_path):
    """create_multimodal_processor() succeeds for the gemma4_unified type.

    This exercises the processor-factory registration and the unified image /
    audio ort-extensions configs (Gemma4ImageTransform at patch_size=48 and
    Gemma4UnifiedAudioFrames).
    """
    _, processor = _load_model_and_processor(unified_model_path)
    assert processor is not None


def test_gemma4_unified_model_io_contract(unified_model_path):
    """The dummy graph inputs use the encoder-free unified names and shapes."""
    model_path = Path(unified_model_path)
    with open(model_path / "genai_config.json") as config_file:
        config = json.load(config_file)
    speech_config_inputs = config["model"]["speech"]["inputs"]
    assert speech_config_inputs == {
        "audio_embeds": "input_features",
        "attention_mask": "input_features_mask",
        "audio_sizes": "audio_sizes",
    }

    vision = onnx.load(model_path / "dummy_vision.onnx")
    vision_inputs = {
        value.name: (
            value.type.tensor_type.elem_type,
            [
                dim.dim_value or dim.dim_param
                for dim in value.type.tensor_type.shape.dim
            ],
        )
        for value in vision.graph.input
    }
    assert vision_inputs == {
        "pixel_values": (
            onnx.TensorProto.FLOAT,
            ["batch_size", 280, _UNIFIED_PIXEL_DIM],
        ),
        "pixel_position_ids": (onnx.TensorProto.INT64, ["batch_size", 280, 2]),
    }

    speech = onnx.load(model_path / "dummy_speech.onnx")
    speech_graph_inputs = {
        value.name: (
            value.type.tensor_type.elem_type,
            [
                dim.dim_value or dim.dim_param
                for dim in value.type.tensor_type.shape.dim
            ],
        )
        for value in speech.graph.input
    }
    assert speech_graph_inputs == {
        "input_features": (
            onnx.TensorProto.FLOAT,
            ["batch_size", "num_frames", _UNIFIED_AUDIO_DIM],
        ),
        "audio_sizes": (onnx.TensorProto.INT64, ["batch_size"]),
        "input_features_mask": (onnx.TensorProto.BOOL, ["batch_size", "num_frames"]),
    }


def test_gemma4_unified_text_only(unified_model_path):
    """Text-only processing (no images/audio)."""
    _, processor = _load_model_and_processor(unified_model_path)
    inputs = processor("What is the capital of France?", images=None)
    assert inputs is not None
    assert "input_ids" in inputs
    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2 and ids.shape[0] == 1


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_unified_vision_contract(
    test_data_path, unified_model_path, relative_image_path
):
    """Unified vision preprocessing produces 6912-dim merged patches (no trim)."""
    _, processor = _load_model_and_processor(unified_model_path)

    image_path = _media_path(test_data_path, relative_image_path)
    images = og.Images.open(image_path)

    inputs = processor("<|image|>Describe this image", images=images)
    assert inputs is not None
    assert "pixel_values" in inputs
    assert "pixel_position_ids" in inputs

    pixel_values = _to_numpy(inputs["pixel_values"])
    assert (
        pixel_values.shape[-1] == _UNIFIED_PIXEL_DIM
    ), f"unified pixel_values feature dim should be {_UNIFIED_PIXEL_DIM}, got {pixel_values.shape[-1]}"
    # Unified feeds the full padded patch grid (no trim); the graph strips
    # padding via position_ids. pixel_position_ids rows must match pixel_values.
    pos = _to_numpy(inputs["pixel_position_ids"])
    assert (
        pos.shape[-2] == pixel_values.shape[-2]
    ), f"position_ids ({pos.shape}) and pixel_values ({pixel_values.shape}) patch counts must match"
    assert pos.shape[-1] == 2
    assert pixel_values.shape == (1, 280, _UNIFIED_PIXEL_DIM)
    assert pixel_values.dtype == np.float32
    assert pos.dtype == np.int64
    valid = np.all(pos[0] >= 0, axis=-1)
    counts = _to_numpy(inputs["num_image_tokens"])
    np.testing.assert_array_equal(counts, [valid.sum()])
    assert (
        valid.any() and not valid.all()
    ), "Asset must exercise real patches and padding"
    np.testing.assert_array_equal(
        pos[0, ~valid], -np.ones((np.count_nonzero(~valid), 2), dtype=np.int64)
    )
    np.testing.assert_array_equal(
        pixel_values[0, ~valid], np.zeros_like(pixel_values[0, ~valid])
    )


def test_gemma4_unified_vision_int32_positions(
    test_data_path, unified_model_path, tmp_path
):
    """INT32 positions match the INT64 source, including XY and padded -1 rows."""
    images = og.Images.open(
        _media_path(test_data_path, Path("images") / "australia.jpg")
    )
    _, reference_processor = _load_model_and_processor(unified_model_path)
    reference = reference_processor("<|image|>Describe", images=images)
    expected = _to_numpy(reference["pixel_position_ids"])
    assert np.any(expected > 0), "Exercise nonzero XY coordinates"
    assert np.any(np.all(expected == -1, axis=-1)), "Exercise retained unified padding"
    model_path = tmp_path / "unified_int32"
    shutil.copytree(unified_model_path, model_path)
    _set_vision_position_dtype(
        onnx, model_path / "dummy_vision.onnx", onnx.TensorProto.INT32
    )
    _, processor = _load_model_and_processor(model_path)
    actual = processor("<|image|>Describe", images=images)
    positions = _to_numpy(actual["pixel_position_ids"])
    assert positions.dtype == np.int32
    assert positions.shape == (1, 280, 2)
    np.testing.assert_array_equal(positions, expected)
    for name in ("pixel_values", "num_image_tokens", "input_ids"):
        np.testing.assert_array_equal(
            _to_numpy(actual[name]), _to_numpy(reference[name])
        )


@pytest.mark.parametrize("relative_audio_path", [Path("audios") / "jfk.flac"])
def test_gemma4_unified_audio_contract(
    test_data_path, unified_model_path, relative_audio_path
):
    """Unified audio preprocessing produces raw 640-sample frames; audio_sizes = frame count."""
    _, processor = _load_model_and_processor(unified_model_path)

    audio_path = _media_path(test_data_path, relative_audio_path)

    audios = og.Audios.open(audio_path)
    inputs = processor("<|audio|>Transcribe this audio", audios=audios)
    assert inputs is not None
    assert "audio_embeds" in inputs
    assert "audio_sizes" in inputs

    audio_embeds = _to_numpy(inputs["audio_embeds"])
    assert (
        audio_embeds.shape[-1] == _UNIFIED_AUDIO_DIM
    ), f"unified audio_embeds feature dim should be {_UNIFIED_AUDIO_DIM}, got {audio_embeds.shape[-1]}"
    assert audio_embeds.dtype == np.float32

    # Unified: each 640-sample frame is exactly one audio token, so audio_sizes
    # equals the number of frames (no stride-2 subsampling).
    num_frames = audio_embeds.shape[-2]
    audio_sizes = _to_numpy(inputs["audio_sizes"])
    assert (
        audio_sizes[0] == num_frames
    ), f"unified audio_sizes should equal frame count {num_frames}, got {audio_sizes[0]}"
    assert audio_sizes.dtype == np.int64
    assert audio_sizes.shape == (1,)
    mask = _to_numpy(inputs["audio_attention_mask"])
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.ones((1, num_frames), dtype=np.bool_))


def test_gemma4_unified_audio_multiple_clips_rejected(
    test_data_path, unified_model_path
):
    _, processor = _load_model_and_processor(unified_model_path)
    audio = _media_path(test_data_path, Path("audios") / "jfk.flac")
    with pytest.raises(RuntimeError, match="only 1 audio clip per prompt"):
        processor("<|audio|><|audio|>Transcribe", audios=og.Audios.open(audio, audio))


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_gemma4_unified_modality_sessions_generate(
    test_data_path, unified_model_path, tmp_path, modality
):
    """Execute the real modality -> embedding -> decoder path and advance search."""
    model_path = tmp_path / "model"
    shutil.copytree(unified_model_path, model_path)
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_name = "image_features" if modality == "image" else "audio_features"
    config["model"]["embedding"]["inputs"] = {
        "input_ids": "input_ids",
        feature_name: feature_name,
    }
    extra_feature_names = ()
    if modality == "audio":
        # Gemma4 always has a vision session. Native embedding binding therefore
        # supplies image_features even for audio-only prompts (empty [0, 2048]).
        config["model"]["embedding"]["inputs"]["image_features"] = "image_features"
        extra_feature_names = ("image_features",)
    config["model"]["vocab_size"] = 8
    config["model"]["eos_token_id"] = [1]
    config["search"]["past_present_share_buffer"] = False
    if modality == "image":
        config["model"]["speech"] = {"filename": "", "config_filename": ""}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _create_dynamic_embedding_model(
        onnx,
        model_path / "dummy_embedding.onnx",
        feature_name=feature_name,
        consume_features=True,
        extra_feature_names=extra_feature_names,
    )
    _create_dynamic_decoder_model(
        onnx, model_path / "dummy_text.onnx", consume_embeddings=True
    )

    model, processor = _load_model_and_processor(model_path)
    if modality == "image":
        images = og.Images.open(
            _media_path(test_data_path, Path("images") / "australia.jpg")
        )
        inputs = processor("<|image|>Describe", images=images)
        feed = {
            name: _to_numpy(inputs[name])
            for name in ("pixel_values", "pixel_position_ids")
        }
        session = ort.InferenceSession(
            os.fspath(model_path / "dummy_vision.onnx"),
            providers=["CPUExecutionProvider"],
        )
        features = session.run(None, feed)[0]
        valid = np.all(feed["pixel_position_ids"][0] >= 0, axis=-1)
        expected = feed["pixel_values"][0, valid].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(
            features, np.broadcast_to(expected, features.shape), rtol=1e-5, atol=1e-6
        )
        changed_pixels = feed["pixel_values"].copy()
        changed_pixels[0, np.flatnonzero(valid)[0]] += 0.25
        changed = session.run(None, {**feed, "pixel_values": changed_pixels})[0]
        np.testing.assert_allclose(changed[0] - features[0], 0.25, rtol=0, atol=1e-5)
        changed_positions = feed["pixel_position_ids"].copy()
        changed_positions[0, np.flatnonzero(valid)[0]] = -1
        trimmed = session.run(None, {**feed, "pixel_position_ids": changed_positions})[
            0
        ]
        np.testing.assert_allclose(trimmed, features[1:], rtol=0, atol=0)
    else:
        audios = og.Audios.open(
            _media_path(test_data_path, Path("audios") / "jfk.flac")
        )
        inputs = processor("<|audio|>Transcribe", audios=audios)
        feed = {
            "input_features": _to_numpy(inputs["audio_embeds"]),
            "input_features_mask": _to_numpy(inputs["audio_attention_mask"]),
            "audio_sizes": _to_numpy(inputs["audio_sizes"]),
        }
        session = ort.InferenceSession(
            os.fspath(model_path / "dummy_speech.onnx"),
            providers=["CPUExecutionProvider"],
        )
        features = session.run(None, feed)[0]
        assert features.shape == (1, int(feed["audio_sizes"][0]), 2048)
        expected = (
            feed["input_features"].mean(axis=2) * feed["input_features_mask"]
        ).sum()
        expected += feed["audio_sizes"][0]
        np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)
        # Controlled PCM makes mask consumption observable even if the asset's
        # first frame happens to have zero mean.
        controlled = {**feed, "input_features": np.ones_like(feed["input_features"])}
        all_valid = session.run(None, controlled)[0]
        changed_mask = feed["input_features_mask"].copy()
        changed_mask[0, 0] = False
        masked = session.run(None, {**controlled, "input_features_mask": changed_mask})[
            0
        ]
        np.testing.assert_allclose(all_valid - masked, 1.0, rtol=0, atol=0)
        changed_pcm = session.run(
            None, {**controlled, "input_features": controlled["input_features"] * 2}
        )[0]
        np.testing.assert_allclose(
            changed_pcm - all_valid, feed["input_features"].shape[1], rtol=0, atol=0
        )
        changed_sizes = feed["audio_sizes"] - 1
        counted = session.run(None, {**controlled, "audio_sizes": changed_sizes})[0]
        assert counted.shape == (1, int(changed_sizes[0]), 2048)
        np.testing.assert_allclose(all_valid[:, :-1] - counted, 1.0, rtol=0, atol=0)
        features = features.reshape(
            -1, 2048
        )  # Match native speech -> embedding handoff.

    # Verify that the modality output is consumed, not just declared, by embedding.
    embedding = ort.InferenceSession(
        os.fspath(model_path / "dummy_embedding.onnx"),
        providers=["CPUExecutionProvider"],
    )
    ids = _to_numpy(inputs["input_ids"])
    embedding_feed = {"input_ids": ids, feature_name: features}
    if modality == "audio":
        embedding_feed["image_features"] = np.empty((0, 2048), dtype=np.float32)
    embeds = embedding.run(None, embedding_feed)[0]
    np.testing.assert_allclose(
        embeds, features.mean(dtype=np.float64), rtol=1e-5, atol=1e-5
    )
    changed_embeds = embedding.run(
        None, {**embedding_feed, feature_name: features + 1}
    )[0]
    np.testing.assert_allclose(changed_embeds - embeds, 1.0, rtol=0, atol=1e-4)

    native_signals = []
    for mutate in (False, True):
        if mutate:
            # Change an actual processor tensor before the second native
            # SetInputs, not merely the inputs to the direct ORT self-check.
            if modality == "image":
                changed_input = feed["pixel_values"] + np.float32(1)
                inputs["pixel_values"] = changed_input
                features = session.run(None, {**feed, "pixel_values": changed_input})[0]
            else:
                changed_input = np.ones_like(feed["input_features"])
                inputs["audio_embeds"] = changed_input
                features = session.run(None, {**feed, "input_features": changed_input})[
                    0
                ].reshape(-1, 2048)
        expected_signal = np.float32(features.mean(dtype=np.float64))
        expected_logits = np.zeros(8, dtype=np.float32)
        expected_logits[2] = expected_signal
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=ids.shape[1] + 2, do_sample=False)
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        np.testing.assert_array_equal(generator.get_sequence(0), ids[0])
        # SetInputs runs prefill. Observe its logits before search transforms
        # them or a later decoder step consumes empty modality tensors.
        logits = np.asarray(generator.get_logits()).reshape(-1, 8)
        np.testing.assert_allclose(
            logits, np.broadcast_to(expected_logits, logits.shape), rtol=1e-5, atol=1e-5
        )
        native_signals.append(float(logits[-1, 2]))
        generator.generate_next_token()
        sequence = np.asarray(generator.get_sequence(0))
        assert sequence.size == ids.shape[1] + 1
        np.testing.assert_array_equal(sequence[:-1], ids[0])
        assert sequence[-1] == int(np.argmax(expected_logits))
    assert not np.isclose(
        native_signals[0], native_signals[1]
    ), "Native logits must consume changed modality inputs"
