# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for the DecoderState input_ids injection fix (PR #2148).

Bug: DecoderState constructor checked *combined* session_info_ (decoder + vision +
embedding) for HasInput('input_ids').  The embedding session always declares
input_ids, so the check incorrectly injected input_ids into the decoder for models
like Mistral3 whose decoder has no input_ids input — causing an ORT error
"Invalid Feed Input Name: input_ids".

Fix: use a decoder-only SessionInfo for the HasInput('input_ids') check.

Two test model variants (in test/models/):
  - multimodal-decoder-no-input-ids/   Mistral3-like: embedding has input_ids,
                                        decoder does NOT.
  - multimodal-decoder-with-input-ids/ Gemma4-like:   both embedding and decoder
                                        declare input_ids.

Temporary partitioned Gemma3 graphs also check host image injection and decoder
token-type markers during prefill and successive decoding.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


def _run_text_generation(
    model_path: str, prompt: np.ndarray | None = None, chunk_size: int | None = None
) -> list[np.ndarray]:
    """Load the model and run one round of greedy text generation (text-only, no image).

    Appends a seed prompt then generates up to max_length tokens.
    Raises if the underlying ORT sessions receive an unexpected input feed.
    """
    if prompt is None:
        prompt = np.array([[2]], dtype=np.int32)

    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    options = {"do_sample": False, "max_length": 8, "batch_size": prompt.shape[0]}
    if chunk_size is not None:
        options["chunk_size"] = chunk_size
    params.set_search_options(**options)

    generator = og.Generator(model, params)
    generator.append_tokens(prompt)

    while not generator.is_done():
        generator.generate_next_token()

    return [generator.get_sequence(i).copy() for i in range(prompt.shape[0])]


@pytest.mark.parametrize("relative_model_path", [Path("multimodal-decoder-no-input-ids")])
def test_decoder_no_input_ids_does_not_inject_input_ids(test_data_path, relative_model_path):
    """Mistral3-like model: decoder declares no input_ids input.

    With the fix, DecoderState uses decoder-only SessionInfo and does NOT inject
    input_ids into decoder feeds.  Generation must succeed.

    Without the fix, DecoderState would use combined session_info_ (which includes
    the embedding session that always has input_ids) and incorrectly inject input_ids
    into the decoder, causing ORT to raise "Invalid Feed Input Name: input_ids".
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # Should not raise — decoder receives only the inputs it declared
    _run_text_generation(model_path)


@pytest.mark.parametrize("relative_model_path", [Path("multimodal-decoder-with-input-ids")])
def test_decoder_with_input_ids_receives_input_ids(test_data_path, relative_model_path):
    """Gemma4-like model: decoder declares input_ids as one of its inputs.

    With the fix, DecoderState uses decoder-only SessionInfo and correctly injects
    input_ids into decoder feeds because the decoder session declares it.
    Generation must succeed.
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # Should not raise — decoder receives input_ids because it declared it
    _run_text_generation(model_path)


@pytest.mark.parametrize(
    "relative_model_path",
    [
        Path("multimodal-decoder-no-input-ids"),
        Path("multimodal-decoder-with-input-ids"),
    ],
)
def test_multimodal_prefill_chunking_matches_unchunked_generation(test_data_path, relative_model_path):
    """Chunking must preserve generation for both decoder input_ids contracts."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # A five-token prompt with chunk_size=2 exercises two full chunks, a final
    # partial chunk, and the decode step after the embedding view is restored.
    prompt = np.array([[2, 3, 4, 5, 6]], dtype=np.int32)
    unchunked = _run_text_generation(model_path, prompt)
    chunked = _run_text_generation(model_path, prompt, chunk_size=2)

    assert len(chunked) == len(unchunked)
    for actual, expected in zip(chunked, unchunked, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_multimodal_prefill_chunking_falls_back_for_batched_prompts(test_data_path):
    """A batch of embeddings is non-contiguous per sequence and must not be sliced."""
    model_path = os.fspath(Path(test_data_path) / "multimodal-decoder-with-input-ids")
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    prompt = np.array([[2, 3, 4, 5], [6, 5, 4, 3]], dtype=np.int32)
    unchunked = _run_text_generation(model_path, prompt)
    with_chunk_size = _run_text_generation(model_path, prompt, chunk_size=2)

    assert len(with_chunk_size) == len(unchunked)
    for actual, expected in zip(with_chunk_size, unchunked, strict=True):
        np.testing.assert_array_equal(actual, expected)


def _write_gemma3_partitioned_fixture(test_data_path, model_path, marker_dtype):
    onnx = pytest.importorskip("onnx")
    source = Path(test_data_path) / "multimodal-decoder-no-input-ids"
    model_path.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, model_path / name)
    helper = onnx.helper
    tp = onnx.TensorProto

    def value(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    def save(name, inputs, outputs, nodes, constants):
        graph = helper.make_graph(
            nodes, name, inputs, outputs, [onnx.numpy_helper.from_array(v, k) for k, v in constants.items()]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, model_path / f"{name}.onnx")

    save(
        "vision",
        [value("pixel_values", tp.FLOAT, ["R", 4])],
        [value("image_features", tp.FLOAT, ["R", 4])],
        [
            helper.make_node("Mul", ["pixel_values", "scale"], ["scaled_pixels"]),
            helper.make_node("Add", ["scaled_pixels", "bias"], ["image_features"]),
        ],
        {"scale": np.array(2, dtype=np.float32), "bias": np.array(0.5, dtype=np.float32)},
    )
    save(
        "embedding",
        [value("input_ids", tp.INT32, ["B", "S"])],
        [value("inputs_embeds", tp.FLOAT, ["B", "S", 4])],
        [
            helper.make_node("Cast", ["input_ids"], ["float_ids"], to=tp.FLOAT),
            helper.make_node("Unsqueeze", ["float_ids", "last_axis"], ["expanded_ids"]),
            helper.make_node("Add", ["expanded_ids", "offset"], ["inputs_embeds"]),
        ],
        {"last_axis": np.array([2], dtype=np.int64), "offset": (0.1 * np.arange(4)).astype(np.float32)},
    )
    # Echo every marker and embedding row, not just the logits used for sampling.
    save(
        "decoder",
        [
            value("inputs_embeds", tp.FLOAT, ["B", "S", 4]),
            value("attention_mask", tp.INT64, ["B", "totalS"]),
            value("position_ids", tp.INT64, ["B", "S"]),
            value("decoder_markers", tp.INT32 if marker_dtype == np.int32 else tp.INT64, ["B", "S"]),
        ],
        [value("logits", tp.FLOAT, ["B", "S", 10])],
        [
            helper.make_node("Cast", ["decoder_markers"], ["float_markers"], to=tp.FLOAT),
            helper.make_node("Unsqueeze", ["float_markers", "last_axis"], ["marker_column"]),
            helper.make_node("Mul", ["inputs_embeds", "zero"], ["padding"]),
            helper.make_node("Mul", ["marker_column", "zero"], ["zero_column"]),
            helper.make_node("Add", ["zero_column", "winner"], ["winning_column"]),
            helper.make_node(
                "Concat", ["marker_column", "inputs_embeds", "padding", "winning_column"], ["logits"], axis=2
            ),
        ],
        {
            "last_axis": np.array([2], dtype=np.int64),
            "zero": np.array(0, dtype=np.float32),
            "winner": np.array(1000, dtype=np.float32),
        },
    )
    config = {
        "model": {
            "type": "gemma3",
            "partitioning": "openvino",
            "bos_token_id": 1,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "image_token_id": 3,
            "vocab_size": 10,
            "context_length": 64,
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": 4,
                "head_size": 4,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "token_type_ids": "decoder_markers",
                },
                "outputs": {"logits": "logits"},
                "session_options": {"provider_options": []},
            },
            "embedding": {
                "filename": "embedding.onnx",
                "inputs": {"input_ids": "input_ids"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
                "session_options": {"provider_options": []},
            },
            "vision": {
                "filename": "vision.onnx",
                "inputs": {"pixel_values": "pixel_values"},
                "outputs": {"image_features": "image_features"},
                "session_options": {"provider_options": []},
            },
        },
        "search": {"do_sample": False, "num_beams": 1, "max_length": 9, "past_present_share_buffer": False},
    }
    (model_path / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")


def _assert_gemma3_partitioned_generation(generator, expected_embeddings, expected_markers):
    for step in range(3):
        generator.get_logits()
        embeddings = generator.get_output("inputs_embeds").copy()
        logits = generator.get_output("logits").copy()
        assert embeddings.shape == expected_embeddings.shape
        assert logits.shape == (*expected_markers.shape, 10)
        assert embeddings.dtype == logits.dtype == np.float32
        np.testing.assert_allclose(embeddings, expected_embeddings, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(logits[:, :, 0], expected_markers)
        expected_logits = np.zeros(logits.shape, dtype=np.float64)
        expected_logits[:, :, 0] = expected_markers
        expected_logits[:, :, 1:5] = expected_embeddings
        expected_logits[:, :, 9] = 1000
        np.testing.assert_allclose(logits, expected_logits, rtol=1e-6, atol=1e-6)
        print(
            f"step={step} embeddings={embeddings.tolist()} shape={embeddings.shape} dtype={embeddings.dtype} "
            f"logits_shape={logits.shape} markers={logits[:, :, 0].tolist()} "
            f"max_abs={np.max(np.abs(embeddings - expected_embeddings)):.9g}"
        )
        generator.generate_next_token()
        np.testing.assert_array_equal(generator.get_next_tokens(), [9])
        expected_embeddings = np.array([[[9, 9.1, 9.2, 9.3]]], dtype=np.float64)
        expected_markers = np.zeros((1, 1), dtype=np.int64)


@pytest.mark.parametrize("marker_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_gemma3_partitioned_image_prefill_and_decode(test_data_path, tmp_path, marker_dtype, chunk_size):
    model_path = tmp_path / "gemma3-partitioned"
    _write_gemma3_partitioned_fixture(test_data_path, model_path, marker_dtype)
    prompt = np.array([[2, 3, 4, 5, 6]], dtype=np.int32)
    markers = np.array([[0, 1, 1, 0, 0]], dtype=marker_dtype)
    inputs = og.NamedTensors()
    inputs["input_ids"] = og.Tensor(prompt)
    inputs["token_type_ids"] = og.Tensor(markers)
    inputs["pixel_values"] = og.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
    inputs["num_image_tokens"] = og.Tensor(np.array([2], dtype=np.int64))
    model = og.Model(os.fspath(model_path))
    params = og.GeneratorParams(model)
    if chunk_size is not None:
        params.set_search_options(chunk_size=chunk_size)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    expected = prompt[:, :, None].astype(np.float64) + 0.1 * np.arange(4)
    expected[0, 1:3] = [[2.5, 4.5, 6.5, 8.5], [10.5, 12.5, 14.5, 16.5]]
    _assert_gemma3_partitioned_generation(generator, expected, markers)


@pytest.mark.parametrize("marker_dtype", [np.int32, np.int64])
def test_gemma3_partitioned_text_only_and_decode(test_data_path, tmp_path, marker_dtype):
    model_path = tmp_path / "gemma3-partitioned"
    _write_gemma3_partitioned_fixture(test_data_path, model_path, marker_dtype)
    prompt = np.array([[2, 3, 4, 5, 6]], dtype=np.int32)
    inputs = og.NamedTensors()
    inputs["input_ids"] = og.Tensor(prompt)
    model = og.Model(os.fspath(model_path))
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    expected = prompt[:, :, None].astype(np.float64) + 0.1 * np.arange(4)
    _assert_gemma3_partitioned_generation(generator, expected, np.zeros(prompt.shape, dtype=marker_dtype))


@pytest.mark.parametrize("vision_filename", ["", "vision.onnx"])
def test_gemma3_partitioned_rejects_vision_pipeline(test_data_path, tmp_path, vision_filename):
    model_path = tmp_path / "gemma3-partitioned"
    _write_gemma3_partitioned_fixture(test_data_path, model_path, np.int64)
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["vision"]["filename"] = vision_filename
    config["model"]["vision"]["pipeline"] = [
        {
            "vision": {
                "filename": "vision.onnx",
                "inputs": ["pixel_values"],
                "outputs": ["image_features"],
                "run_on_cpu": True,
            }
        }
    ]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError):
        og.Model(os.fspath(model_path))
