# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""End-to-end RewindTo(0) coverage for multimodal generators.

MultiModalPipelineState::RewindTo() lets a multimodal prompt be replayed
Each test runs a prompt, generates, rewinds to zero, replays
the same prompt, and requires the replay to match both the first run and a
separately constructed generator.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

MAX_LENGTH = 8
STEPS = 3
REPLAYS = 3

QWEN_PROMPT = [151652, 151655, 10, 20]  # vision_start, image_pad, text, text
PHI3V_PROMPT = [2, 3, 4]


def _model_path(test_data_path, model_name):
    path = os.fspath(Path(test_data_path) / model_name)
    if not os.path.exists(path):
        pytest.skip(f"Test model not found: {path}")
    return path


def _qwen_inputs():
    inputs = og.NamedTensors()
    inputs["input_ids"] = np.asarray([QWEN_PROMPT], dtype=np.int32)
    inputs["pixel_values"] = np.zeros((4, 1536), dtype=np.float32)
    inputs["image_grid_thw"] = np.asarray([[1, 2, 2]], dtype=np.int64)
    inputs["num_image_tokens"] = np.asarray([1], dtype=np.int64)
    return inputs


def _phi3v_inputs():
    inputs = og.NamedTensors()
    inputs["input_ids"] = np.asarray([PHI3V_PROMPT], dtype=np.int32)
    inputs["pixel_values"] = np.zeros((1, 1, 3, 8, 8), dtype=np.float32)
    inputs["image_sizes"] = np.asarray([[8, 8]], dtype=np.int64)
    inputs["num_image_tokens"] = np.asarray([1], dtype=np.int64)
    return inputs


def _new_generator(model):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=MAX_LENGTH)
    return og.Generator(model, params)


def _run(generator):
    """Drive a prefilled generator to STEPS tokens, capturing what is observable.

    image_features is read first: the prompt step is the only point at which the
    embedding model holds it, since the next step shrinks it to zero tokens.
    """
    features = generator.get_input("image_features")
    logits = [float(generator.get_logits().reshape(-1)[0])]
    for _ in range(STEPS):
        generator.generate_next_token()
        logits.append(float(generator.get_logits().reshape(-1)[0]))
    return features, logits, generator.get_sequence(0).tolist()


def _assert_matches(actual, expected, label):
    features, logits, sequence = actual
    expected_features, expected_logits, expected_sequence = expected
    assert sequence == expected_sequence, f"{label}: token sequence differs"
    assert logits == expected_logits, f"{label}: logits differ"
    np.testing.assert_array_equal(features, expected_features, err_msg=f"{label}: image_features differ")


@pytest.mark.parametrize(
    "model_name, prompt, make_inputs, feature_shape, feature_value",
    [
        ("qwen3-5", QWEN_PROMPT, _qwen_inputs, (1, 1024), 0.01),
        ("multimodal-decoder-with-input-ids", PHI3V_PROMPT, _phi3v_inputs, (1, 64), 0.0),
        ("multimodal-decoder-no-input-ids", PHI3V_PROMPT, _phi3v_inputs, (1, 64), 0.0),
    ],
)
def test_rewind_to_zero_replays_image_prompt(
    test_data_path, model_name, prompt, make_inputs, feature_shape, feature_value
):
    model = og.Model(_model_path(test_data_path, model_name))

    generator = _new_generator(model)
    generator.set_inputs(make_inputs())
    baseline = _run(generator)

    # The encoder's own output, not an empty or uninitialized buffer.
    assert baseline[0].shape == feature_shape
    np.testing.assert_allclose(baseline[0], feature_value, rtol=1e-6)

    for replay in range(REPLAYS):
        generator.rewind_to(0)
        generator.append_tokens(prompt)
        _assert_matches(_run(generator), baseline, f"replay {replay}")

    fresh = _new_generator(model)
    fresh.set_inputs(make_inputs())
    _assert_matches(_run(fresh), baseline, "fresh generator")


@pytest.mark.parametrize(
    "model_name, prompt, make_inputs",
    [
        ("qwen3-5", QWEN_PROMPT, _qwen_inputs),
        ("multimodal-decoder-with-input-ids", PHI3V_PROMPT, _phi3v_inputs),
        ("multimodal-decoder-no-input-ids", PHI3V_PROMPT, _phi3v_inputs),
    ],
)
def test_rewind_inside_prompt_raises_and_leaves_state_usable(test_data_path, model_name, prompt, make_inputs):
    """RewindSplitsPrompt rejects 0 < new_length < prompt_length; must raise, not corrupt state."""
    model = og.Model(_model_path(test_data_path, model_name))

    generator = _new_generator(model)
    generator.set_inputs(make_inputs())
    baseline = _run(generator)

    for new_length in range(1, len(prompt)):
        with pytest.raises(RuntimeError, match="Cannot rewind to a length inside the prompt"):
            generator.rewind_to(new_length)

    # The rejected calls above must not have mutated state, kv cache, or position ids.
    generator.rewind_to(0)
    generator.append_tokens(prompt)
    _assert_matches(_run(generator), baseline, "replay after rejected mid-prompt rewinds")


def test_rewind_into_generated_continuation_rejected_without_snapshot(test_data_path):
    """A nonzero rewind with no matching recurrent-state snapshot must raise, not corrupt state."""
    model = og.Model(_model_path(test_data_path, "qwen3-5"))

    generator = _new_generator(model)
    generator.set_inputs(_qwen_inputs())
    baseline = _run(generator)  # sequence length == len(QWEN_PROMPT) + STEPS == 7

    mid_continuation = len(QWEN_PROMPT) + 1  # > prompt_length, < current_length: legal per RewindSplitsPrompt
    with pytest.raises(RuntimeError, match="does not support rewinding to that length"):
        generator.rewind_to(mid_continuation)

    # The rejected call above must not have mutated the sequence, kv cache, or position ids.
    generator.rewind_to(0)
    generator.append_tokens(QWEN_PROMPT)
    _assert_matches(_run(generator), baseline, "replay after rejected mid-continuation rewind")


def test_snapshot_state_enables_rewind_into_generated_continuation(test_data_path):
    """snapshot_state() must reach RecurrentState::Snapshot through DecoderState/
    MultiModalPipelineState, or CanRewindTo(index) can never become true for qwen3-5."""
    model = og.Model(_model_path(test_data_path, "qwen3-5"))

    generator = _new_generator(model)
    generator.set_inputs(_qwen_inputs())
    generator.get_logits()  # prompt step; sequence length == len(QWEN_PROMPT)

    snapshot_length = len(QWEN_PROMPT) + 1
    generator.generate_next_token()  # sequence length == snapshot_length
    assert generator.get_sequence(0).shape[0] == snapshot_length
    generator.snapshot_state()

    generator.generate_next_token()  # sequence length == snapshot_length + 1
    expected_logits = generator.get_logits().copy()  # matches the replay's Run() count below
    generator.generate_next_token()  # sequence length == snapshot_length + 2
    expected_sequence = generator.get_sequence(0).copy()

    for _ in range(REPLAYS):
        generator.rewind_to(snapshot_length)
        assert generator.get_sequence(0).shape[0] == snapshot_length
        generator.generate_next_token()
        np.testing.assert_array_equal(generator.get_sequence(0), expected_sequence)
        np.testing.assert_array_equal(generator.get_logits(), expected_logits)


@pytest.mark.parametrize(
    "model_name",
    ["multimodal-decoder-with-input-ids", "multimodal-decoder-no-input-ids"],
)
def test_rewind_to_zero_replays_text_only_prompt(test_data_path, model_name):
    """No image, so image_features is the empty [0, hidden] tensor that
    AllocateEmptyFeatures supplies rather than an encoder result."""
    model = og.Model(_model_path(test_data_path, model_name))

    generator = _new_generator(model)
    generator.append_tokens(PHI3V_PROMPT)
    baseline = _run(generator)
    assert baseline[0].shape == (0, 64)

    for replay in range(REPLAYS):
        generator.rewind_to(0)
        generator.append_tokens(PHI3V_PROMPT)
        _assert_matches(_run(generator), baseline, f"replay {replay}")

    fresh = _new_generator(model)
    fresh.append_tokens(PHI3V_PROMPT)
    _assert_matches(_run(fresh), baseline, "fresh generator")
