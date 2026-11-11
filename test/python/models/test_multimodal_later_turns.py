# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""set_inputs after the first prompt: text turns continue the sequence, images and audio are refused.

Multimodal states take extra inputs only once, so images or audio in a later call would be dropped
while their placeholder tokens stay in input_ids.
"""

import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from test_lfm2_audio_models import _model_path, _write_wav

MESSAGE = "Only a generator's first inputs can carry images or audio"
# name: (media prompt, text prompt)
MODELS = {
    "lfm2-vl": ("<image>Describe this image.", "And now?"),
    "qwen3-5": ("<|vision_start|><|image_pad|><|vision_end|>Describe this image.", "And now?"),
    "lfm2-audio": ("Transcribe <|audio|>", "And now?"),
}


def _setup(test_data_path, tmp_path, name):
    """The model, a function that processes the media prompt, and the processed text prompt."""
    media_prompt, text_prompt = MODELS[name]
    model_path = _model_path(test_data_path) if name == "lfm2-audio" else os.fspath(Path(test_data_path) / name)
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    image = os.fspath((Path(test_data_path).parent / "images" / "cars.jpg").resolve())
    clip = _write_wav(tmp_path / "clip.wav", 0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000))

    def media():
        if name == "lfm2-audio":
            return processor(media_prompt, audios=og.Audios.open(clip))
        return processor(media_prompt, images=og.Images.open(image))

    return model, media, processor(text_prompt)


def _generator(model):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=2048)
    return og.Generator(model, params)


@pytest.mark.parametrize("first_turn", ["media", "text"])
@pytest.mark.parametrize("name", list(MODELS))
def test_later_set_inputs_with_media_raises(test_data_path, tmp_path, name, first_turn):
    model, media, text = _setup(test_data_path, tmp_path, name)
    generator = _generator(model)
    generator.set_inputs(media() if first_turn == "media" else text)
    generator.generate_next_token()
    sequence = generator.get_sequence(0).copy()

    with pytest.raises(RuntimeError, match=MESSAGE):
        generator.set_inputs(media())
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)

    # The refused call leaves the generator usable for a text turn.
    generator.set_inputs(text)
    generator.generate_next_token()
    assert len(generator.get_sequence(0)) == len(sequence) + text["input_ids"].as_numpy().shape[1] + 1


@pytest.mark.parametrize("name", list(MODELS))
def test_media_twice_before_generating_raises(test_data_path, tmp_path, name):
    model, media, _ = _setup(test_data_path, tmp_path, name)
    generator = _generator(model)
    generator.set_inputs(media())

    with pytest.raises(RuntimeError, match=MESSAGE):
        generator.set_inputs(media())


@pytest.mark.parametrize("name", list(MODELS))
def test_media_after_append_tokens_raises(test_data_path, tmp_path, name):
    # append_tokens also uses up the generator's one hand-off of extra inputs.
    model, media, text = _setup(test_data_path, tmp_path, name)
    generator = _generator(model)
    generator.append_tokens(text["input_ids"].as_numpy())
    length = len(generator.get_sequence(0))

    with pytest.raises(RuntimeError, match=MESSAGE):
        generator.set_inputs(media())
    assert len(generator.get_sequence(0)) == length


def test_media_after_rewind_to_zero_raises(test_data_path, tmp_path):
    # Rewinding does not clear the media the state already holds, so the generator cannot take new media.
    model, media, _ = _setup(test_data_path, tmp_path, "qwen3-5")
    generator = _generator(model)
    generator.set_inputs(media())
    generator.generate_next_token()
    generator.rewind_to(0)

    with pytest.raises(RuntimeError, match=MESSAGE):
        generator.set_inputs(media())


@pytest.mark.parametrize("count", [0, 1, -1])
def test_later_int32_image_count(test_data_path, tmp_path, count):
    # Hand-built inputs can hold the count in another integer type than the processor's int64.
    model, media, text = _setup(test_data_path, tmp_path, "lfm2-vl")
    generator = _generator(model)
    generator.set_inputs(media())
    text["num_image_tokens"] = np.array([0, count], dtype=np.int32)

    if count:
        with pytest.raises(RuntimeError, match=MESSAGE):
            generator.set_inputs(text)
    else:
        generator.set_inputs(text)


@pytest.mark.parametrize("name", list(MODELS))
def test_later_text_set_inputs_matches_append_tokens(test_data_path, tmp_path, name):
    # The processor's text-only output still carries the media count tensors (num_image_tokens [0],
    # audio_sizes [0]); a later turn with them must run exactly as appending its tokens does.
    model, media, text = _setup(test_data_path, tmp_path, name)
    logits, sequences = [], []
    for later in ("set_inputs", "append_tokens"):
        generator = _generator(model)
        generator.set_inputs(media())
        for _ in range(2):
            generator.generate_next_token()
        if later == "set_inputs":
            generator.set_inputs(text)
        else:
            generator.append_tokens(text["input_ids"].as_numpy())
        logits.append(generator.get_logits().copy())
        for _ in range(3):
            generator.generate_next_token()
        sequences.append(generator.get_sequence(0).copy())
    np.testing.assert_array_equal(logits[0], logits[1])
    np.testing.assert_array_equal(sequences[0], sequences[1])
