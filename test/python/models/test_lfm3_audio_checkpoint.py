# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
The runtime's mel values and token ids against the real LiquidAI/LFM3-Audio-2B classes. Opt-in, since they run the
checkpoint's remote code: set ORTGENAI_LFM3_AUDIO_CHECKPOINT to a local snapshot of the checkpoint.

    ORTGENAI_LFM3_AUDIO_CHECKPOINT=<snapshot> python -m pytest test/python/models/test_lfm3_audio_checkpoint.py \
        --test_models test/models
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from test_lfm2_audio_models import _synthetic_signal, _write_wav
from test_lfm3_audio_models import AUDIO_MARKER, SAMPLE_RATE, _reference_mel

CHECKPOINT = os.environ.get("ORTGENAI_LFM3_AUDIO_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="ORTGENAI_LFM3_AUDIO_CHECKPOINT is not set")

NUM_MELS = 128
AUDIO = {"type": "audio"}
SYSTEM = {"role": "system", "content": "Perform ASR in English."}
CONVERSATIONS = {
    "asr": [SYSTEM, {"role": "user", "content": [AUDIO]}],
    "chat": [{"role": "user", "content": [AUDIO]}],
    "two-clips": [{"role": "user", "content": [AUDIO, {"type": "text", "text": "and"}, AUDIO]}],
}


@pytest.fixture
def model(test_data_path, tmp_path, monkeypatch) -> og.Model:
    """The fixture pipeline with a 128-mel speech graph and the checkpoint's tokenizer: enough to run the processor."""
    pytest.importorskip("onnx")
    monkeypatch.syspath_prepend(os.fspath(Path(__file__).resolve().parents[1] / "create"))
    from create_dummy_lfm3_audio_models import create_dummy_speech_model  # noqa: PLC0415

    model_dir = tmp_path / "model"
    shutil.copytree(Path(test_data_path) / "lfm3-audio", model_dir)
    create_dummy_speech_model(os.fspath(model_dir / "dummy_speech.onnx"), np.random.default_rng(0), NUM_MELS)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(Path(CHECKPOINT) / name, model_dir / name)
    config_path = model_dir / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"].update(num_mels=NUM_MELS, audio_token_id=125031, vocab_size=125034)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return og.Model(os.fspath(model_dir))


@pytest.fixture(scope="module")
def reference():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)


@pytest.mark.parametrize("num_samples", [1, 400, 401, 561, 8000, 16000, 48000])
def test_lfm3_audio_mel_equals_the_checkpoint_extractor(model, reference, tmp_path, num_samples):
    signal = _synthetic_signal(num_samples / SAMPLE_RATE)
    clip = _write_wav(tmp_path / "clip.wav", signal)

    inputs = model.create_multimodal_processor()(AUDIO_MARKER, audios=og.Audios.open(clip))

    expected = reference.feature_extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"]
    assert inputs["audio_embeds"].as_numpy().shape == expected.shape
    # Bounds from real clips: float32 rounding moves near-silent bands only
    diff = np.abs(inputs["audio_embeds"].as_numpy() - expected)[0]
    assert diff.max() <= 0.02
    assert diff[_reference_mel(signal, NUM_MELS)[1] >= 1e-6].max(initial=0) <= 1e-4


@pytest.mark.parametrize("name", CONVERSATIONS)
def test_lfm3_audio_ids_equal_the_checkpoint_processor(model, reference, tmp_path, name):
    conversation = CONVERSATIONS[name]
    num_clips = sum(item == AUDIO for message in conversation for item in message["content"])
    signals = [_synthetic_signal(1.0 / (i + 1), seed=i) for i in range(num_clips)]
    clips = [_write_wav(tmp_path / f"clip{i}.wav", signal) for i, signal in enumerate(signals)]
    prompt = og.Tokenizer(model).apply_chat_template(json.dumps(conversation), add_generation_prompt=True)

    inputs = model.create_multimodal_processor()(prompt, audios=og.Audios.open(*clips))

    expected = reference(text=prompt, audio=signals, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_ids"]
    np.testing.assert_array_equal(inputs["input_ids"].as_numpy(), expected)
