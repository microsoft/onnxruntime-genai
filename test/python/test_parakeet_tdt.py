# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Parakeet TDT speech recognition model.

Mirrors the structure of the existing nemotron / whisper tests:
- creation tests (model + processor + generator)
- transcription tests on bundled audio clips (jfk.flac, tedlium_long_120s.flac)
- a WER assertion (< 0.1) against a reference transcription

Reference transcriptions are intentionally left blank — fill in the empty
string in REFERENCE_TRANSCRIPTIONS once the ground truth has been confirmed.
"""

import os
import re
from pathlib import Path

import pytest

import onnxruntime_genai as og


# Reference transcriptions for the bundled audio clips. Leave the values empty
# to skip the WER assertion (the test will still run end-to-end and confirm
# the model produces non-empty output).
REFERENCE_TRANSCRIPTIONS = {
    "jfk.flac": "",
    "tedlium_long_120s.flac": "",
}


def _normalize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split into whitespace tokens for WER."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def _wer(reference: str, hypothesis: str) -> float:
    """Standard Levenshtein-based word error rate."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # Classic edit-distance DP.
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / float(n)


def _transcribe(model_path: str, audio_path: str) -> str:
    """Run the parakeet pipeline end-to-end and return the decoded text."""
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    audios = og.Audios.open(audio_path)
    inputs = processor("", audios=audios)

    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    while not generator.is_done():
        generator.generate_next_token()

    tokens = list(generator.get_sequence(0))
    if tokens:
        # The processor injects a single SOS placeholder at index 0; skip it.
        tokens = tokens[1:]
    return processor.decode(tokens).strip()


def test_parakeet_create(parakeet_tdt_model_path):
    """Model + processor + generator construction should succeed."""
    model = og.Model(parakeet_tdt_model_path)
    processor = model.create_multimodal_processor()
    assert processor is not None

    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)
    assert generator is not None


@pytest.mark.parametrize("relative_audio_path", [Path("audios") / "jfk.flac"])
def test_parakeet_transcribe_jfk(parakeet_tdt_model_path, test_data_path, relative_audio_path):
    """Transcribe the short JFK clip and compare to reference (if provided)."""
    audio_path = os.fspath(Path(test_data_path) / relative_audio_path)
    if not os.path.exists(audio_path):
        pytest.skip(f"Audio file not found: {audio_path}")

    transcription = _transcribe(parakeet_tdt_model_path, audio_path)
    assert transcription, "Empty transcription returned"

    reference = REFERENCE_TRANSCRIPTIONS.get(relative_audio_path.name, "")
    if reference:
        wer = _wer(reference, transcription)
        assert wer < 0.1, f"WER too high: {wer:.3f}\nref: {reference}\nhyp: {transcription}"


@pytest.mark.parametrize("relative_audio_path", [Path("audios") / "tedlium_long_120s.flac"])
def test_parakeet_transcribe_long(parakeet_tdt_model_path, test_data_path, relative_audio_path):
    """Transcribe the 120s TED clip and compare to reference (if provided)."""
    audio_path = os.fspath(Path(test_data_path) / relative_audio_path)
    if not os.path.exists(audio_path):
        pytest.skip(f"Audio file not found: {audio_path}")

    transcription = _transcribe(parakeet_tdt_model_path, audio_path)
    assert transcription, "Empty transcription returned"

    reference = REFERENCE_TRANSCRIPTIONS.get(relative_audio_path.name, "")
    if reference:
        wer = _wer(reference, transcription)
        assert wer < 0.1, f"WER too high: {wer:.3f}\nref: {reference}\nhyp: {transcription}"
