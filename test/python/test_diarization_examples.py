# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_nemotron_diarization import speech_audio, streaming_models


@pytest.mark.parametrize("language", ["cpp", "csharp", "python"])
@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.parametrize("level", ["off", "word", "segment"])
def test_streaming_examples_metadata(
    tmp_path, streaming_models, speech_audio, language, diarization, level
):
    if language != "python":
        key = (
            "NEMOTRON_CPP_EXAMPLE_PATH"
            if language == "cpp"
            else "NEMOTRON_CSHARP_EXAMPLE_PATH"
        )
        binary = os.environ.get(key)
        if not binary:
            pytest.skip(f"Set {key} to run the {language} example")

    soundfile = pytest.importorskip("soundfile")
    asr_path, diarization_path = streaming_models
    package = tmp_path / "model"
    package.mkdir()
    config = json.loads((asr_path / "genai_config.json").read_text())
    for asset in asr_path.iterdir():
        if asset.name != "genai_config.json":
            (package / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
    model_config = config["model"]
    model_config["timestamp_level"] = level
    for component in ["encoder", "decoder", "joiner"]:
        model_config[component]["session_options"] = {"provider_options": []}
    model_config.pop("vad", None)
    model_config.pop("diarization", None)
    if diarization:
        (package / "diarization").symlink_to(diarization_path, target_is_directory=True)
        model_config["diarization"] = {
            "filename": "diarization/model.onnx",
            "right_context": 4,
            "session_options": {"provider_options": []},
        }
    (package / "genai_config.json").write_text(json.dumps(config))
    audio = tmp_path / "audio.wav"
    soundfile.write(audio, speech_audio, 16000, subtype="PCM_16")
    if language == "cpp":
        command = [binary, "--model_path", str(package), "--audio_file", str(audio)]
    elif language == "csharp":
        command = ["dotnet", binary, str(package), str(audio), "follow_config"]
    else:
        script = Path(__file__).parents[2] / "examples" / "python" / "model-asr.py"
        command = [
            sys.executable, str(script), "--model_path", str(package),
            "--audio_file", str(audio), "--execution_provider", "cpu",
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    summary = re.search(r"\n={60}\n  ([^\n]+)\n={60}", result.stdout)
    assert summary is not None, result.stdout
    assert summary.group(1).strip() == "And so, my fellow Americans"
    labels = re.findall(r"\[speaker_[0-7]\]", result.stdout)
    times = re.findall(r"\[\d+(?:\.\d+)? - \d+(?:\.\d+)?\]", result.stdout)
    assert bool(labels) == diarization, result.stdout
    assert bool(times) == (level != "off"), result.stdout
