# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


@pytest.fixture(scope="module")
def streaming_models():
    asr = os.environ.get("NEMOTRON_STREAMING_MODEL_PATH")
    diarization = os.environ.get("NEMOTRON_DIARIZATION_MODEL_PATH")
    if not asr or not diarization:
        pytest.skip("Set both Nemotron model paths for real-audio integration tests")
    return Path(asr), Path(diarization)


@pytest.fixture(scope="module")
def speech_audio():
    soundfile = pytest.importorskip("soundfile")
    signal = pytest.importorskip("scipy.signal")
    audio, sample_rate = soundfile.read(
        Path(__file__).parents[1] / "audios" / "jfk.flac", dtype="float32"
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    divisor = np.gcd(sample_rate, 16000)
    audio = signal.resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    return np.asarray(audio[:63937], dtype=np.float32)


@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.parametrize("level", ["off", "word", "segment"])
def test_real_audio_metadata_modes(tmp_path, streaming_models, speech_audio, diarization, level):
    asr_path, diarization_path = streaming_models
    config = json.loads((asr_path / "genai_config.json").read_text())
    for asset in asr_path.iterdir():
        if asset.name != "genai_config.json":
            (tmp_path / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
    model_config = config["model"]
    model_config["timestamp_level"] = level
    for component in ["encoder", "decoder", "joiner"]:
        model_config[component]["session_options"] = {"provider_options": []}
    model_config.pop("vad", None)
    model_config.pop("diarization", None)
    if diarization:
        (tmp_path / "diarization").symlink_to(diarization_path, target_is_directory=True)
        model_config["diarization"] = {
            "filename": "diarization/model.onnx",
            "right_context": 4,
            "session_options": {"provider_options": []},
        }
    (tmp_path / "genai_config.json").write_text(json.dumps(config))
    model = og.Model(str(tmp_path))
    processor = og.StreamingProcessor(model)
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()
    parameters = og.GeneratorParams(model)
    parameters.set_search_options(max_length=512)
    generator = og.Generator(model, parameters)
    fragments = []
    records = []
    metadata = diarization or level != "off"

    def collect(result):
        fragments.append(result.text)
        assert result.timestamp_metadata is not None
        if level == "word":
            assert result.timestamp_metadata.segments == []
            records.extend(result.timestamp_metadata.words)
        else:
            assert result.timestamp_metadata.words == []
            records.extend(result.timestamp_metadata.segments)

    def decode(inputs):
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
            if metadata:
                for token in generator.get_next_tokens_with_metadata():
                    collect(stream.decode_with_metadata(token))
            else:
                for token in generator.get_next_tokens():
                    fragments.append(stream.decode(token))

    chunk = model_config["chunk_samples"]
    for start in range(0, len(speech_audio), chunk):
        inputs = processor.process(speech_audio[start:start + chunk])
        if inputs is not None:
            decode(inputs)
    while (inputs := processor.flush()) is not None:
        decode(inputs)
    if metadata:
        collect(stream.finalize_metadata())
        trailing = stream.finalize_metadata().timestamp_metadata
        assert trailing.words == [] and trailing.segments == []
        assert records
        assert "".join(record.text for record in records) == "".join(fragments)
        for record in records:
            if diarization:
                assert record.speaker_label in {f"speaker_{index}" for index in range(8)}
            else:
                assert record.speaker_label is None
            if level == "off":
                assert record.start_frame is None and record.stop_frame is None
                assert record.start_time is None and record.stop_time is None
            else:
                assert 0 <= record.start_frame < record.stop_frame
                assert record.start_time < record.stop_time
                assert record.stop_time <= len(speech_audio) / 16000 + 0.08
    assert "".join(fragments).strip()
    print(f"diarization={diarization}, timestamps={level}: {''.join(fragments)}")
