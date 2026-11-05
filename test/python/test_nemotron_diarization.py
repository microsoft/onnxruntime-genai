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
    parameters = og.GeneratorParams(model)
    parameters.set_search_options(max_length=512)
    generator = og.Generator(model, parameters)
    postprocessor = og.ASRPostProcessor(generator)
    fragments = []
    records = []
    metadata = diarization or level != "off"
    borrowed = []

    def collect(result):
        fragments.append(result.text)
        timestamps = result.timestamp_result
        if not metadata:
            assert timestamps is None
            return
        assert timestamps is not None
        if level == "word":
            assert timestamps.segments == []
            completed = timestamps.words
        else:
            assert timestamps.words == []
            completed = timestamps.segments
        for record in completed:
            records.append((record.text, record.frame_interval, record.time_interval, record.speaker_label))
            borrowed.append(record)

    def process_current():
        previous = borrowed[:]
        borrowed.clear()
        postprocessor.process_next()
        for record in previous:
            with pytest.raises(RuntimeError, match="expired"):
                _ = record.speaker_label
            with pytest.raises(RuntimeError, match="expired"):
                _ = record.has_timestamp
        for result in postprocessor.get_next_results():
            collect(result)

    def decode(inputs):
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
            process_current()

    chunk = model_config["chunk_samples"]
    for start in range(0, len(speech_audio), chunk):
        inputs = processor.process(speech_audio[start:start + chunk])
        if inputs is not None:
            decode(inputs)
    while (inputs := processor.flush()) is not None:
        decode(inputs)
    postprocessor.finalize()
    for result in postprocessor.get_next_results():
        collect(result)
    if metadata:
        postprocessor.finalize()
        trailing = postprocessor.get_next_results()[0].timestamp_result
        assert trailing.words == [] and trailing.segments == []
        assert records
        assert "".join(record[0] for record in records) == "".join(fragments)
        for _, frame_interval, time_interval, speaker_label in records:
            if diarization:
                assert speaker_label in {f"speaker_{index}" for index in range(8)}
            else:
                assert speaker_label is None
            if level == "off":
                assert frame_interval is None and time_interval is None
            else:
                start_frame, stop_frame = frame_interval
                start_time, stop_time = time_interval
                assert 0 <= start_frame < stop_frame
                assert start_time < stop_time
                assert stop_time <= len(speech_audio) / 16000 + 0.08
    assert "".join(fragments).strip()
    print(f"diarization={diarization}, timestamps={level}: {''.join(fragments)}")
