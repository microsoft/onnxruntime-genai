# Model ASR (Streaming) — C# Example

This example demonstrates real-time streaming speech recognition using the
ONNX Runtime GenAI C# `StreamingProcessor` API. It drives any streaming
encoder/decoder ASR model exposed through onnxruntime-genai:

- NVIDIA Nemotron streaming RNN-T (`nemotron_speech`, multilingual)
- Moonshine streaming encoder-decoder (`streaming_enc_dec_asr`, English only)

Audio is streamed through the model in chunks (simulating a microphone feed),
and transcribed text is printed incrementally as it becomes available.

## Prerequisites

- .NET 8.0 SDK or later
- ONNX Runtime GenAI C# package v0.19.0 ([installation instructions](https://onnxruntime.ai/docs/genai/howto/install))
- A supported streaming ASR ONNX model, for example:
  - `nvidia/nemotron-speech-streaming-en-0.6b` (Nemotron)
  - a Moonshine streaming export (`streaming_enc_dec_asr`)

## Build

```bash
cd examples/csharp/
dotnet build ModelASR -c Release
```

The GenAI `0.19.0` baseline contains the timestamp APIs, but these speaker-aware
examples additionally require the diarization branch's optional intervals and speaker
properties. Build the matching bindings from the repository root:

```bash
dotnet build examples/csharp/ModelASR -c Release -p:UseLocalGenAI=true
```

Use the matching native libraries on `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows).
The local project reference does not supply the native binaries.

## Run

```bash
cd ./ModelASR/bin/Release/net8.0/
./ModelASR <model_path> <audio_file.wav> [execution_provider]
```

### Arguments

| Argument | Description |
| --- | --- |
| `model_path` | Path to the streaming ASR ONNX model directory |
| `audio_file.wav` | Path to a WAV audio file (any sample rate — resampled automatically) |
| `execution_provider` | *(Optional)* Execution provider: `cpu`, `cuda`, `dml`, or `follow_config` (default: `follow_config`) |
| `--use_vad true` | *(Optional, Nemotron-only)* Enable Silero VAD if the model's `genai_config.json` has a `vad` section |

### Example

```bash
# CPU inference
./ModelASR /path/to/nemotron-cpu-int4 /path/to/audio.wav

# CUDA inference
./ModelASR /path/to/nemotron-cuda-int4 /path/to/audio.wav cuda
```

### Output

The example prints transcribed text incrementally as each audio chunk is processed,
followed by a summary with the full transcript, audio duration, wall-clock time,
and real-time factor (RTFx).

For Nemotron, configure `model.timestamp_level` in `genai_config.json` as `word` or `segment`
to print completed timestamp records instead of incremental token text. `ASRPostProcessor`
owns decoding and grouping, consumes every sequential emission through `ProcessNext()`,
and exposes read-only per-call results through `GetNextResults()`.
Results expose `Text` and nullable `TimestampResult` properties; timestamps expose
`Words` and `Segments`. Records expose `Text`, nullable `FrameInterval` / `TimeInterval`
pairs, and nullable `SpeakerLabel`. With diarization enabled, `word` prints labeled
words, `segment` prints timed speaker turns, and `off` prints untimed speaker turns.
Without diarization, `off` prints ordinary text. `all` is not supported.
These objects borrow the processor's current output. The sample appends needed
values to its own transcript before the next processing or finalization call.
`FinalizeProcessing()` completes trailing records. `off` (the default) returns ordinary text
without timestamps; Moonshine supports only this plain-text path. See
[Nemotron streaming timestamps](../../../docs/nemotron_speech_timestamps.md) for configuration and
output formats.

### Segment timestamps

Set the existing `model.timestamp_level` field to `segment` in the model's
`genai_config.json` before loading the model, then use the same ModelASR executable:

```bash
./ModelASR /path/to/nemotron-model /path/to/audio.wav cuda
```

The sample prints completed segments as `[StartTime - StopTime] SegmentText`,
adding `[speaker_N]` with diarization. The final summary remains plain text.
It drains all remaining audio and calls `FinalizeProcessing()` to complete the trailing
segment. Timestamp state is initialized by `ASRPostProcessor(generator)`; no
explicit metadata-state initialization is needed. The generator must outlive the
processor, and the sample consumes each borrowed record before the next processing
or finalization call.

```text
------------------------------------------------------------
 This is an example of streaming speech recognition...
============================================================
  This is an example of streaming speech recognition using Nemotron.
============================================================
  Audio: 10.50s | Wall: 2.13s | RTFx: 4.93x
```
