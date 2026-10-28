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
- ONNX Runtime GenAI C# package ([installation instructions](https://onnxruntime.ai/docs/genai/howto/install))
- A supported streaming ASR ONNX model, for example:
  - `nvidia/nemotron-speech-streaming-en-0.6b` (Nemotron)
  - a Moonshine streaming export (`streaming_enc_dec_asr`)

## Build

```bash
cd examples/csharp/
dotnet build ModelASR -c Release
```

This example uses officially released GenAI packages only. The current `0.17.0` package references
do not contain the required `TokenPostProcessor` APIs, even for plain-text mode. Completing this
example is pending the planned postprocessor refactor and a new GenAI package release. Both the CPU
and CUDA package references must be updated to that release before the example can build and run.
No local project reference or local-build switch is provided.

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

For Nemotron, configure `model.timestamp_level` in `genai_config.json` as `word`, `segment`, or `all`
to print completed timestamp records instead of incremental token text. `TokenPostProcessor`
owns decoding and grouping, consumes every sequential emission through `ProcessNextTokens()`,
and exposes read-only per-call results through `GetNextTokenResults()`.
Results expose `GetText()` and nullable `GetTimestamps()`; timestamps expose
`GetWords()` and `GetSegments()`, with text and time accessors on each record.
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

The sample builds the timestamped transcript from completed segments, formatted as
`[StartTime - StopTime] SegmentText`, rather than concatenating token fragments.
It flushes remaining audio and calls `FinalizeProcessing()` to complete the trailing
segment. Timestamp state is initialized by `TokenPostProcessor(generator)`; no
explicit metadata-state initialization is needed. The generator must outlive the
processor, and the sample copies each formatted record into its own transcript
before the next processing or finalization call.

```text
------------------------------------------------------------
 This is an example of streaming speech recognition...
============================================================
  This is an example of streaming speech recognition using Nemotron.
============================================================
  Audio: 10.50s | Wall: 2.13s | RTFx: 4.93x
```
