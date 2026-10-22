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

The timestamp-capable example requires bindings and a native GenAI library containing the timestamp
APIs; the current `0.17.0` package references do not contain them, even when running in plain-text mode.
The timestamp implementation is available on `main` after
[PR #2591](https://github.com/microsoft/onnxruntime-genai/pull/2591); diarization also
requires the implementation from [PR #2719](https://github.com/microsoft/onnxruntime-genai/pull/2719).
Until a feature-containing package is published, build from the root of that checkout with:

```bash
dotnet build examples/csharp/ModelASR -c Release -p:UseLocalGenAI=true
```

Use the matching native GenAI, ONNX Runtime, and execution-provider libraries at runtime. Make their
directories available on `LD_LIBRARY_PATH` on Linux or `PATH` on Windows. The native build also needs
the corresponding ONNX Runtime Extensions lexical lookup API. Update both the CPU and CUDA package
references to a feature-containing release before using the package-based build.

## Run

For diarization, use bindings and native binaries built from the corresponding
implementation branch until a package containing the metadata APIs is released.
The pinned `0.17.0` packages are not sufficient. Build the local bindings with:

```bash
dotnet build ModelASR -c Release -p:UseLocalGenAI=true
```

Make the matching GenAI native libraries available to the application through
`LD_LIBRARY_PATH` on Linux or `PATH` on Windows.

```bash
cd ./ModelASR/bin/Release/net8.0/
./ModelASR <model_path> <audio_file.wav> [execution_provider]
```

### Arguments

| Argument | Description |
|---|---|
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
to print completed timestamp records instead of incremental token text. `TokenPostProcessor`
owns decoding and grouping, consumes every sequential emission through `ProcessNextTokens()`,
and exposes read-only per-call results through `GetNextTokenResults()`.
The same processor handles diarization; applications do not run a separate speaker
postprocessor or tokenizer metadata stream.
Results expose `GetText()` and nullable `GetTimestamps()`; timestamps expose
`GetWords()` and `GetSegments()`, with text and time accessors on each record.
These objects borrow the processor's current output. The sample appends needed
values to its own transcript before the next processing or finalization call.
`FinalizeProcessing()` completes trailing records. With diarization disabled, `off`
(the default) returns ordinary text without timestamps; Moonshine supports only this plain-text path. See
[Nemotron streaming timestamps](../../../docs/nemotron_speech_timestamps.md) for configuration and
output formats.

```
------------------------------------------------------------
 This is an example of streaming speech recognition...
============================================================
  This is an example of streaming speech recognition using Nemotron.
============================================================
  Audio: 10.50s | Wall: 2.13s | RTFx: 4.93x
```

## Speaker-attributed output

Configure the model's `diarization.filename` and optional `right_context` in
`genai_config.json`, as described in the [streaming diarization guide](../../../docs/nemotron_speech_timestamps.md#streaming-diarization).
The postprocessor automatically includes the configured metadata:

| `timestamp_level` | With diarization enabled |
|---|---|
| `off` | Speaker-turn segments without timestamps |
| `word` | Timestamped words with speaker labels |
| `segment` | Timestamped speaker-turn segments |

Without diarization, timestamp-only modes still work and `off` retains ordinary
incremental text. `all` is not supported. Metadata output prints completed records
once; the final summary is the plain transcript. The example flushes every pending
audio chunk before finalizing metadata. VAD statistics count actually skipped
central chunks, not calls waiting for diarization look-ahead.

`GetTimestamps()` is non-null for diarization-only output even though record timing
getters return null. `GetSpeakerLabel()` reads the optional label from the same
borrowed record. All output is consumed before the next postprocessor mutation.
