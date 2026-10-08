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
Until a feature-containing package is published, build from the repository root with:

```bash
dotnet build examples/csharp/ModelASR -c Release -p:UseLocalGenAI=true
```

Use the matching native GenAI, ONNX Runtime, and execution-provider libraries at runtime. Make their
directories available on `LD_LIBRARY_PATH` on Linux or `PATH` on Windows. The native build also needs
the corresponding ONNX Runtime Extensions metadata support. Update both the CPU and CUDA package
references to a feature-containing release before using the package-based build.

## Run

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

For Nemotron, configure `model.timestamp_level` in `genai_config.json` as `word`, `segment`, or `all`
to print completed timestamp records instead of incremental token text. The tokenizer stream
initializes metadata automatically, and finalization emits the trailing records. `off` (the default)
keeps ordinary decoding; Moonshine supports only this plain-text path. See
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
