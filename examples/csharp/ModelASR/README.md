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

```
------------------------------------------------------------
 This is an example of streaming speech recognition...
============================================================
  This is an example of streaming speech recognition using Nemotron.
============================================================
  Audio: 10.50s | Wall: 2.13s | RTFx: 4.93x
```
