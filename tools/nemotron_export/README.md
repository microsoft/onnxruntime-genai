# Nemotron ONNX Export Tool

Export NVIDIA Nemotron-Speech-Streaming-En-0.6b ASR model to ONNX format
for inference with ONNX Runtime GenAI.

## Prerequisites

```bash
conda create -n nemotron-export python=3.10 -y
conda activate nemotron-export
pip install Cython packaging torch torchaudio onnxruntime
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

## Export

### 1. Export ONNX models (encoder, decoder, joint)

```bash
python export_nemotron_to_onnx.py --output_dir ./onnx_models
```

The decoder is exported with explicit LSTM state I/O (`h_in`/`c_in` → `h_out`/`c_out`)
for stateful RNNT decoding.

Options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `nvidia/nemotron-speech-streaming-en-0.6b` | HuggingFace model name or `.nemo` path |
| `--output_dir` | `./onnx_models` | Output directory |
| `--chunk_size` | `1.12` | Streaming chunk size (0.08, 0.16, 0.56, 1.12 seconds) |
| `--opset_version` | `17` | ONNX opset version |
| `--device` | `cpu` | Device for export (`cpu` or `cuda`) |

### 2. Export tokenizer

```bash
python export_tokenizer.py --output_dir ./onnx_models
```

Extracts the SentencePiece vocabulary from NeMo and converts it to HuggingFace
Unigram format (`tokenizer.json` + `tokenizer_config.json`) compatible with
ORT Extensions' T5Tokenizer path.

## Optimize

Optimize the encoder model with graph fusion + INT4 quantization.
Decoder and Joint remain FP32 (they are small and run many times per token).

```bash
python optimize_encoder.py --model_dir ./onnx_models --output_dir ./onnx_models_optimized
```

This applies two optimization stages to the encoder:

1. **Graph fusion** (`model_type=conformer`) — Fuses LayerNorm + residual Add
   into `SkipLayerNormalization` ops (96 fusions across 24 layers).
2. **INT4 quantization** — Converts FP32 MatMul weights to 4-bit
   (`MatMulNBits`, symmetric, block_size=32). Reduces encoder from ~2.4 GB → ~648 MB.

Options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | `./onnx_models` | Source directory with original ONNX models |
| `--output_dir` | `./onnx_models_optimized` | Output directory for optimized models |
| `--block_size` | `32` | INT4 quantization block size |
| `--accuracy_level` | `4` | Accuracy level (0=unset, 4=highest) |
| `--skip_fusion` | | Skip graph fusion stage |
| `--skip_quantization` | | Skip INT4 quantization stage |

## Test

```bash
# End-to-end test via onnxruntime-genai (requires built wheel)
python test_e2e.py

# Real speech test with jfk.flac
python test_real_speech.py
```

## Exported Files

| File | Description |
|------|-------------|
| `encoder.onnx` (+`.data`) | FastConformer encoder (24 layers, ~2.4 GB weights) |
| `decoder.onnx` (+`.data`) | RNNT prediction network (2 LSTM layers, stateful h/c I/O) |
| `joint.onnx` (+`.data`) | Joint network (encoder + decoder → logits) |
| `genai_config.json` | Model configuration for onnxruntime-genai |
| `audio_processor_config.json` | Mel spectrogram parameters (16kHz, 128 mels, 512 FFT) |
| `tokenizer.json` | HuggingFace Unigram tokenizer (1025 tokens) |
| `tokenizer_config.json` | T5Tokenizer class routing for ORT Extensions |
| `vocab.txt` | Raw vocabulary (one token per line) |

## Scripts

| Script | Purpose |
|--------|---------|
| `export_nemotron_to_onnx.py` | Export encoder, decoder, joint ONNX models from NeMo |
| `export_tokenizer.py` | Extract vocab from NeMo and create ORT-compatible tokenizer |
| `optimize_encoder.py` | Optimize encoder: graph fusion (conformer) + INT4 quantization |
| `test_e2e.py` | End-to-end test: model load, tokenizer, inference, raw ONNX baseline |
| `test_real_speech.py` | Real speech test with NeMo preprocessing, compares OG vs raw ORT |
