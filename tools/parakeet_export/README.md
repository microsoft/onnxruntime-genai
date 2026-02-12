# Parakeet-TDT ONNX Export Tool

Export NVIDIA Parakeet-TDT-0.6B-v3 ASR model to ONNX format
for inference with ONNX Runtime GenAI.

## Architecture

Parakeet-TDT uses a Token-and-Duration Transducer (TDT) architecture:
- **Encoder**: FastConformer (24 layers, 1024 hidden, ~600M params)
- **Decoder**: RNNT prediction network (2 LSTM layers, 640 hidden)
- **Joint**: TDT joint network — outputs both token logits and duration logits

The key difference from standard RNNT (e.g., Nemotron) is that the joint
network output has shape `[B, T, U, V + 1 + num_durations]` where the last
`num_durations` values are duration logits. During greedy decode, the predicted
duration tells the decoder how many encoder frames to skip, making inference
faster.

## Prerequisites

```bash
conda create -n parakeet-export python=3.10 -y
conda activate parakeet-export
pip install Cython packaging torch torchaudio onnxruntime
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

## Export

### 1. Export ONNX models (encoder, decoder, joint)

```bash
python export_parakeet_to_onnx.py --output_dir ./onnx_models
```

The decoder is exported with explicit LSTM state I/O (`h_in`/`c_in` → `h_out`/`c_out`)
for stateful RNNT decoding.

Options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `nvidia/parakeet-tdt-0.6b-v3` | HuggingFace model name or `.nemo` path |
| `--output_dir` | `./onnx_models` | Output directory |
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
   into `SkipLayerNormalization` ops.
2. **INT4 quantization** — Converts FP32 MatMul weights to 4-bit
   (`MatMulNBits`, symmetric, block_size=32).

Options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | `./onnx_models` | Source directory with original ONNX models |
| `--output_dir` | `./onnx_models_optimized` | Output directory for optimized models |
| `--block_size` | `32` | INT4 quantization block size |
| `--accuracy_level` | `4` | Accuracy level (0=unset, 4=highest) |
| `--skip_fusion` | | Skip graph fusion stage |
| `--skip_quantization` | | Skip INT4 quantization stage |
| `--quant_method` | `rtn` | `rtn`, `k_quant_mixed`, or `hqq` |

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
| `joint.onnx` (+`.data`) | TDT joint network (encoder + decoder → token logits + duration logits) |
| `genai_config.json` | Model configuration for onnxruntime-genai (includes TDT durations) |
| `audio_processor_config.json` | Mel spectrogram parameters (16kHz, 128 mels, 512 FFT) |
| `tokenizer.json` | HuggingFace Unigram tokenizer |
| `tokenizer_config.json` | T5Tokenizer class routing for ORT Extensions |
| `vocab.txt` | Raw vocabulary (one token per line) |

## Scripts

| Script | Purpose |
|--------|---------|
| `export_parakeet_to_onnx.py` | Export encoder, decoder, TDT joint ONNX models from NeMo |
| `export_tokenizer.py` | Extract vocab from NeMo and create ORT-compatible tokenizer |
| `optimize_encoder.py` | Optimize encoder: graph fusion (conformer) + INT4 quantization |
| `test_e2e.py` | End-to-end test: model load, tokenizer, inference, raw ONNX TDT baseline |
| `test_real_speech.py` | Real speech test with NeMo preprocessing, compares OG vs raw ORT with TDT decode |

## TDT vs RNNT

| Feature | RNNT (Nemotron) | TDT (Parakeet) |
|---------|----------------|----------------|
| Joint output | `[B, T, U, V+1]` | `[B, T, U, V+1+num_durations]` |
| Blank handling | Advance 1 frame | Advance by predicted duration |
| Durations | N/A | e.g., `[0, 1, 2, 4, 8]` |
| Speed | Baseline | ~64% faster |
