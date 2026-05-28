# End-to-End Demo: LLM Inference with Quantized KV Cache on CPU

This demo shows how to export and run a popular LLM with INT8 quantized KV cache
using onnxruntime-genai on CPU. By default it uses Qwen2.5-0.5B (non-gated, small).

Quantizing the KV cache from FP32 to INT8 reduces memory bandwidth by ~4x during
token generation, improving throughput for long-sequence inference.

---

## Step 1: Prerequisites

- Linux (Ubuntu 20.04+) or Windows with WSL2
- Python 3.10+
- ~4 GB RAM (for Qwen2.5-0.5B; ~8 GB for larger models)

---

## Step 2: Set Up the Environment

```bash
# Clone the repositories (if not already cloned)
git clone https://github.com/microsoft/onnxruntime.git
git clone https://github.com/microsoft/onnxruntime-genai.git

# Create a virtual environment
cd onnxruntime-genai
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy sentencepiece protobuf onnx-ir
```

---

## Step 3: Build ONNX Runtime from Source

The quantized KV cache feature requires ONNX Runtime built from the `main` branch.

```bash
cd ../onnxruntime

# Build (CPU only, Release mode)
./build.sh --config Release --build_shared_lib --parallel --skip_tests --build_wheel

# Install the wheel
pip install build/Linux/Release/dist/onnxruntime-*.whl --force-reinstall
```

> **Tip:** You can install onnxruntime nightly package instead in this step to save some time.

---

## Step 4: Install ONNX Runtime GenAI Model Builder

The demo uses the model builder (pure Python) from onnxruntime-genai for exporting.
No C++ compilation needed — just add it to the Python path:

```bash
cd ../onnxruntime-genai

# Option A: Install in editable mode
pip install -e .

# Option B: If the above fails (C++ build issues), just use the path directly
# The demo script auto-discovers builder.py via relative path
```

---

## Step 5: (Optional) Log in to HuggingFace

Only needed for gated models (e.g., Llama). The default model (Qwen2.5-0.5B) is
public and doesn't require login.

```bash
pip install huggingface_hub
hf auth login
# Paste your access token when prompted
```

---

## Step 6: Run the Demo

### Quick start (default scales)

```bash
cd examples/python

python quantized_kv_cache_cpu_demo.py
```

Or specify a different model:

```bash
python quantized_kv_cache_cpu_demo.py \
    --model meta-llama/Llama-3.2-1B-Instruct
```

This will:
1. Export the model to ONNX with INT8 quantized KV cache
2. Run text generation on 3 test prompts
3. Print throughput (tokens/sec)

### With calibration (better accuracy)

```bash
python quantized_kv_cache_cpu_demo.py --calibrate
```

Calibration runs a few forward passes on sample text to find optimal
quantization scales for each layer's K and V projections.

### Compare with FP32 baseline

```bash
python quantized_kv_cache_cpu_demo.py --calibrate --compare
```

This exports both FP32 and INT8 models, runs generation with each, then
prints a side-by-side comparison of output quality and performance.

### Custom prompt

```bash
python quantized_kv_cache_cpu_demo.py \
    --prompt "Summarize the theory of relativity in three sentences:"
```

---

## Step 7: Interpret the Results

Example output:

```
============================================================
RESULTS SUMMARY
============================================================
  Model: Qwen/Qwen2.5-0.5B-Instruct
  KV Cache: int8_per_tensor
  Total tokens: 150
  Total time: 4.23s
  Throughput: 35.5 tokens/sec

  Run with --compare to see FP32 baseline comparison.
============================================================
Demo complete!
```

When running with `--compare`, you'll also see:
- Whether outputs match exactly (they often diverge slightly due to quantization)
- Speedup ratio (INT8 vs FP32 throughput)

---

## All Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local path |
| `--output_dir`, `-o` | `./quantized_kv_demo_output` | Directory for exported models |
| `--quant_type` | `int8_per_tensor` | One of: `int8_per_tensor`, `int8_per_channel`, `int4_per_tensor`, `int4_per_channel` |
| `--calibrate` | off | Run calibration before export |
| `--scale_file` | None | Path to pre-computed scales JSON |
| `--compare` | off | Also run FP32 baseline |
| `--max_length` | 100 | Max tokens to generate |
| `--skip_export` | off | Reuse previously exported model |
| `--prompt` | None | Custom prompt text |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: onnx_ir` | Run `pip install onnx-ir` |
| Model export fails with 403 | Use a non-gated model or run `hf auth login` |
| `Tensor type mismatch` in GQA | Install ORT built from `main` with quantized KV cache support |
| Poor output quality | Use `--calibrate` for data-driven scales |
| Slow build | Add `--parallel N` and `--skip_tests` to `build.sh` |

---

## How It Works (Brief)

1. **Model Builder** (from onnxruntime-genai) exports the ONNX graph with GQA operators
   configured for quantized KV cache (INT8 inputs/outputs, scale initializers,
   quantization attributes).
2. **ONNX Runtime** executes the GQA operator using optimized MLAS kernels
   (AVX2/AVX512-VNNI on x86, NEON on ARM) that quantize K/V on cache write and
   dequantize on attention read.
3. **Demo script** runs a greedy decoding loop via `onnxruntime.InferenceSession`,
   feeding INT8 KV cache tensors back each step. No onnxruntime-genai C++ extension
   required — just `onnxruntime` and `transformers` for tokenization.
