# InternLM2 Model Export Example

This example demonstrates how to export InternLM2 models to ONNX format using ONNX Runtime GenAI.

## Supported Models

All InternLM2 model sizes are supported:

- ✅ **InternLM2-1.8B** - Tested and verified
- ✅ **InternLM2-7B** - Fully compatible
- ✅ **InternLM2-20B** - Fully compatible
- ✅ **InternLM2-Chat variants** - All sizes supported

The implementation is architecture-based and automatically adapts to any InternLM2 model size.

## Model Architecture

InternLM2 uses a Llama-based architecture with the following key features:

- **Attention**: Grouped Query Attention (GQA) with grouped/interleaved QKV layout
- **Normalization**: RMSNorm
- **Activation**: SiLU
- **Positional Encoding**: RoPE with theta=1,000,000

## Export Example

### Basic Export (FP32)

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-onnx-cpu-fp32 \
    --precision fp32 \
    --execution_provider cpu
```

### INT4 Quantization (RTN)

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-onnx-cpu-int4 \
    --precision int4 \
    --execution_provider cpu
```

### INT4 AWQ Quantization (Better Quality) - 1.8B

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-onnx-cpu-int4-awq \
    --precision int4 \
    --execution_provider cpu \
    --extra_options int4_accuracy_level=4
```

### INT4 AWQ Quantization - 7B (Recommended for Production)

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-7b \
    --output ./internlm2-7b-onnx-cpu-int4-awq \
    --precision int4 \
    --execution_provider cpu \
    --extra_options int4_accuracy_level=4
```

**Note:** For 7B and 20B models, INT4 quantization is highly recommended to reduce memory usage while maintaining quality.

### GPU Export (7B on CUDA)

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-7b \
    --output ./internlm2-7b-onnx-cuda-int4 \
    --precision int4 \
    --execution_provider cuda \
    --extra_options int4_accuracy_level=4
```

## Inference Example

```python
import onnxruntime_genai as og

# Load the model
model = og.Model("./internlm2-1.8b-onnx-cpu-int4-awq")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set generation parameters
prompt = "What is the meaning of life?"
tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(
    max_length=200,
    temperature=0.7,
    top_p=0.9,
    top_k=40
)

# Generate text
generator = og.Generator(model, params)
generator.append_tokens(tokens)

print(prompt, end="", flush=True)
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end="", flush=True)
print()
```

## Key Implementation Notes

### Grouped QKV Layout

InternLM2 uses a grouped/interleaved QKV weight layout for efficient Grouped Query Attention:

- Layout: `[Group0: Q0,Q1,K0,V0 | Group1: Q2,Q3,K1,V1 | ...]`
- Each KV group contains multiple Q heads followed by K and V
- The implementation correctly handles this layout during weight extraction

### Model Configuration

The exported model uses:
- **model_type**: "llama" (for ONNX Runtime GenAI compatibility)
- **tokenizer_class**: "LlamaTokenizer" (SentencePiece-based)

## Model Specifications

| Feature | InternLM2-1.8B | InternLM2-7B | InternLM2-20B |
|---------|----------------|--------------|---------------|
| Parameters | 1.8B | 7B | 20B |
| Hidden Size | 2048 | 4096 | 6144 |
| Layers | 24 | 32 | 48 |
| Attention Heads (Q) | 16 | 32 | 48 |
| KV Heads | 8 | 8 | 8 |
| Context Length | 32,768 | 32,768 | 32,768 |
| Vocab Size | 92,544 | 92,544 | 92,544 |

## Model Size Comparison

| Model | Original Size | INT4 Quantized | FP16 | Recommended RAM |
|-------|--------------|----------------|------|-----------------|
| **InternLM2-1.8B** | ~3.6 GB | ~1.0 GB | ~3.6 GB | 4 GB |
| **InternLM2-7B** | ~14 GB | ~3.8 GB | ~14 GB | 8 GB |
| **InternLM2-20B** | ~40 GB | ~10.5 GB | ~40 GB | 24 GB |

**Recommendations:**
- **1.8B**: Great for CPU inference, testing, and rapid prototyping
- **7B**: Best balance of quality and performance, recommended for most applications
- **20B**: Highest quality, requires GPU for practical inference

## Quick Start Scripts

We provide convenient scripts for exporting InternLM2-7B:

**Linux/Mac:**
```bash
cd examples/python/internlm2
chmod +x export_7b.sh
./export_7b.sh
```

**Windows (PowerShell):**
```powershell
cd examples\python\internlm2
.\export_7b.ps1
```

## References

- Model Hub (1.8B): https://huggingface.co/internlm/internlm2-1_8b
- Model Hub (7B): https://huggingface.co/internlm/internlm2-7b
- Model Hub (20B): https://huggingface.co/internlm/internlm2-20b
- Paper: https://arxiv.org/abs/2403.17297
- GitHub: https://github.com/InternLM/InternLM
