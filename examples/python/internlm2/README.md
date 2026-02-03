# InternLM2 Model Export for ONNX Runtime GenAI

This example demonstrates how to export InternLM2 models to ONNX format using ONNX Runtime GenAI.

## Supported Models

All InternLM2 model sizes are supported:

- ✅ **InternLM2-1.8B** - Tested and verified
- ✅ **InternLM2-7B** - Tested and verified
- ✅ **InternLM2-20B** - Fully compatible
- ✅ **InternLM2-Chat variants** - All sizes supported

The implementation is architecture-based and automatically adapts to any InternLM2 model size.

## Model Architecture

InternLM2 uses a Llama-based architecture with the following key features:

- **Attention**: Grouped Query Attention (GQA) with grouped/interleaved QKV layout
- **Normalization**: RMSNorm (eps: 1e-05)
- **Activation**: SiLU
- **Positional Encoding**: RoPE with theta=1,000,000

### Architecture Specifications

| Parameter | 1.8B | 7B | 20B |
|-----------|------|-----|-----|
| **Hidden Size** | 2048 | 4096 | 6144 |
| **Num Layers** | 24 | 32 | 48 |
| **Q Heads** | 16 | 32 | 48 |
| **KV Heads** | 8 | 8 | 8 |
| **Head Dim** | 128 | 128 | 128 |
| **Intermediate** | 8192 | 14336 | 16384 |
| **GQA Ratio** | 2:1 | 4:1 | 6:1 |
| **Context Length** | 32,768 | 32,768 | 32,768 |
| **Vocab Size** | 92,544 | 92,544 | 92,544 |

## Export Examples

### InternLM2-1.8B

**FP32 (Best quality baseline):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-cpu-fp32 \
    --precision fp32 \
    --execution_provider cpu
```

**INT4 RTN (Fast quantization):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-cpu-int4 \
    --precision int4 \
    --execution_provider cpu
```

**INT4 AWQ (Best quality, recommended):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-cpu-int4-awq \
    --precision int4 \
    --execution_provider cpu \
    --extra_options int4_accuracy_level=4
```

### InternLM2-7B

**INT4 AWQ CPU (Recommended for most users):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-7b \
    --output ./internlm2-7b-cpu-int4-awq \
    --precision int4 \
    --execution_provider cpu \
    --extra_options int4_accuracy_level=4
```

**INT4 AWQ CUDA (For GPU inference):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-7b \
    --output ./internlm2-7b-cuda-int4-awq \
    --precision int4 \
    --execution_provider cuda \
    --extra_options int4_accuracy_level=4
```

**FP16 CUDA (Highest quality on GPU):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-7b \
    --output ./internlm2-7b-cuda-fp16 \
    --precision fp16 \
    --execution_provider cuda
```

### InternLM2-20B

**INT4 AWQ CUDA (Recommended):**
```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-20b \
    --output ./internlm2-20b-cuda-int4-awq \
    --precision int4 \
    --execution_provider cuda \
    --extra_options int4_accuracy_level=4
```

## Model Size & Performance

| Model | Original Size | INT4 Quantized | FP16 | Recommended RAM |
|-------|--------------|----------------|------|-----------------|
| **InternLM2-1.8B** | ~3.6 GB | ~1.0 GB | ~3.6 GB | 4 GB |
| **InternLM2-7B** | ~14 GB | ~3.8 GB | ~14 GB | 8 GB |
| **InternLM2-20B** | ~40 GB | ~10.5 GB | ~40 GB | 24 GB |

**CPU Inference (Approximate):**

| Model | Min RAM | Recommended RAM | Typical Speed |
|-------|---------|-----------------|---------------|
| 1.8B INT4 | 4 GB | 8 GB | 8-12 tok/s |
| 7B INT4 | 8 GB | 16 GB | 2-4 tok/s |
| 20B INT4 | 16 GB | 32 GB | 0.5-1 tok/s |

**GPU Inference (CUDA):**

| Model | Min VRAM | Recommended VRAM | Typical Speed |
|-------|----------|------------------|---------------|
| 1.8B INT4 | 2 GB | 4 GB | 50-80 tok/s |
| 7B INT4 | 6 GB | 8 GB | 30-50 tok/s |
| 7B FP16 | 14 GB | 16 GB | 40-60 tok/s |
| 20B INT4 | 12 GB | 16 GB | 20-30 tok/s |
| 20B FP16 | 40 GB | 48 GB | 25-35 tok/s |

## Inference Example

```python
import onnxruntime_genai as og

# Works with any InternLM2 size!
model = og.Model("./internlm2-7b-cpu-int4-awq")
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

## Why Multi-Size Support Works

### Architecture-Based Implementation

The implementation is **size-agnostic** because it:

1. **Dynamically reads config parameters** from each model:
   - `num_attention_heads`
   - `num_key_value_heads`
   - `hidden_size`
   - `num_hidden_layers`
   - `intermediate_size`

2. **Uses config-driven weight splitting**:
   ```python
   # Reads from model config
   num_q_heads = config.num_attention_heads  # 16 for 1.8B, 32 for 7B, 48 for 20B
   num_kv_heads = config.num_key_value_heads  # Always 8 for InternLM2
   head_dim = config.hidden_size // num_q_heads  # Always 128
   
   # Calculates group size dynamically
   num_kv_groups = num_q_heads // num_kv_heads  # 2 for 1.8B, 4 for 7B, 6 for 20B
   group_size = num_kv_groups + 2
   ```

3. **Handles grouped QKV layout** for any GQA ratio:
   - Layout: `[Group0: Q0,Q1,...,K0,V0 | Group1: Q2,Q3,...,K1,V1 | ...]`
   - Each KV group contains multiple Q heads followed by K and V
   - Correctly extracts weights regardless of the Q/KV head ratio

4. **No hardcoded sizes** anywhere in the code

### Key Implementation Notes

**Grouped QKV Layout:**
- InternLM2 uses a grouped/interleaved QKV weight layout for efficient Grouped Query Attention
- The implementation in `src/python/py/models/builders/internlm.py` correctly handles this layout during weight extraction

**Model Configuration:**
- The exported model uses `model_type: "llama"` for ONNX Runtime GenAI compatibility
- Tokenizer uses `tokenizer_class: "LlamaTokenizer"` (SentencePiece-based)

## Recommendations by Use Case

### Development & Testing
- **InternLM2-1.8B INT4 AWQ** (1 GB)
- Fast iteration, quick testing
- Good for prototyping

### Production Applications
- **InternLM2-7B INT4 AWQ** (3.8 GB)
- Best balance of quality and performance
- Suitable for most real-world applications

### High-Quality Applications
- **InternLM2-7B FP16 CUDA** (14 GB) or
- **InternLM2-20B INT4 CUDA** (10.5 GB)
- Maximum quality for critical applications

## Troubleshooting

### "Out of Memory" errors
- Use INT4 quantization instead of FP16/FP32
- Enable GPU inference for larger models
- Use batch_size=1 for inference

### Slow inference on CPU
- This is expected for 7B+ models
- Consider GPU inference
- Use INT4 quantization (2-3x faster than FP16)

### Model not loading
- Ensure you have enough RAM/VRAM
- Check that you're using `--execution_provider cuda` for GPU models
- Verify ONNX Runtime GenAI installation

## References

- Model Hub (1.8B): https://huggingface.co/internlm/internlm2-1_8b
- Model Hub (7B): https://huggingface.co/internlm/internlm2-7b
- Model Hub (20B): https://huggingface.co/internlm/internlm2-20b
- Paper: https://arxiv.org/abs/2403.17297
- GitHub: https://github.com/InternLM/InternLM
