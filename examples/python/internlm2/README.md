# InternLM2 Model Export Example

This example demonstrates how to export InternLM2 models to ONNX format using ONNX Runtime GenAI.

## Supported Models

- InternLM2-1.8B
- InternLM2-7B
- InternLM2-20B

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

### INT4 AWQ Quantization (Better Quality)

```bash
python -m onnxruntime_genai.models.builder \
    --input internlm/internlm2-1_8b \
    --output ./internlm2-1.8b-onnx-cpu-int4-awq \
    --precision int4 \
    --execution_provider cpu \
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

| Feature | InternLM2-1.8B |
|---------|----------------|
| Parameters | 1.8B |
| Hidden Size | 2048 |
| Layers | 24 |
| Attention Heads (Q) | 16 |
| KV Heads | 8 |
| Context Length | 32,768 |
| Vocab Size | 92,544 |

## References

- Model Hub: https://huggingface.co/internlm/internlm2-1_8b
- Paper: https://arxiv.org/abs/2403.17297
- GitHub: https://github.com/InternLM/InternLM
