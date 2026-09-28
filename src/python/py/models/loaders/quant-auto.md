# quant_auto Format

Models using `quant_method: quant_auto` in their `quantize_config.json` are supported by the `QuantAutoModel` loader (`src/python/py/models/loaders/quant_auto.py`).

## Tensor layout

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `*.weight` | `[out, in]` | F16 | Unpacked; one integer value (0–15) per element |
| `*.scales` | `[out*n_groups, 1]` | F16 | Output-first flat order |
| `*.zeros`  | `[out*n_groups, 1]` | F16 | Same shape as scales; renamed to `.qzeros` during load |
| `embed_tokens.weight` | `[vocab, hidden]` | F16 | Quantized; dequantized to F16 at export |

Key properties:

- Weights are stored as **unpacked F16** (one integer value per element, range 0–15). No bitwise packing.
- `bits` and `group_size` are **absent** from the config; `bits` defaults to 4 and `group_size` is inferred from tensor shapes (`hidden // (scales.numel() // vocab)`).
- Zero points use the `.zeros` suffix (not `.qzeros`); the loader normalises this on read.
- Fused projections (`qkv_proj`, `gate_up_proj`) are stored in `(out, in)` order and split on dim=0 using group-size-aware boundaries.

## Tied embedding and lm_head

When `tie_word_embeddings=True` the embedding weight and lm_head weight are the same tensor. The loader:

1. **Embedding Gather**: dequantizes to F16 using the trained scales/zeros so token lookups return proper activations.
2. **lm_head**: keeps the native asymmetric int4 quantization (trained scales/zeros retained as a `QuantizedTensorModule`) so the builder emits a `MatMulNBits` node with the original zero-point. Re-deriving a symmetric RTN scheme suppresses the low-magnitude EOS logit and causes repetition/no-EOS — this path avoids that.
