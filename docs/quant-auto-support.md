# quant_auto Support

## Context

A model using `quant_method: quant_auto` could not be exported through OGA's model builder. The format uses a custom quantization scheme developed for NPU deployment that differs from all four formats OGA previously supported (AWQ, GPTQ, Olive, Quark).

**Files changed:**
- `src/python/py/models/loaders/quant_auto.py` — new `QuantAutoModel` loader
- `src/python/py/models/loaders/base.py` — `.zeros` → `.qzeros` name normalization
- `src/python/py/models/loaders/quant_model.py` — dispatch for `quant_auto`
- `src/python/py/models/builders/base.py` — tied-embedding export fixes

---

## Root Cause Analysis: Six Layers of Incompatibility

### 1. Tensor Naming — `.zeros` not recognized

**Problem:** The quant_auto format uses `.zeros` as the zero-point suffix. OGA's regex patterns only match `(qzeros|weight_zero_point)`. Every `.zeros` tensor fell through to `NotImplementedError`.

**Fix:** Added one line to `normalize_vlm_weight_name()`:
```python
name = re.sub(r'\.zeros$', '.qzeros', name)
```
This remaps `.zeros` → `.qzeros` before any regex matching, requiring zero changes to the 400+ lines of existing pattern code.

---

### 2. Fused-Layer Split Direction and Boundary

**Problem — split direction:** `qkv_proj` and `gate_up_proj` are stored in `(out_features, in_features)` layout (same as Olive), requiring a dim=0 split. OGA's non-Olive path splits on dim=1, which produces the wrong result for GQA models where Q ≠ KV output size.

**Fix:** Extended the existing Olive dim=0 split condition to include `quant_auto` for `qweight`:
```python
# Before:
if quant_type == "olive":
# After:
if quant_type in {"olive", "quant_auto"}:
```

**Problem — split boundary for scales/zeros:** Unlike Olive (which stores scales as a 2D `(out, n_groups)` matrix), quant_auto stores scales and zeros as a flat `(out*n_groups, 1)` column vector in output-first order. Splitting a flat `(491520, 1)` QKV scales tensor at row `q_size=3072` gives only 3072 rows instead of the required `q_size * n_groups = 294912` rows — leaving the other projections with wrong scale assignments.

**Fix:** Separate branches for `quant_auto` scales/zeros that compute the split boundary as `q_size * ng` (not just `q_size`), preserving the flat output-first format that `pack_ort_format` expects:
```python
if quant_type == "quant_auto":
    qkv_out = q_size + kv_size + kv_size
    ng = tensor.shape[0] // qkv_out          # infer n_groups from tensor shape
    q_rows  = q_size  * ng
    kv_rows = kv_size * ng
    tensor_map["self_attn.q_proj.scales"] = tensor[:q_rows, :]
    tensor_map["self_attn.k_proj.scales"] = tensor[q_rows : q_rows + kv_rows, :]
    tensor_map["self_attn.v_proj.scales"] = tensor[q_rows + kv_rows :, :]
```
Same pattern applied to qzeros for QKV, and scales/zeros for gate_up_proj using `intermediate_size * ng` as the midpoint.

---

### 3. Container Format — No Bitwise Packing, uint8 Zero Points

**Problem:** AWQ/GPTQ/Olive/Quark all store weights in packed int32 or uint8 containers (multiple int4 values per element). quant_auto stores weights as float16 with one integer value per element (range 0–15) — no packing at all. Calling the standard `unpack_on_row()` path on float16 data produces garbage via `torch.bitwise_right_shift` on floats.

**Fix:** `QuantAutoModel.repack()` bypasses the unpack step entirely and calls `pack_ort_format()` directly after a dtype cast:
```python
def repack(self, module):
    if module.qzeros is not None:
        # Reshape flat (out*ng,1) → (out,ng) → transpose → (ng,out) so
        # pack_zeros_ort_format's internal .T gives (out,ng) → ORT output-first layout
        ng = module.qzeros.numel() // module.out_features
        module.qzeros = module.qzeros.reshape(module.out_features, ng).T.to(torch.uint8).contiguous()
    intweight = module.qweight.to(torch.int32)   # F16 int values → int32
    self.pack_ort_format(module, intweight.T)     # expects (in_features, out_features)
```

The zero point transpose is required because `pack_zeros_ort_format` applies its own `.T` internally before packing — so entering as `(ng, out)` produces the `(out, ng/2)` byte layout ORT's `MatMulNBits` kernel expects.

---

### 4. Shape Inference — No Packing Factor

**Problem:** `set_properties()` derives `in_features` from `qweight.shape[1] * 8 // bits` for Olive (which uses uint8 packing, so each column holds 2 values). For quant_auto, weight shape is `(out, in)` directly — no packing factor applies.

**Fix:** `QuantAutoModel.set_properties()` reads shapes directly:
```python
proj.out_features = proj.qweight.shape[0]
proj.in_features  = proj.qweight.shape[1]    # no * 8 // bits
n_groups = proj.scales.reshape(proj.out_features, -1).shape[1]
proj.group_size = proj.in_features // n_groups
```

---

### 5. Config Loading — bits/group_size Absent

**Problem:** `QuantizedModel.__init__` reads `config["bits"]` and `config["group_size"]` directly. The quant_auto config only contains `{"quant_method": "quant_auto"}` — accessing missing keys raises a `KeyError` immediately.

**Fix:** `QuantAutoModel.__init__` extracts the values with `.get()` before calling `super().__init__()`, passing them as keyword arguments so the base class never touches the missing keys:
```python
global_bits = quant_attrs["config"].get("bits", 4)
global_group_size = quant_attrs["config"].get("group_size", -1)
super().__init__(..., global_bits=global_bits, global_group_size=global_group_size)
```

---

### 6. Embedding and lm_head Not Dequantized

**Problem (discovered during inference testing):** The base class stores `model.embed_tokens.weight` as-is. For quant_auto, this is the raw int4 tensor (values 0.0–15.0 in float16). The model builder wrote these integer values directly into the ONNX embedding table, causing every token lookup to return a vector of values like `[3., 7., 8., 3., ...]` instead of proper float activations. This caused logit values of ~18,000 instead of the expected ±15 range.

Additionally, `lm_head` (tied to the embedding via `tie_word_embeddings=True`) was being processed as a separate `QuantizedTensorModule` — packing the raw int4 embedding values through `pack_ort_format` and producing a `MatMulNBits` node with completely wrong scales.

**Fix:** `QuantAutoModel.dequantize_embedding()` runs after `super().__init__()` and loads the embedding's scales and zeros from the safetensors files. The group size is inferred from tensor shapes (`ng = sc.numel() // vocab; gs = hidden // ng`). The **embedding Gather** table is dequantized to float16 so token lookups return proper activations:
```python
w_dq = ((w.float().reshape(-1, gs) - zp.float()) * sc.float()).reshape(w.shape).half()
self.embedding.weight = w_dq
```

The tied **lm_head** is handled separately — see next section for why it is kept quantized rather than dequantized.

---

### 7. Tied lm_head: Native Asymmetric int4 (EOS Fix)

**Problem (discovered during generation testing):** An earlier fix dequantized the tied lm_head to float16 and let the builder handle it. This produced runnable output but the model rarely emitted EOS — generations ran to the token cap with tail repetition / word-salad degeneration.

**Root cause — the zero-point was silently dropped:**
1. A float16 lm_head has no `qweight`, so `make_matmul_int4` falls back to a plain float `MatMul` (base.py) carrying **no quantization metadata** — bits, group_size, and zero-point are all gone.
2. At save time `to_int4()` runs `MatMulNBitsQuantizer` with `is_symmetric=True` (the default), re-quantizing that float MatMul with a **symmetric** RTN grid — one scale per block, grid pinned to a fixed center, **no zero-point**.

The QAT model was trained on an *asymmetric* grid. Collapsing to fp16 and re-fitting a symmetric grid introduced ~7% RMS logit noise, which suppressed the low-magnitude EOS logit → no-EOS / repetition.

**Fix:** Keep the tied lm_head as a `QuantizedTensorModule` using the model's **own trained** scales/zeros, so it flows through the normal quantized path and exports as a 4-input **asymmetric** `MatMulNBits` (with a zero-point), exactly like every other linear layer. Properties are set here because `set_properties()` already ran during `super().__init__()`, when lm_head was still a plain `TensorModule`:
```python
lm = QuantizedTensorModule()
lm.qweight     = w                 # (vocab, hidden) integer values stored as F16
lm.scales      = sc                # (vocab*ng, 1)
lm.qzeros      = zp                # (vocab*ng, 1)
lm.out_features = w.shape[0]       # vocab
lm.in_features  = w.shape[1]       # hidden
lm.bits         = self.global_bits # 4
lm.group_size   = gs               # 32
self.lm_head = lm
```
The shared `repack(self.lm_head)` in `__init__` then packs it. Because the node is already a `MatMulNBits`, the builder's `to_int4()` pass skips it — the trained zero-point is preserved end-to-end.

---

## Factory Registration

`QuantModel.from_pretrained()` dispatches `"quant_auto"` to the new class:
```python
elif quant_type == "quant_auto":
    model = QuantAutoModel(quant_type, **kwargs)
```

---

## Backward Compatibility

All four existing formats (AWQ, GPTQ, Olive, Quark) are **completely unchanged**. Every `quant_auto` code path is either a new `elif` branch or gated by `quant_type == "quant_auto"` / `quant_type in {"olive", "quant_auto"}`. The `.zeros` normalization in `normalize_vlm_weight_name()` is safe because no existing model tensor ends with the exact suffix `.zeros`.

---

## quant_auto Tensor Format Reference

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `qkv_proj.weight` | `[out, in]` | F16 | `(out, in)` layout, values 0–15 |
| `qkv_proj.scales` | `[out*n_groups, 1]` | F16 | output-first flat order |
| `qkv_proj.zeros`  | `[out*n_groups, 1]` | F16 | Same shape as scales |
| `gate_up_proj.weight` | `[2*intermediate_size, in]` | F16 | `2 × intermediate_size` rows |
| `down_proj.weight` | `[out, in]` | F16 | `(out, in)` |
| `embed_tokens.weight` | `[vocab, hidden]` | F16 | Quantized; dequantized at export |

Key properties: weights stored as unpacked F16 integer values (one value per element), group_size inferred from scales shape, bits/group_size absent from config (use defaults: bits=4, group_size inferred per-module).
