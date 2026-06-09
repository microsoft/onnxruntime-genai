---
name: new-model
description: >-
  Add support for a new model architecture to onnxruntime-genai's Python model
  builder, and debug numerical parity / quality issues in the exported ONNX
  model. Use when asked to "add a new model", "support <arch> in the model
  builder", export a HuggingFace model to ONNX GenAI format, or when an exported
  model produces garbage/incoherent output, fails ShapeInferenceError, or has
  low logits correlation vs the reference PyTorch model. Covers builder
  dispatch, the State/builder pattern, MoE/QMoE quantization encoding, and a
  systematic parity-debugging workflow.
---

# Adding a New Model to onnxruntime-genai

This skill explains how to (1) add a new model architecture to the Python model
builder and (2) debug numerical parity problems in the exported model.

## 1. Orientation — key files

- `src/python/py/models/builder.py` — top-level entry. `create_model()` dispatches
  on `config.architectures[0]` (the HF architecture string) to a concrete builder
  class. New architectures are wired in here.
- `src/python/py/models/builders/base.py` — the `Model` base class. Contains all
  shared graph-construction helpers (`make_*` methods), attention, MLP, MoE/QMoE,
  rotary embedding, KV-cache, and quantization logic. **Read this first** — most
  new models reuse 90%+ of it.
- `src/python/py/models/builders/<family>.py` — per-family subclasses (e.g.
  `qwen.py`, `llama.py`, `gptoss.py`). A new model usually subclasses an existing
  family builder or `Model` directly and overrides only what differs.
- `src/python/py/models/README.md` and `DESIGN.md` — supported models, extra
  options, and design principles. Honor the `.github/instructions/python-model-builder.instructions.md`
  rules (prefer `self.make_<op>` wrappers; reduce duplication via base-class reuse).

The C++ runtime side (only needed for new model *types*, not new arches that reuse
an existing type):
- `src/models/model_type.h` — canonical list of LLM/VLM/ALM type strings.
- `src/config.cpp` / `src/config.h` — `genai_config.json` schema.
- `src/models/` — `State` subclasses, processors, position inputs, caches.

## 2. Decide: is this really a *new* architecture?

Before writing any code, compare the new model's `config.json` to existing ones:

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<hf_id>'); print(c.architectures, c.model_type)"
```

- If `architectures[0]` already matches an existing dispatch branch in
  `builder.py`, the model may build with **zero code changes** — just try it.
  (Example: a point-release that keeps the same arch class string as its
  predecessor. Always test this hypothesis first; it saves days.)
- If only a few hyperparameters differ (rope theta, layer counts, head dims,
  MoE expert counts), the existing builder usually flows them through `config`
  unchanged. Add a new dispatch branch only when graph *structure* differs.

## 3. Adding the builder

1. **Dispatch**: add an `elif config.architectures[0] == "<ArchString>":` branch
   in `create_model()` that instantiates your builder class.
2. **Builder class**: subclass the closest existing family builder. Override
   `__init__` (to set arch-specific attrs) and the specific `make_*` methods that
   differ (e.g. `make_attention`, `make_mlp`, `make_moe`, rotary cache builders).
3. **Reuse**: route every emitted node through the `self.make_<op>` wrappers so
   shapes/values are registered. Do not hand-roll `make_node` + `make_value`.
4. **Weights**: the base loader iterates HF weights by name. If the new model
   packs/repacks weights (e.g. interleaving gate/up for fused SwiGLU, or splitting
   QKV), do the repack in the builder and keep names aligned with the op inputs.

### MoE / QMoE specifics (high-bug-risk area)

- `make_moe_op` emits `MoE` (fp16) or `QMoE` (int4/int8). `make_qmoe_weights`
  quantizes and packs each expert weight `[N, K]`.
- **CUDA QMoE weight encoding (critical):** the kernel is a CUTLASS fpA_intB
  mixed GEMM that consumes **offline-prepacked** weights. The proven recipe
  (see `_cutlass_prepacked_blockwise_quantize` in `base.py`):
  1. transpose weight to `[K, N]`;
  2. `onnxruntime...quantize_matmul_4bits(qw, w_T, scales, zp, block, N, K, is_symmetric=True)`;
  3. **keep the SIGNED scales** — do NOT `abs()` them. The kernel dequantizes as
     `(q - 8) * scale`, and `quantize_matmul_4bits` folds the block-anchor sign
     into the scale. Taking `abs()` corrupts every block whose anchor is negative
     and produces garbage (this is a real bug that masquerades as "int4 quality
     loss");
  4. `pack_weights_for_cuda_mixed_gemm(qw_reshaped, N, K, bits, force_arch=80)` —
     **always force_arch=80**: all int4 QMoE prepacking assumes the SM80-style
     interleaved layout, which is correct for every SM ≥ 80 (Ampere/Ada/Hopper,
     incl. RTX 4090 = SM89 and H100/H200 = SM90);
  5. reshape to `[K, N/pack]`. Stack experts → weights `[E, K, N/pack]`, scales
     `[E, N, K/block]`.
- The QMoE node then uses the **default** `weights_prepacked` (omit the attribute;
  default = prepacked). Do **not** set `weights_prepacked=0` (the raw-weight +
  runtime-PrePack-hook path is finiteness-checked only and is not bit-correct).
- **CUDA QMoE only supports `block_size` 64 or 128.** Assert this in the builder.
- Emit the activation attributes that match the model's activation: for standard
  SwiGLU `silu(gate)*up`, use `activation_alpha=1.0, activation_beta=0.0` and no
  `swiglu_limit`. GPT-OSS-style clamped SwiGLU uses `alpha=1.702, beta=1.0,
  swiglu_limit=7.0`. Wrong activation attrs silently degrade parity.
- `swiglu_fusion=1` expects gate/up **interleaved** `[g0,u0,g1,u1,...]`; repack HF
  concatenated `[gate|up]` accordingly before quantizing.
- The router (gate) MatMul is **external** to QMoE and runs in fp16. To keep
  routing exact under int4, add the router and any shared-expert-gate MatMul node
  names to `quant_attrs["nodes_to_exclude"]`.

## 4. Exporting

Models are usually exported via an Olive recipe (`olive run --config <text.json>`)
or directly:

```bash
python -m onnxruntime_genai.models.builder -m <hf_id> -o <out_dir> -p int4 -e cuda \
  --extra_options int4_block_size=128
```

After editing the builder in `src/python/py/models/`, the **installed** package is
what runs. Copy your edits to every install location, or reinstall:

```bash
SRC=src/python/py/models/builders/base.py
for d in $(python -c "import onnxruntime_genai,os;print(os.path.join(os.path.dirname(onnxruntime_genai.__file__),'models','builders'))"); do cp "$SRC" "$d/"; done
```

When re-exporting with Olive, clear its cache so the modelbuilder pass actually
re-runs: `rm -rf .olive-cache/<workflow>`.

## 5. Debugging parity / garbage output (systematic workflow)

When the model loads but generates garbage or has low logits correlation, isolate
the error layer-by-layer and component-by-component. **Do not guess** — bisect.

### Metrics: how `corr` and `relL2` are computed

Throughout this workflow two scalar metrics quantify how close a candidate tensor
`a` is to a reference tensor `b` (same shape, flattened):

- **`corr`** — Pearson correlation of the flattened tensors. Measures *shape /
  direction* agreement and is scale-invariant, so it stays high even if `a` is a
  uniformly scaled version of `b`. `corr ≈ 1.0` is good; `corr ≈ 0` means
  uncorrelated (random) output.
- **`relL2`** — relative L2 (Euclidean) error, `‖a − b‖₂ / ‖b‖₂`. Measures
  *magnitude* error and is NOT scale-invariant. `relL2 ≈ 0` is perfect; `relL2 ≈
  √2 (~1.41)` for zero-mean tensors means `a` is effectively random w.r.t. `b`.

Use them together: high `corr` with high `relL2` points at a **scale/sign bug**
(right direction, wrong magnitude — e.g. dropped scale sign); low `corr` means the
values are scrambled (a layout/encoding bug). Example helper:

```python
import numpy as np
import torch

def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])

def relL2(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return float(torch.norm(a - b) / (torch.norm(b) + 1e-9))

# Example: compare an ONNX op output against a PyTorch reference
print(f"corr={corr(onnx_out, ref):.4f} relL2={relL2(onnx_out, ref):.4f}")
```

When comparing next-token logits, additionally check that the **argmax matches**
(`int(a.argmax()) == int(b.argmax())`) — argmax agreement is what actually
determines greedy-decoding correctness.

### Step 0 — Establish ground truth
Run the HF model in PyTorch and capture next-token logits for a fixed prompt.
Good parity: ONNX-vs-HF next-token correlation ≥ 0.99 and matching argmax.

### Step 1 — fp16 first (separate structure from quantization)
Export an **fp16 (non-quantized)** variant and compare to HF.
- fp16 corr ≈ 1.0  ⟹ the graph, kernels, rotary embeddings, caches, and runtime
  are all correct; any remaining problem is **purely quantization**.
- fp16 corr still low ⟹ a **structural** bug (wrong rotary, wrong cache wiring,
  wrong attention/MLP graph, transposed weights). Fix this before touching quant.

### Step 2 — Layer-by-layer residual probe
Add the per-layer residual tensors (e.g. `/model/layers.{n}/input_layernorm/output`)
as extra graph outputs in **both** the int4 and fp16 models, feed identical
`inputs_embeds`, and compare each layer's `corr`/`relL2`. Find the first layer
where correlation collapses and which **layer type** (attention vs MoE vs linear-
attention) degrades fastest.

### Step 3 — Sub-component isolation
Within the worst layer, probe sub-tensors (attn output, post-attn residual, MoE
output, layer residual) int4-vs-fp16 with the **same inputs**. This pinpoints the
offending op (e.g. `moe_out relL2 0.4` ⟹ the QMoE path).

### Step 4 — Op-level standalone test
Reproduce the suspect op in isolation with small synthetic weights and compare:
- the ORT op output **vs a pure-PyTorch reference built from the ORIGINAL
  (un-quantized) weights**. **Crucial:** do NOT validate a quantized op only
  against a torch reference that uses the *same* (de)quantization — a sign/scale
  bug can be self-consistent and pass while still corrupting real weights. Always
  anchor against the original full-precision weights.
- Sweep encoding conventions (transpose y/n, offset 8/zp, signed vs abs scale,
  unpack order) and pick the one that reconstructs the original weights with high
  corr and low relL2.

### Step 5 — Weight-fidelity sanity check
Quantize→dequantize a real weight matrix and measure `relL2` vs the original.
`relL2 ≈ √2` (~1.41) means the dequant is **random** (encoding bug), not merely
lossy. Healthy int4 RTN is ~0.05–0.15. This quickly distinguishes an encoding bug
from genuine quantization quality loss.

### Common real bugs (seen in practice)
- **abs() on signed quant scales** → garbage that looks like "int4 quality loss".
- **Wrong CUTLASS `force_arch`** in prepacking → finite but wrong output.
- **Duplicate node/value names** (e.g. two `.../Mul` outputs in a shared-expert
  subgraph) → `ShapeInferenceError` / wrong tensor picked downstream. Give each
  emitted node a unique name.
- **Missing/incorrect activation attrs** (`activation_alpha/beta`, `swiglu_limit`)
  → degraded but not random.
- **Quantizing routing-critical tiny matmuls** (router, gate) → wrong expert
  selection; exclude them from int4.
- **Running Python from a directory that shadows the installed package** (e.g. a
  local `onnxruntime/` folder) → stale code. Run from a neutral dir.

## 6. Validation checklist
- [ ] fp16 export: next-token corr ≥ 0.99 vs HF, coherent greedy text.
- [ ] int4 export: per-layer corr stays high; final next-token corr ≥ ~0.95.
- [ ] Coherent end-to-end generation (text and, for VLMs, image prompts).
- [ ] Memory footprint fits the target GPU.
- [ ] TTFT / tokens-per-second benchmarked.
- [ ] Builder edits copied to all install locations; Olive cache cleared before
      re-export.
