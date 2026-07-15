# Model Builder Quantization Config — Design

> **Status:** Draft for review. This document proposes a *unified* quantization
> configuration for the ONNX Runtime GenAI model builder. It is design-only; no code
> has been changed yet. The mechanics of the individual quantizers (RTN / k-quant /
> QMoE / CUDA prepack) are documented separately in
> [quantization.md](quantization.md); this doc is about the **option surface** that
> selects and places them.

## 1. Motivation

The model builder's quantization controls have grown organically and are now hard to
reason about, inconsistent, and difficult to extend. Concretely:

1. **`int4_`-prefixed options that are not int4-specific.** `int4_block_size`,
   `int4_is_symmetric`, `int4_accuracy_level`, `int4_op_types_to_quantize`,
   `int4_nodes_to_exclude` all apply to *any* weight bit-width. Once int8 (PR
   [#2275](https://github.com/microsoft/onnxruntime-genai/pull/2275)) and future fp8 /
   mxfp4 land, the prefix is misleading.
2. **`--precision` conflates two independent things:** the model's I/O (activation)
   dtype and the *weight* quantization type. `precision=int4` means "int4 weights **and**
   fp16 I/O"; there is no clean way to say "int8 weights with bf16 I/O".
3. **Quant type is encoded three different ways.** A CLI `--precision` string
   (`int4`), a bare bit count in `customized_weight_config` (`{"bits": 8}`), and a
   quant-type name (`moe_quant_type=mxfp4`, `matmul_mixed_precision=last_matmul:int8`).
   None of them can express fp8 variants (e4m3 vs e5m2), mxfp4, or nvfp4 uniformly.
4. **Per-node control is limited.** You can exclude nodes (`int4_nodes_to_exclude`) or
   upgrade a *fixed* set of selectors (`matmul_mixed_precision` →
   `last_matmul` / `mixed_layers` / `linear_attn`). There is no general
   per-layer / per-node recipe (by name, regex, layer range, or module role).
5. **No KV-cache quantization surface** for GroupQueryAttention.
6. **Flat `KEY=VALUE` extra_options only.** Olive and `olive-recipes` already drive the
   builder from JSON, and their RTN/GPTQ passes already express **per-node
   `overrides`** — but that structure is flattened away before it reaches the builder.

The goal of this design: **one dtype vocabulary, one structured config**, that covers
today's needs (int4 / int8 / fp16 / bf16), the near-term roadmap
(fp8 e4m3 & e5m2, mxfp4, nvfp4), and future bit-widths (int2, int6, …), with per-node /
per-layer placement, and that can be passed as JSON to consolidate with `olive-recipes`.

## 2. What exists today (analysis)

### 2.1 CLI

| Flag | Values | Meaning |
| --- | --- | --- |
| `--precision` | `int4`, `bf16`, `fp16`, `fp32` (`int8` proposed in #2275) | Sets **both** I/O dtype and weight dtype. |
| `--execution_provider` | `cpu`, `cuda`, `dml`, `webgpu`, `NvTensorRtRtx` | Target EP (affects defaults). |
| `--extra_options` | `KEY=VALUE ...` | Everything else, flat. |

`set_io_dtype()` and `set_onnx_dtype()` in
[builder.py](builder.py) derive the two dtypes from `precision` (+ a few EP/flag special
cases such as `use_cuda_bf16`, `use_webgpu_fp32`, `int4` on CPU → fp32 I/O).

### 2.2 Weight (MatMul) quantization extra_options

| Option | Default | Notes |
| --- | --- | --- |
| `int4_block_size` | 32 | Block size for MatMulNBits. |
| `int4_is_symmetric` | true | int4 ↔ uint4. |
| `int4_accuracy_level` | 4 (cpu/webgpu) / 0 | MatMul activation accumulation type. |
| `int4_op_types_to_quantize` | `("MatMul",)` | e.g. also `Gather`. |
| `int4_nodes_to_exclude` | `[]` | Skip specific nodes. |
| `int4_algo_config` | `default` | Base method: `default` / `rtn` / `k_quant` (+ legacy compound aliases). |
| `matmul_mixed_precision` | — | `selector:quant_type,...`; selectors `last_matmul` / `mixed_layers` / `linear_attn`; types `int4` / `int8`. |
| `use_qdq` | false | Emit QDQ instead of MatMulNBits. |
| `matmulnbits_weights_prepacked` | 0 | CUDA fpA_intB layout (0/1/2). |

The base-method/mixed-precision split and the legacy aliases (`rtn_last`,
`k_quant_mixed`, …) are described in [quantization.md](quantization.md). Internally the
per-node placement is carried as `customized_weight_config = {node_name: {"bits": N}}`.

### 2.3 MoE (QMoE) extra_options

| Option | Default | Notes |
| --- | --- | --- |
| `moe_quant_type` | `int4` | `int4` / `int8` / `mxfp4`. |
| `use_8bits_moe` | false | **Deprecated** → `moe_quant_type=int8`. |
| `qmoe_block_size` | 32 (128 trt-rtx) | `<=0` = per-channel. |
| `qmoe_weights_prepacked` | -1 | CUDA QMoE layout (-1 auto / 0 / 1). |

### 2.4 PR #2275 (add int8 precision)

PR [#2275](https://github.com/microsoft/onnxruntime-genai/pull/2275) (approved) adds a
`--precision int8` option, generalizes the post-export MatMulNBits quantization pass
(`to_nbits`) to a configurable bit-width (4 or 8), routes INT8/UINT8 through it in
`save_model`, and renames `make_matmul_int4` → `make_matmul_op`. It is a natural, minimal
step, but it extends the *same* entangled surface (a new `precision` enum value, an
implicit int8 dtype) rather than introducing a general quant-type model. This design
subsumes it: `int8` becomes one value in the unified vocabulary, and `--precision int8`
becomes a preset.

## 3. Design overview

Two ideas do the heavy lifting:

1. A **unified quant-dtype vocabulary** — a single set of dtype strings used everywhere a
   quantization type is named (dense weights, MoE experts, KV cache, per-node overrides).
2. A **structured `quantization` config** (JSON) that is *target-oriented*: it describes
   what to do to each part of the model (dense weights, embeddings, LM head, MoE experts,
   KV cache) and lets any target be refined by ordered **per-node / per-layer overrides**.

The existing flat options become thin **back-compat aliases** that desugar into this
config, so existing recipes and CLIs keep producing byte-identical models.

## 4. Unified quant-dtype vocabulary

A dtype string names a storage/quantization scheme. It resolves to a descriptor with a
`kind`, bit-width, float exponent/mantissa (where applicable), signedness, and a default
block-scaling scheme.

| Dtype | Kind | Bits | Details |
| --- | --- | --- | --- |
| `fp32` | float | 32 | No weight quantization (pass-through / I/O). |
| `fp16` | float | 16 | No weight quantization (pass-through / I/O). |
| `bf16` | float | 16 | No weight quantization (pass-through / I/O). |
| `int8` / `uint8` | int | 8 | Signed / unsigned integer. |
| `int4` / `uint4` | int | 4 | Signed / unsigned integer (today's default). |
| `int2` / `int6` … | int | 2 / 6 | Future integer widths (schema already accepts them). |
| `fp8_e4m3` | float | 8 | 1-4-3, OCP FP8. |
| `fp8_e5m2` | float | 8 | 1-5-2, OCP FP8. |
| `fp4_e2m1` | float | 4 | 1-2-1 FP4 (unscaled). |
| `mxfp4` | mx | 4 | e2m1 weights + `ue8m0` (float8_e8m0) block scale, block 32. |
| `mxfp8_e4m3` / `mxfp8_e5m2` | mx | 8 | FP8 weights + `ue8m0` block scale, block 32. |
| `nvfp4` | nv | 4 | e2m1 weights + `fp8_e4m3` block scale (block 16) + per-tensor fp32 global scale. |

Notes:

- **Signedness / symmetry.** For integer dtypes, `int*` is the symmetric storage and
  `uint*` the asymmetric one; equivalently, the dtype can be `int` with `symmetric:
  false`. Both spellings are accepted and normalize to the same descriptor (mirrors
  today's `int4` ↔ `uint4` via `int4_is_symmetric`).
- **Block scaling is part of the dtype for `mx`/`nv`.** `mxfp4` fixes block 32 with a
  `ue8m0` scale; `nvfp4` fixes block 16 with an fp8 scale + fp32 global. For plain
  `int*` / `fp8_*`, block scaling is controlled by the `block_size` field (§5).
- **Extensibility.** Adding a dtype = adding one row to a descriptor table; no new option
  name is introduced. This is the property today's `matmul_mixed_precision` already aims
  for by using type *names* rather than bit counts.

## 5. The `quantization` config

A single object, `quantization`, is the primary surface. Every field is optional; an
empty object means "no quantization, I/O follows `io_dtype`".

```jsonc
{
  "quantization": {
    "io_dtype": "fp16",            // activation / model I/O dtype (fp16 | bf16 | fp32)

    "weights": {                    // dense (non-MoE) weight quantization
      "type": "int4",              // any dtype from §4 (or "none" to skip)
      "block_size": 32,            // int > 0, or 0 / "per_channel" for per-channel
      "symmetric": true,           // int only; ignored for float/mx/nv
      "method": "rtn",             // default | rtn | k_quant | gptq | awq | hqq
      "accuracy_level": 0,          // MatMul activation accumulation type
      "op_types": ["MatMul"],      // which op types to quantize (e.g. add "Gather")
      "overrides": [                // ordered; first match wins (§6)
        { "match": { "preset": "last_matmul" }, "type": "int8" }
      ]
    },

    "embeddings": { "type": "int4", "op_type": "Gather" },  // role shortcut (§6.2)
    "lm_head":    { "type": "int8" },                        // role shortcut

    "moe": {                        // MoE expert weights (QMoE)
      "type": "mxfp4",             // int4 | int8 | mxfp4 | nvfp4 | fp8_* ...
      "block_size": 32,
      "weights_prepacked": -1       // CUDA QMoE layout (-1 auto | 0 | 1)
    },

    "kv_cache": {                   // FUTURE (§8) — GroupQueryAttention KV quant
      "type": "fp8_e4m3",
      "block_size": 0
    },

    "runtime": {                    // layout / emission knobs (not numeric)
      "use_qdq": false,
      "matmulnbits_weights_prepacked": 0
    }
  }
}
```

### 5.1 `block_size` and per-channel

`block_size` is an integer group size. `0` (or the string `"per_channel"`) selects
per-channel quantization (one scale per output channel). This unifies today's
`int4_block_size` and `qmoe_block_size<=0` into a single convention. For `mx`/`nv`
dtypes the block size is fixed by the dtype and this field is ignored (a mismatch is a
validation error).

### 5.2 `io_dtype` is decoupled from `weights.type`

The model's activation/I/O dtype is set independently of the weight quant type. This is
the key fix for problem #2: `weights.type=int8` + `io_dtype=bf16` is now expressible. EP
constraints still apply and are validated (e.g. fp16 I/O is unsupported on the CPU EP;
bf16 I/O only on CUDA), reusing the logic in `set_io_dtype()`.

## 6. Per-node / per-layer overrides

Each target (`weights`, `moe`, …) accepts an ordered `overrides` list. Each entry has a
`match` and the fields to apply (`type`, `block_size`, `symmetric`, `method`, or
`exclude: true`). **First matching entry wins**; unmatched nodes use the target default.

### 6.1 Match selectors

| Selector | Matches | Example |
| --- | --- | --- |
| `name` | Exact node name | `"/lm_head/MatMul"` |
| `name_regex` | Regex over node name | `".*/mlp/down_proj/MatMul"` |
| `layers` | Layer index list or range | `[0, 1, 2]` or `"0-3,31"` |
| `role` | Module role (see §6.2) | `"down_proj"` |
| `op_type` | ONNX op type | `"MatMul"` / `"Gather"` / `"QMoE"` |
| `preset` | Named group (reuses today's selectors) | `"last_matmul"` / `"mixed_layers"` / `"linear_attn"` |

Multiple keys in one `match` are ANDed (e.g. `layers: "0-3"` + `role: "down_proj"`).

### 6.2 Roles

Roles are stable, model-agnostic names for the standard projections the builder already
emits, so a recipe need not know exact node paths:
`qkv_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
`lm_head`, `embed`, and the linear-attention projections `in_proj_a/b/qkv/z`, `out_proj`.
The top-level `embeddings` and `lm_head` keys are sugar for `role: embed` / `role:
lm_head` overrides.

### 6.3 Presets keep the current mixed-precision behavior

The three existing selectors are preserved as `preset` values with identical semantics
(`last_matmul` = the LM head; `mixed_layers` = llama.cpp's sensitive-layer set;
`linear_attn` = hybrid-attention projections + their MLPs), so
`matmul_mixed_precision=last_matmul:int8` maps to
`overrides: [{ "match": { "preset": "last_matmul" }, "type": "int8" }]`.

## 7. Passing the config to the builder

Three equivalent entry points, in precedence order (later overrides earlier):

1. **`--config path/to/quant.json`** — a JSON file whose top-level `quantization`
   object is loaded directly. (May also point at a full builder config; only the
   `quantization` key is read here.)
2. **`--extra_options quant_config=<path-or-inline-json>`** — same object, inline or by
   path, for callers that only have the flat extra_options channel.
3. **Legacy flat options** (`int4_*`, `moe_quant_type`, `matmul_mixed_precision`, …) —
   desugared into the config (§9), then merged. Explicit `quant_config` fields win over
   legacy flags on conflict.

### 7.1 Olive / olive-recipes consolidation

Olive's `ModelBuilder` pass (`olive/passes/onnx/model_builder.py` in the Olive repo)
gains a single `quantization` `PassConfigParam` (a dict) that is passed straight through
as `quant_config`. This lets an `olive-recipes` file express the whole quant recipe in
the same JSON it already uses, and the per-node `overrides` in Olive's RTN/GPTQ passes
map 1:1 onto `weights.overrides`. Example recipe fragment:

```jsonc
{
  "type": "ModelBuilder",
  "precision": "int4",
  "quantization": {
    "weights": {
      "type": "int4", "block_size": 128, "method": "rtn",
      "overrides": [
        { "match": { "role": "lm_head" },  "type": "int8" },
        { "match": { "role": "embed" },    "type": "int8" }
      ]
    }
  }
}
```

## 8. KV-cache quantization (future)

GroupQueryAttention will gain an optional quantized KV cache. The surface is designed now
so recipes are forward-compatible, but it is **not yet implemented**.

```jsonc
"kv_cache": { "type": "int8" }          // or fp8_e4m3 / fp8_e5m2 / int4 / fp4_e2m1
```

`type` uses the same §4 vocabulary; `block_size` (default per-token/per-head, TBD)
controls grouping. Separate K and V settings may be added later via
`kv_cache.key` / `kv_cache.value` sub-objects if they need to differ. This section is
marked **designed, not implemented**.

## 9. Back-compat mapping (flat option → config)

Every current option desugars deterministically; models remain byte-identical.

| Legacy option | Config path |
| --- | --- |
| `--precision int4` | `io_dtype=fp16`, `weights.type=int4` |
| `--precision int8` (#2275) | `io_dtype=fp16`, `weights.type=int8` |
| `--precision fp16/bf16/fp32` | `io_dtype=<same>`, `weights.type=none` |
| `int4_block_size` | `weights.block_size` |
| `int4_is_symmetric` | `weights.symmetric` (or `type` int↔uint) |
| `int4_accuracy_level` | `weights.accuracy_level` |
| `int4_op_types_to_quantize` | `weights.op_types` |
| `int4_nodes_to_exclude` | `weights.overrides += {match:{name}, exclude:true}` |
| `int4_algo_config` (base) | `weights.method` |
| `int4_algo_config` (legacy alias) | `weights.method` + preset overrides (§6.3) |
| `matmul_mixed_precision=sel:type` | `weights.overrides += {match:{preset:sel}, type}` |
| `moe_quant_type` | `moe.type` |
| `use_8bits_moe=true` | `moe.type=int8` |
| `qmoe_block_size` | `moe.block_size` (`<=0` → per-channel) |
| `qmoe_weights_prepacked` | `moe.weights_prepacked` |
| `matmulnbits_weights_prepacked` | `runtime.matmulnbits_weights_prepacked` |
| `use_qdq` | `runtime.use_qdq` |

Deprecation is *soft*: legacy options keep working, emit no warning initially, and can be
warned/removed on a later major version.

## 10. Worked examples

**(a) int4 body + int8 LM head** (== legacy `rtn_last`):

```jsonc
"quantization": {
  "io_dtype": "fp16",
  "weights": { "type": "int4", "block_size": 32, "method": "rtn",
    "overrides": [ { "match": { "role": "lm_head" }, "type": "int8" } ] }
}
```

**(b) k_quant + int8 on sensitive layers** (== legacy `k_quant_mixed`):

```jsonc
"weights": { "type": "int4", "method": "k_quant",
  "overrides": [
    { "match": { "preset": "last_matmul" }, "type": "int8" },
    { "match": { "preset": "mixed_layers" }, "type": "int8" }
  ] }
```

**(c) mxfp4 MoE experts + int4 dense + fp16 I/O:**

```jsonc
"quantization": {
  "io_dtype": "fp16",
  "weights": { "type": "int4", "block_size": 32 },
  "moe": { "type": "mxfp4" }
}
```

**(d) fp8 (e4m3) per-channel weights:**

```jsonc
"weights": { "type": "fp8_e4m3", "block_size": "per_channel" }
```

**(e) nvfp4 weights (block 16 fixed by dtype):**

```jsonc
"weights": { "type": "nvfp4" }
```

**(f) per-layer override by range + role** (down_proj of first 4 layers to int8):

```jsonc
"weights": { "type": "int4",
  "overrides": [ { "match": { "layers": "0-3", "role": "down_proj" }, "type": "int8" } ] }
```

**(g) future KV-cache fp8 on GQA:**

```jsonc
"quantization": { "weights": { "type": "int4" }, "kv_cache": { "type": "fp8_e4m3" } }
```

## 11. Implementation & migration plan

Staged so each step is independently shippable and testable.

1. **`QuantConfig` model** — a dataclass/schema (in `builders/`) with `from_dict`,
   validation, and a dtype-descriptor table (§4). Pure data + validation; no builder
   wiring yet. Unit-tested standalone.
2. **Legacy adapter** — a `from_extra_options()` that desugars today's flat options into
   `QuantConfig` per the §9 table. Golden test: for a matrix of legacy option sets, the
   resulting `QuantConfig` (and the derived `customized_weight_config` / algo config /
   moe attrs) equals what `make_quant_init` produces today.
3. **Builder consumption** — replace the scattered logic in
   [base.py](builders/base.py) (`make_quant_init`, `resolve_int4_quant_config`,
   `normalize_matmul_mixed_precision`, `make_matmul_mixed_precision`, `make_algo_config`)
   with reads from `QuantConfig`. `to_nbits` builds its per-node type map from
   `weights.overrides` (it already handles a per-node `bits` map). No numeric change ⇒
   existing builder tests must stay green.
4. **CLI/entry points** — add `--config` and `quant_config` handling in
   [builder.py](builder.py); make `--precision` a preset that seeds `io_dtype` +
   `weights.type`. Validate in `check_extra_options`.
5. **Olive passthrough** — add the `quantization` param to Olive's `ModelBuilder` pass;
   document it in `olive-recipes`.
6. **Docs** — fold the option reference into [quantization.md](quantization.md) and link
   this design; add the JSON examples.
7. **New dtypes** (incremental, behind the same surface): fp8_e4m3/e5m2, mxfp8, nvfp4,
   then KV-cache quant (§8). Each is a descriptor-table entry + the corresponding
   quantizer path; no option-surface change.

### Regression guardrails

- The current builder tests (`test_mixed_precision_config.py`, `test_tied_embeddings.py`,
  `test_qmoe_weights.py`) must pass unchanged after step 3.
- A byte-identity check: build a small model with a legacy option set and with its
  desugared `quantization` JSON; assert the two ONNX outputs are identical.

## 12. Open questions

1. `quant_config` accepts both a path and an inline JSON string — confirm both are
   desired (recommended: yes, to match Olive's dict passthrough).
2. Should `weights.overrides` matching short-circuit on first match (proposed) or merge
   fields across all matches? First-match is simpler and predictable; merge is more
   expressive. Proposed: first-match, revisit if needed.
3. Naming: `weights` vs `dense` vs `matmul` for the dense-weight target. Proposed:
   `weights`.
4. Separate K/V KV-cache settings — defer until a concrete need appears.
