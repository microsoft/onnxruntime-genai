# Qwen3.8-Flash-Next (`qwen38_flash_graph`) Package Builder

`tools/python/qwen38_flash_next/build_package.py` procedurally constructs self-contained ONNX Runtime GenAI model packages for **Qwen3.8-Flash-Next** (`model.type: "qwen38_flash_graph"`) + **DFlash2** (`S=8`, 7-draft speculative decoding) directly from the two Hugging Face checkpoints (`Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-DFlash2`), without depending on any pre-exported `.onnx` model.

The default standard-RoPE export with unified caches supports:
- **Single unified `model.onnx` across all context lengths (`32K` / `128K` / `256K`)**: cache capacity dimensions (`kv_capacity` and `state_capacity`) are symbolic and `contrib_const_dsa_total_seq` is set to `[262152]`, so a single `model.onnx` serves `32K` (`num_blocks: 142`), `128K` (`num_blocks: 525`), and `256K` (`num_blocks: 1037`) without duplicate `model_128k.onnx` / `model_256k.onnx` files.
- **Pure GPU execution** up to **256K (`262,144`) context length** on H200 / high-VRAM GPUs (`gpu-128k`, `gpu-256k` -> `model.onnx`).
- **QMoE CPU/GPU hybrid offload** on 20–36.5 GB consumer GPUs (`consumer-gpu-24g`, `consumer-gpu-24g-host-64g`, `consumer-gpu-30g-ram-48g-32k`, `consumer-gpu-32g` -> `model.onnx` + `expert-profile.bin`).
- **Authoritative protocol boundary token IDs** (`bot_token_id: 248058`, `eot_token_id: 248059`, `bor_token_id: 248068`, `eor_token_id: 248069`) and **Engine dynamic batching** (`max_batch_size: 1`, `max_scheduled_tokens: 8192`).

Opt-in fixed-capacity exports instead select one target graph and disable
automatic context profiles and Engine dynamic batching (see section 5).

---

## 1. Building Predefined RC5 Packages from the Hugging Face Checkpoints

```bash
# Build all 4 core RC5 packages directly from /data/tlwu/Qwen3.8-Flash-Next and /data/tlwu/Qwen3.8-Flash-Next-DFlash2:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --models-root /home/tianlei/models \
  --link-root /data/tlwu \
  --presets rc5_all

# Or build a single predefined preset (e.g., 2b375_x02_y14_z32_dp23_b64):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --out /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64
```

| Preset Name | Alias | `(x INT4, y INT3, z INT2)` | KL Mode | Avg Bits | QMoE Size | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | Min VRAM (`gpu-128k` / `gpu-256k`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2` | `all_int4` | `(48, 0, 0)` | `rank` | `4.000` | `63.28 GiB` | `75.42 GiB` (`80.99 GB`) | `77.90 GiB` (`83.65 GB`) | `83 GB` / `87 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64` | `top16` | `(16, 32, 0)` | `rank` | `3.333` | `51.56 GiB` | `64.79 GiB` (`69.57 GB`) | `67.63 GiB` (`72.62 GB`) | `72 GB` / `76 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64` | `top08` | `(8, 40, 0)` | `rank` | `3.167` | `48.63 GiB` | `61.87 GiB` (`66.43 GB`) | `64.71 GiB` (`69.48 GB`) | `69 GB` / `73 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64` | `2b375` | `(2, 14, 32)` | `dp23` | `2.375` | `37.06 GiB` | `50.30 GiB` (`54.01 GB`) | `52.00 GiB` (`55.84 GB`) | `57 GB` / `61 GB` |

---

## 2. Building Custom `(x INT4 + y INT3 + z INT2)` QMoE Partitions Sorted by KL Similarity

```bash
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  -x 2 -y 14 -z 32 --kl-mode dp23 \
  --out /home/tianlei/models/qwen_flash_next_custom_2_14_32
```

---

## 3. Exporting Zero-Copy GGUF Model Packages (`build_package_gguf.py`)

`tools/python/qwen38_flash_next/build_package_gguf.py` exports `unsloth/Qwen3.8-Flash-Next-GGUF` quantization variants (`UD-IQ1_S`, `UD-IQ2_XS`, `UD-IQ3_XXS`) and the `IQ4_NL` n-gram (`engram`) GGUF directly to `qwen38_flash_graph` packages using `GatherPacked`, `MatMulPacked`, and `MoEPacked`, sharing dynamic cache surgery (`ensure_unified_dynamic_cache_model`), protocol boundary tokens, and config migration helpers with `build_package.py`:

```bash
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64 \
  --out-root /data/tlwu \
  --variants UD-IQ1_S,UD-IQ2_XS,UD-IQ3_XXS
```

| GGUF Variant | Total GPU Weights | 32K Peak VRAM (`num_blocks=142`, min `141`) | 64K Peak VRAM (`num_blocks=270`, min `269`) | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | `runtime_profiles` Min VRAM (`gpu-32k` / `gpu-64k` / `gpu-128k` / `gpu-256k`) |
|---|---:|---:|---:|---:|---:|---:|
| **`UD-IQ1_S`** | `35.66 GiB` (`38.29 GB`) | `41.39 GiB` (`44.45 GB`) | `41.74 GiB` (`44.82 GB`) | `42.85 GiB` (`46.00 GB`) | `45.54 GiB` (`48.90 GB`) | `47 GB` / `48 GB` / `49 GB` / `52 GB` |
| **`UD-IQ2_XS`** | `44.46 GiB` (`47.74 GB`) | `50.14 GiB` (`53.84 GB`) | `50.49 GiB` (`54.21 GB`) | `51.59 GiB` (`55.39 GB`) | `54.28 GiB` (`58.28 GB`) | `56 GB` / `57 GB` / `58 GB` / `61 GB` |
| **`UD-IQ3_XXS`** | `51.73 GiB` (`55.55 GB`) | `57.44 GiB` (`61.68 GB`) | `57.79 GiB` (`62.05 GB`) | `58.89 GiB` (`63.24 GB`) | `61.59 GiB` (`66.13 GB`) | `64 GB` / `65 GB` / `66 GB` / `69 GB` |

---

## 4. Component Precision Options (`precision_options.py`)

Both `build_package.py` and `build_package_gguf.py` accept `--precision-profile {default,fp32,fp32_fp16kv,fp32_fp16kv_v2,fp32_int8kv_v2}` as well as individual per-component flags (`--kv-cache-dtype`, `--rotary-cache-dtype`, `--qsa-qkv-dtype`, `--indexer-dtype`, `--gdn-gate-dtype`, `--gdn-precision`, `--ple-engram`, `--ple-norm-dtype`, `--small-weights-source`, `--emit-int8kv-variant`, `--consolidate-data-files`):

- **Dual-ONNX (`model.onnx` + `model_int8kv.onnx`) & Consolidated External Data (`model.onnx.data`)**:
  - With `--precision-profile fp32_fp16kv_v2` (or `--emit-int8kv-variant`), the builder emits **both** `model.onnx` (`FP16` KV cache) and `model_int8kv.onnx` (`INT8` KV cache with inline `node_GuardedDSA_*_{k,v}_scale` initializers) in the same package directory, sharing the exact same external weight files.
  - `genai_config.json` `runtime_profiles` automatically select `"filename": "model_int8kv.onnx"` for VRAM-constrained offload profiles (`consumer-gpu-24g`, `consumer-gpu-32g`, etc., where `session.moe.offload.mode == "required"`) and inherit `"model.onnx"` (`FP16` KV cache) for full-VRAM profiles (`gpu-128k`, `gpu-256k`).
  - With `--consolidate-data-files` (`consolidate_package_external_data`), all target decoder external `.data` files are merged into a single `4096`-byte page-aligned `model.onnx.data` file (compatible with `PosixEnv::MapFileIntoMemory` and `PromoteFileMappingToSharedReadOnly` for QMoE offload), leaving only 3 `.data` files in the package (`embedding.onnx.data`, `dflash2.onnx.data`, and `model.onnx.data`).

```bash
# Build rc5_top16_fp32_fp16kv_v2, rc5_all_int4_fp32_fp16kv_v2, and rc5_2b375_fp32_fp16kv_v2
# (emits both model.onnx [FP16 KV] and model_int8kv.onnx [INT8 KV] + consolidated model.onnx.data):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset top16_44_33_b64,qwen_flash_next_int4_int8_rc5_contrib_dflash2,2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32_fp16kv_v2 \
  --consolidate-data-files \
  --out /data/tlwu/qwen_flash_next_rc5_v2

# Build rc5_all_int4_fp32_fp16kv (FP16 KV cache + FP32 rotary cache + FP32 GDN/indexer/PLE norms + Q8_0 engram):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset qwen_flash_next_int4_int8_rc5_contrib_dflash2 \
  --precision-profile fp32_fp16kv \
  --out /data/tlwu/qwen_flash_next_rc5_all_int4_fp32_fp16kv

# Build rc5_2b375 with BF16 KV cache + FP32 GDN/indexer/PLE norms:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32 \
  --out /data/tlwu/qwen_flash_next_rc5_2b375

# Build GGUF UD-IQ3_XXS and UD-IQ2_XS with Q8_0 engram, FP32 long-context path, and FP16 KV cache:
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64_unified \
  --out-root /data/tlwu \
  --variants UD-IQ3_XXS,UD-IQ2_XS \
  --precision-profile fp32_fp16kv \
  --pkg-suffix _q8_engram_fp32_fp16kv
```

## 5. Configurable YaRN and context capacity (RC5 exporter)

The original command in section 1 remains the standard-RoPE export path. Without
`--rope-config`, `--max-context-length` or `--cache-layout fixed`, target/QSA tables
remain exactly 262144 rows, DFlash2 tables remain 262148 rows, and the existing
unified-cache profiles are retained. The decoder configuration now correctly
reports the graph's **24 query heads**, not 32; graph architecture is unchanged.

A fresh export requires the original BF16 **target and DFlash2 checkpoints**,
their configuration files, and tokenizer assets. Quantized ONNX banks are not
substitutes for missing BF16 checkpoints when quantization is required.

YaRN is an optional export-time dependency. Install the exact qualified upstream
reference in the export environment:

```bash
python -m pip install -r tools/python/qwen38_flash_next/requirements-yarn.txt
```

The adapter verifies the SHA-256 of Transformers' `modeling_rope_utils.py` at
revision `598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1` before calling
`ROPE_INIT_FUNCTIONS["yarn"]`. It fails rather than silently accepting another
implementation. Standard-RoPE exports do not require Transformers.

Export an independent 1M YaRN4 package:

```bash
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /checkpoints/Qwen3.8-Flash-Next \
  --dflash-dir /checkpoints/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --rope-config tools/python/qwen38_flash_next/yarn4.json \
  --max-context-length 1048576 --cache-layout fixed \
  --precision-profile default --prefill-chunk 512 --copy \
  --tactic-cache-dir /model-caches/qwen38-yarn4 \
  --out /models/qwen38-yarn4-1m
```

`--rope-config` accepts a JSON object with the fields shown in `yarn4.json`.
Supported types are `default` and `yarn`; scaling factors must be finite and at
least one. The specialized graph requires head dimension 256, partial rotary
factor 0.25 (effective dimension 64), and MRoPE sections `[11,11,10]`. These are
the text-position table layout, not a new multimodal position-input interface.
YaRN beta parameters and optional `attention_factor` are configurable. The
included factor-4 configuration yields correction bounds 14-22 and magnitude
`1.138629436111989`. Frequencies, phases, trigonometry and magnitude use FP32
before BF16 conversion. Magnitude is applied once in the tables, not again in
attention scaling. QSA tables duplicate the target's 32 columns to 64.

**Context capacity is independent of the YaRN factor.** With logical length `C`:

| Capacity | Formula | C = 1048576 |
|---|---|---:|
| KV and position history | C + 8 | 1048584 |
| QSA compressed state | floor((C + 8) / 4) + 1 | 262147 |
| Positional coverage | C + 16 | 1048592 |

Coverage includes verification and speculative lookahead. DFlash2 uses its own
standard RoPE (head 128, theta 10000), computed in FP64 then converted to FP32
and BF16. It receives the extended row count, **not target YaRN scaling**.

`--cache-layout fixed` constructs numeric cache dimensions directly and emits
only the selected target, `model_1m.onnx` for this example. The config explicitly
selects it. Automatic smaller-context profiles and Engine block-budget resizing
are disabled. Fixed qualification uses BF16 GDN arithmetic, disabled graph
optimization and conservative 512-row GEMM chunks. No generic builder, CUDA
operator or runtime change is required.

Unified caching remains the default. Explicit unified capacities must be
multiples of 256; the exporter sizes the single-slot block budget including the
existing DFlash2 reservation and disables automatic lower-capacity profiles.
New unified capacities require separate GPU qualification. Explicit context/RoPE
exports currently support **default component precision only**. Other precision
profiles/overrides and `--context-tiers` are rejected before export. The legacy
`--base-pkg` graph-conversion route is also rejected for this mode, including
incoming explicit-capacity packages when the new options are omitted.

For safe weight-bank reuse, replace the checkpoint arguments with:

```text
--reuse-banks-from /models/verified-original-rc5 --copy
```

The source must contain complete matching RC5 weight banks. Missing quantized
banks still require real BF16 checkpoints; they are never invented. Reuse
validates all six source positional tables, descriptors and recorded coverage.
Legacy banks without positional provenance must match standard-RoPE mathematics.
The builder then constructs new positional assets using `BinaryBlobWriter`,
regenerates descriptors/pool metadata, and constructs the graphs without
requantizing unchanged weights. Nonstandard legacy tables with no verifiable
provenance are rejected. `bank_metadata.json`, graph metadata and the package
manifest record the resolved positional configuration and reference revision.

Explicit-capacity exports disable tactic writes into model directories. The
default writable cache is a sibling `<output-name>-cache`; an explicit
`--tactic-cache-dir` must be separate from every model/checkpoint directory.
Use a new output directory: the existing builder replaces an existing output.
Do not pass a protected model as `--out`. Keep `--rope-config` outside every
output directory; overlapping resolved paths are rejected before any output
is replaced, including when building multiple presets.

### Tests and qualification

Run the fast, GPU-free suite after installing its ordinary test dependencies and
the optional pinned reference:

```bash
python -m pytest --confcutdir=test/python/builder \
  test/python/builder/test_qwen38_package_builders.py -q
```

To additionally compare actual graph contracts and **every row of all six
positional tables** with a preserved prototype:

```bash
QWEN38_REFERENCE_PACKAGE=/models/validated-yarn4-1m \
  python -m pytest --confcutdir=test/python/builder \
  test/python/builder/test_qwen38_package_builders.py -q
```

Also set `QWEN38_EXPORTED_PACKAGE=/models/new-yarn4-1m` to compare a complete
new export's three graphs, external extents, logical tables, pool bindings and
critical configuration with that prototype. This check reads positional tensors
and graph headers, not all quantized weights.

The inherited RC5 graphs have raw stock-ONNX-checker limitations: the target
logits output has no declared shape, and DFlash2 uses two default-domain ORT
`SimplifiedLayerNormalization` nodes. Checker-only qualification may add the
symbolic output shape and custom-domain aliases in a temporary copy. These
adjustments must not rewrite the exported/runtime graphs.

The fast suite does not need a full checkpoint export. Package qualification
must separately verify every external range, pool binding, initializer shape,
capacity/selection setting and logical table hash against the reference.
Compare logical tensors and normalized graph contracts, not physical file
offsets or repeated multi-hundred-GB weight-file comparisons.

For GPU qualification, use the pinned native ORT
`8c1f2936809a17054d793aeba309c6b0e5236a85` and GenAI
`f022ff6c897e7a13e6eac1d37dd384ff79d6665e` and verify their loaded paths.
Keep writable caches outside model directories. Progress through short/4K,
131K, 524K, 768K, 900K and **1048544 input + 32 generated** tokens with initial
512-token prefill chunks. Check finite logits, validation statuses, QSA bounds,
absolute positions, actual GPU table consumption and high-position rotations,
DFlash2 execution, phase-specific device/process memory, timings and acceptance.
Run lightweight retrieval diagnostics only after execution passes.

The exporter-generated **fixed-capacity 1M YaRN4 package** passed
**1048544 input + 32 generated tokens** on an **A100 80GB**, matching the
post-export-patched prototype's generated tokens and final logits. Qualification
used validated weight-bank reuse, default component precision, the pinned native
runtime above and 512-token prefill chunks. All six positional tables matched
the prototype bit-for-bit; GPU table readbacks, sampled rotary/QSA checks and
DFlash2 execution passed with zero observed operator validation errors.

The full-capacity campaign used target capture threshold **4**. The package's
default threshold **2** was exercised separately at 4K and in near-1M retrieval
diagnostics with facts at 1%, 50% and 90% depth. These results establish engineering
execution correctness for the qualified configuration, not comprehensive
long-context model quality or performance certification. They do not automatically
qualify other exports, precision modes or unified layouts. Fresh checkpoint
quantization remains unqualified because the original BF16 checkpoints were
unavailable. Broad long-context quality and exact DFlash2 speculative losslessness
remain unproven; the known width-dependent discrepancy is deferred.
The original 128K/256K VRAM estimator is not used as admission evidence for new
explicit-capacity packages.

