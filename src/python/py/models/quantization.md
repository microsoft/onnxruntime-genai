# INT4 Weight Quantization Methods

This document describes the INT4 weight-only quantization methods exposed by the
ONNX Runtime GenAI model builder via the `int4_algo_config` extra option, and the
**orthogonal int8 bit-placement flags** that can be combined with any base method.

All methods quantize the constant weight `B` of `MatMul` nodes block-wise (default
block size 32, set with `int4_block_size`) into a `MatMulNBits` contrib op. They are
all *round-to-nearest* (RTN) — none of them use calibration data or error feedback
(unlike GPTQ/AWQ/HQQ).

## Design: method vs. bit placement

A quantization configuration has two independent parts:

1. **Base method** — *how* each weight block is rounded (the scale/rounding math).
2. **Bit placement** — *which* MatMuls are upgraded from int4 to int8.

These used to be entangled in compound names (e.g. `rtn_last` = `rtn` + int8 LM head,
`k_quant_mixed` = `k_quant` + int8 on sensitive layers). They are now decoupled:

| Concept | Option | Values |
| --- | --- | --- |
| Base method | `int4_algo_config` | `default`, `rtn`, `k_quant` |
| Int8 last MatMul (LM head) | `last_matmul_weight_int8` | `true` / `false` |
| Int8 sensitive layers (llama.cpp mixed) | `int8_mixed_layers` | `true` / `false` |
| Int8 linear-attention layers | `int8_linear_attn` | `true` / `false` |

The bit-placement flags can be combined with **any** base method. For example
`int4_algo_config=rtn` + `last_matmul_weight_int8=true` is equivalent to the legacy `rtn_last`.

## Base methods

### `default`
`algo_config` passed to `MatMulNBitsQuantizer` is `None`, so it uses ONNX Runtime's
built-in `DefaultWeightOnlyQuantizer` (C++ MLAS kernel `quantize_matmul_4bits`).

For **symmetric** int4 (the GenAI default, `int4_is_symmetric=true`) the per-block
scale is:

```
scale = max(|w|) / 8            # halfRange = 2^(bits-1) = 8 for int4 [-8, 7]
```

This uses the full negative range of int4 (`-8`), giving the finest step. Node tensors
are named `*.weight_Q4` / `*.weight_scales`.

> Note: the `DefaultWeightOnlyQuantizer` applies a **single global** bit-width per
> pass and ignores per-node `customized_weight_config`. To support int8 bit placement
> on top of `default`, the builder quantizes the int4 body first (excluding the int8
> nodes) and then upgrades just those nodes in a second pass (see `to_int4`). The int4
> body remains byte-identical to a plain `default` model.
>
> ⚠️ The second (int8) pass uses the **RTN** quantizer, *not* a second DEFAULT pass.
> The DEFAULT (MLAS) symmetric 8-bit format emits **signed int8 with negative scales**,
> which the `MatMulNBits` runtime kernel does **not** consume correctly (it expects
> unsigned uint8 offset-by-128 with positive scales, as produced by RTN/k-quant).
> A DEFAULT-int8 LM head measured MMLU **0.6722** vs **~0.798** for the RTN-int8 LM head
> on GPT-OSS-20B — i.e. the DEFAULT 8-bit QOperator output is effectively broken at
> runtime, so RTN is used for the int8 nodes.

### `rtn`
Uses `RTNWeightOnlyQuantConfig` → Intel Neural Compressor's `rtn_quantize`. Also plain
round-to-nearest. Node tensors are named `*.weight_Q4G{block}` / `*.weight_scale`.

For **symmetric** int4 the scale is:

```
scale = 2 * max(|w|) / (maxq - minq) = 2 * max(|w|) / 15 = max(|w|) / 7.5
```

RTN stores uint4 values with zero point 8, so its dequantized integer levels are
still `q - 8 = [-8, 7]`. The difference from `default` is the scale spacing: RTN
fits the dynamic range over 15 steps (`±7.5`) instead of using the negative half
range 8.

> ⚠️ **Known accuracy gap vs. `default`.** The resulting step is `8 / 7.5 ≈ 6.7 %`
> coarser than `default`, which measurably lowers accuracy. On GPT-OSS-20B (MMLU,
> 14042 samples) `default` scored **0.7980** and `rtn` scored **0.7762** with all
> else held equal. A more accurate symmetric formula would be
> `scale = max(|w|) / 2^(bits-1)` (i.e. `/8`), matching `default`.

### `k_quant`
Uses `KQuantWeightOnlyQuantConfig` → Neural Compressor's k-quant (a more elaborate
per-block scaling derived from llama.cpp's K-quants). Honors per-node
`customized_weight_config`, so int8 bit placement works in a single pass.

## QMoE expert-weight quantization

MoE expert weights that are exported as `com.microsoft::QMoE` are quantized by a
separate QMoE path instead of the `MatMulNBits` quantizer above. When
`qmoe_block_size <= 0`, the builder uses per-channel quantization: each expert
matrix has logical shape `[N, K]`, one scale per output channel (`[N]`), and raw
storage shape `[N, K / pack]` before any CUDA prepacking.

For CUDA QMoE `quant_type="int"`, the runtime consumes unsigned int4/int8 storage
with an implicit zero-point offset. The numeric quantization is still symmetric,
but the stored bytes/nibbles are `q + zero_point`. The default export uses the
full integer range while keeping unsigned-offset storage:

| Bits | Numeric range | Scale | Stored value |
| --- | --- | --- | --- |
| 4 | `[-8, 7]` | `max(abs(w)) / 8` | `q + 8` |
| 8 | `[-128, 127]` | `max(abs(w)) / 128` | `q + 128` |

The default QMoE export uses the unsigned-offset contract above because ORT CUDA
QMoE prepack decodes raw int4 as `nibble - 8` and raw int8 as `byte - 128` before
laying weights out for the CUTLASS MoE GEMM.

For testing, set environment variable `GENAI_QMOE_UNSIGNED_FULL_RANGE=0` to use the
previous narrower unsigned-offset range: int4 `[-7, 7]` with scale
`max(abs(w)) / 7`, and int8 `[-127, 127]` with scale `max(abs(w)) / 127`. This is
not exposed as a model-builder `extra_options` flag.

For now, the QMoE quantizer helper is carried in onnxruntime-genai and mirrored by
ORT-side QMoE tests. Keep the two copies in sync. Longer term, this helper should
move into ONNX Runtime quantization tooling so genai can import the shared
implementation directly.

## Int8 bit-placement flags

### `last_matmul_weight_int8` (legacy `_last`)
Upgrades the last MatMul (e.g. `/lm_head/MatMul`) to 8 bits. The LM head is the single largest weight and is
output-sensitive, so int8 here costs ~0.27 GiB but is cheap relative to the body.
Combinable with any base method.

### `int8_mixed_layers` (legacy `k_quant_mixed`)
Promotes the most quantization-sensitive MatMuls to int8, following llama.cpp's mixed
strategy: for the first and last eighth of layers, plus every third layer, the
`attn/qkv_proj`, `attn/v_proj`, and `mlp/down_proj` MatMuls become int8.

### `int8_linear_attn` (legacy `k_quant_linear`)
For hybrid attention models (e.g. Qwen3.5), promotes the linear-attention projections
(`in_proj_*`, `out_proj`) and their MLP (`gate_proj`, `up_proj`, `down_proj`) to int8.
Linear-attention recurrence accumulates quantization error across the sequence (no
softmax normalization), so int8 helps there.

## Legacy aliases (still supported)

These compound `int4_algo_config` values are kept as backward-compatible aliases and
expand to a base method plus flags, producing identical models:

| Legacy value | Equivalent |
| --- | --- |
| `rtn_last` | `int4_algo_config=rtn` + `last_matmul_weight_int8=true` |
| `k_quant_last` | `int4_algo_config=k_quant` + `last_matmul_weight_int8=true` |
| `k_quant_mixed` | `int4_algo_config=k_quant` + `last_matmul_weight_int8=true` + `int8_mixed_layers=true` |
| `k_quant_linear` | `int4_algo_config=k_quant` + `last_matmul_weight_int8=true` + `int8_linear_attn=true` |

Only the legacy aliases (`k_quant_last`, `k_quant_mixed`, `k_quant_linear`) and/or an explicit `last_matmul_weight_int8=true` flag upgrade the LM head; the bare `k_quant` base method does not imply any int8 placement.

## Examples

```bash
# default body (best int4 accuracy) + int8 LM head
python -m onnxruntime_genai.models.builder -m <model> -o <out> -p int4 -e cuda \
  --extra_options int4_algo_config=default last_matmul_weight_int8=true int4_block_size=32

# rtn body + int8 LM head (== legacy rtn_last)
python -m onnxruntime_genai.models.builder -m <model> -o <out> -p int4 -e cuda \
  --extra_options int4_algo_config=rtn last_matmul_weight_int8=true
```
