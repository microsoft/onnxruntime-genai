# INT4 Weight Quantization Methods

This document describes the INT4 weight-only quantization methods exposed by the
ONNX Runtime GenAI model builder via the `int4_algo_config` extra option, and the
**orthogonal `matmul_mixed_precision`** that can be combined with any base method.

All methods quantize the constant weight `B` of `MatMul` nodes block-wise (default
block size 32, set with `int4_block_size`) into a `MatMulNBits` contrib op. They are
all *round-to-nearest* (RTN) — none of them use calibration data or error feedback
(unlike GPTQ/AWQ/HQQ).

## Design: method vs. mixed precision

A quantization configuration has two independent parts:

1. **Base method** — *how* each weight block is rounded (the scale/rounding math).
2. **Mixed precision** — *which* MatMuls use a different quant type than the int4 body.

These used to be entangled in compound names (e.g. `rtn_last` = `rtn` + int8 LM head,
`k_quant_mixed` = `k_quant` + int8 on sensitive layers). They are now decoupled:

| Concept | Option | Values |
| --- | --- | --- |
| Base method | `int4_algo_config` | `default`, `rtn`, `k_quant` |
| Mixed precision | `matmul_mixed_precision` | comma-separated `selector:quant_type` pairs |

`matmul_mixed_precision` maps a node-group **selector** to a **quant type**:

| Selector | Nodes upgraded |
| --- | --- |
| `last_matmul` | The last MatMul (LM head), the single largest, output-sensitive weight |
| `mixed_layers` | The most quantization-sensitive MatMuls (llama.cpp mixed strategy) |
| `linear_attn` | Linear-attention projections and their MLPs (hybrid attention models) |

Supported quant types are `int4` and `int8`. Using a quant-type name (rather than a bare
bit count) lets new schemes such as `fp8`/`fp4` be added without introducing a new option.

`matmul_mixed_precision` can be combined with **any** base method. For example
`int4_algo_config=rtn` + `matmul_mixed_precision=last_matmul:int8` is equivalent to the legacy `rtn_last`.

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
> pass and its config (`DefaultWeightOnlyQuantConfig`) does not accept a per-node
> `customized_weight_config` (unlike RTN/k-quant, which honor per-node bits in one
> pass). To support mixed precision on top of `default`, the builder quantizes the
> int4 body first (excluding the upgraded nodes) and then upgrades just those nodes,
> grouped by their target bit-width, in one **DEFAULT** pass per distinct width (see
> `to_nbits`). The int4 body remains byte-identical to a plain `default` model, and the
> upgraded nodes use the same MLAS quantizer as the body.
>
> The DEFAULT (MLAS) symmetric 8-bit format stores unsigned uint8 offset-by-128 with
> **signed** per-block scales. The `MatMulNBits` CUDA kernels consume negative scales
> **correctly** in every path — default GEMV / batched / dequant+GEMM and the fpA_intB
> int8 GEMM/GEMV — verified on both sm_80 (A100) and sm_90 (H200), and end-to-end on an
> offline-prepacked GPT-OSS-20B model. DEFAULT int8 is also more accurate than RTN int8
> (finer `/128` vs `/127.5` step, and it exploits the asymmetric `[-128,127]` range via
> the scale sign): on GPT-OSS-20B (int4 body + int8 lm_head/mixed-qkv, MMLU-800) DEFAULT
> int8 scored **0.8662** vs **0.8512** for RTN int8, so the int8 pass uses DEFAULT.
>
> (A stale earlier note claimed the DEFAULT-int8 LM head measured MMLU 0.6722 and was
> "broken at runtime"; that was not reproducible — negative scales are handled correctly
> on both architectures.)

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
`customized_weight_config`, so mixed-precision placement works in a single pass.

## CUDA MatMulNBits weight prepacking

All int4/int8 `MatMulNBits` weights are produced by the graph quantizer passes above
(`to_nbits`), on every EP. For the CUDA EP the builder can additionally prepack those
weights into the fpA_intB mixed-GEMM layout the CUDA kernel consumes directly.

Prepacking is a **pure layout transform**: it only rearranges the already-quantized
weight bytes and does not change the numeric values, so it is independent of the base
method (`default`/`rtn`/`k_quant`) and the bit width (int4/int8). It is applied as an
orthogonal post-pass over the quantized `MatMulNBits` nodes (see
`prepack_matmulnbits_weights`), keeping the standard initializer shape
`[N, K / block_size, block_size * bits / 8]` and setting the node attribute
`weight_prepacked = 1` (SM80/Ampere) or `2` (SM90/Hopper).

Enable it with `--extra_options matmulnbits_weights_prepacked=1` (SM80) or `=2` (SM90).
A node is prepacked only when the fpA_intB kernel supports it: symmetric weights, bits
in {4, 8}, `block_size` supported by the target layout (SM80 → {32, 64, 128}, SM90 →
{64, 128}), `K % block_size == 0`, and `N` aligned to the kernel tile (`N % 32` for
int8, `N % 64` for int4). Ineligible nodes (e.g. an `N = 32` MoE router) keep the raw
blockwise layout. An offline-prepacked model must be run with `ORT_FPA_INTB_GEMM`
enabling the relevant nbits (use `ORT_FPA_INTB_GEMM=1` for int4 and int8).

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

For now, the QMoE quantizer helper is carried in onnxruntime-genai and mirrored by
ORT-side QMoE tests. Keep the two copies in sync. Longer term, this helper should
move into ONNX Runtime quantization tooling so genai can import the shared
implementation directly.

## Mixed precision (`matmul_mixed_precision`)

`matmul_mixed_precision` is a comma-separated list of `selector:quant_type` pairs, e.g.
`matmul_mixed_precision=last_matmul:int8,mixed_layers:int8,linear_attn:int4`.

### `last_matmul` (legacy `_last`)
Upgrades the last MatMul (e.g. `/lm_head/MatMul`). The LM head is the single largest weight and is
output-sensitive, so `int8` here costs ~0.27 GiB but is cheap relative to the body.
Combinable with any base method.

### `mixed_layers` (legacy `k_quant_mixed`)
Promotes the most quantization-sensitive MatMuls, following llama.cpp's mixed
strategy: for the first and last eighth of layers, plus every third layer, the
`attn/qkv_proj`, `attn/v_proj`, and `mlp/down_proj` MatMuls are upgraded.

### `linear_attn` (legacy `k_quant_linear`)
For hybrid attention models (e.g. Qwen3.5), promotes the linear-attention projections
(`in_proj_*`, `out_proj`) and their MLP (`gate_proj`, `up_proj`, `down_proj`).
Linear-attention recurrence accumulates quantization error across the sequence (no
softmax normalization), so a higher-precision quant type helps there.

## Legacy aliases (still supported)

These compound `int4_algo_config` values are kept as backward-compatible aliases and
expand to a base method plus flags, producing identical models:

| Legacy value | Equivalent |
| --- | --- |
| `rtn_last` | `int4_algo_config=rtn` + `matmul_mixed_precision=last_matmul:int8` |
| `k_quant_last` | `int4_algo_config=k_quant` + `matmul_mixed_precision=last_matmul:int8` |
| `k_quant_mixed` | `int4_algo_config=k_quant` + `matmul_mixed_precision=last_matmul:int8,mixed_layers:int8` |
| `k_quant_linear` | `int4_algo_config=k_quant` + `matmul_mixed_precision=last_matmul:int8,linear_attn:int8` |

Only the legacy aliases (`k_quant_last`, `k_quant_mixed`, `k_quant_linear`) and/or an explicit `matmul_mixed_precision=last_matmul:int8` upgrade the LM head; the bare `k_quant` base method does not imply any mixed precision.

## Examples

```bash
# default body (best int4 accuracy) + int8 LM head
python -m onnxruntime_genai.models.builder -m <model> -o <out> -p int4 -e cuda \
  --extra_options int4_algo_config=default matmul_mixed_precision=last_matmul:int8 int4_block_size=32

# rtn body + int8 LM head (== legacy rtn_last)
python -m onnxruntime_genai.models.builder -m <model> -o <out> -p int4 -e cuda \
  --extra_options int4_algo_config=rtn matmul_mixed_precision=last_matmul:int8
```
