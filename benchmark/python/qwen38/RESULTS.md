# Qwen3.8 Flash NVFP4 Optimization Results

Measured on one H200, batch one, greedy decoding, with exactly **8,192
uncached input tokens and 512 generated tokens**. End-to-end (E2E)
throughput includes prefill and request completion, but excludes model
loading and warmup. Decode throughput counts the 511 tokens after the
first token.

## Overall improvement

| Metric | Original baseline | Final configuration | Improvement |
|---|---:|---:|---:|
| **End-to-end throughput** | 30.89 tokens/s | **100.48 tokens/s** | **3.25x / +225%** |
| Decode throughput | 56.76 tokens/s | **126.42 tokens/s** | **2.23x / +123%** |
| Time to first token | 7.57 s | **1.05 s** | **86% lower** |
| Total request time | 16.57 s | **5.10 s** | **69% lower** |

The final result pools ten measured requests across two five-request
invocations, each with one excluded warmup. The invocations achieved
100.17 and 100.80 E2E tokens/s, with zero MTP failures. All measured
samples were retained.

## Optimizations and measured gains

Each row uses the relevant comparison from that experiment, not necessarily
the preceding row. **Do not add the gains together:** comparisons sometimes
use different MTP settings, and greedy-output/speculative-acceptance
variability affects timing. GPU kernel and isolated operator times are not
request wall times.

| # | Optimization | Measured before -> after | Gain / interpretation |
|---|---|---|---|
| 1 | **Enable CUDA graphs; increase prefill budget to 8K; tune KV-cache allocation** | E2E **30.89 -> 38.92 TPS** | **+26.0%**; TTFT **7.57 -> 5.32 s** |
| 2 | **Vectorized NVFP4 weight dequantization + warp-shuffle sparse softmax reductions** | E2E **38.92 -> 41.14 TPS** | **+5.7%**; TTFT **5.32 -> 4.72 s**. Tested together; individual gains are not isolated. |
| 3 | **Paired-channel sparse value accumulation, with head-size-256 specialization** | E2E **41.14 -> 43.18 TPS** | **+5.0%**; TTFT **4.72 -> 4.16 s** |
| 4 | **Grouped, tiled sparse-attention prefill**: share selected K/V across four query heads | E2E **43.18 -> 50.11 TPS** | **+16.0%**; TTFT **4.16 -> 2.50 s** |
| 5 | **Split sparse-attention decode across candidate partitions** | E2E **50.34 -> 52.66 TPS** | **+4.6%**; decode **66.58 -> 70.65 TPS** |
| 6 | **Hierarchical sparse-indexer selection: 32-block tiles + warp sorting** | E2E **52.65 -> 54.09 TPS** | **+2.7%**; decode **70.66 -> 73.27 TPS** |
| 7 | **Padding-aware hierarchical indexer merges** | E2E **54.08 -> 54.58 TPS** | **+0.9%** |
| 8 | **Enable FP8 QMoE support and fix speculative EOS-floor handling**, making MTP usable | MTP off -> width seven: E2E **54.57 -> 57.93 TPS** | **+6.2% E2E**, **+8.8% decode**. Benefit of enabling MTP, not FP8 support alone. |
| 9 | **Reduce MTP draft width from seven to one** | E2E **57.93 -> 68.39 TPS**, using the stable repeat | **+18.1%**; avoids excessive draft work. First invocation was slower at **62.80 TPS**. |
| 10 | **Tune short-query attention splits for MTP verification** | Stable E2E **68.39 -> 69.06 TPS** | **+1.0%**; two-row attention **160.61 -> 114.19 us/launch**, about **29% lower** |
| 11 | **Bound hierarchical indexer scoring to live work**, rather than full state capacity | Stable E2E **69.06 -> 70.81 TPS** | **+2.5%**; two-row scoring **49.30 -> 6.39 us**, about **87% lower** |
| 12 | **Select MTP width two instead of one** | Pooled E2E **69.05 -> 72.34 TPS**, 11 requests per width | **+4.8%**; approximately **0.34 s saved/request** |
| 13 | **Increase sparse-prefill grouping from four to six heads** | Attention GPU time **1,465.91 -> 1,183.31 ms** | **19.3% lower attention time**; TTFT about **0.28 s lower**. Fresh E2E comparison: **70.69 -> 71.11 TPS**, only **+0.6%** because of decode variability. |
| 14 | **Bypass unused grouped-GEMM autotuning when inference uses FP4 GEMV** | E2E **74.27 -> 79.79 TPS** | **+7.4%** in the five-request comparison; eliminates the reproduced multi-second tail stall. **A latency fix, not faster GEMV arithmetic.** |
| 15 | **Increase sparse-prefill grouping from six to twelve heads** | Attention GPU time **1,183.12 -> 840.96 ms** | **28.9% lower attention time**; base-only E2E **56.94 -> 59.17 TPS**, **+3.9%**; TTFT **0.344 s lower** |
| 16 | **Temporally parallel prefill convolution**, with a separate alias-safe final-state update | Convolution GPU time **221.88 -> 12.26 ms** | **94.5% lower convolution time**; base-only E2E **+2.0%**; width-two paired E2E **83.64 -> 87.40 TPS**, **+4.5%**. Not all integrated variation is attributable to convolution. |
| 17 | **Enable existing dense GEMM autotuning for the base decoder** | Plain repeat: base E2E **60.66 -> 61.04 TPS**; width two **86.16 -> 87.42 TPS** | **+0.6% base E2E**, **+1.5% integrated E2E**; eligible small projections can select tinygemm2 |
| 18 | **Interleave four independent sparse-attention dot products per warp** | Attention GPU time **840.90 -> 754.66 ms** | **10.3% lower attention time**; pooled base E2E **+1.1%**, width-two E2E **+0.4%**; TTFT about **85-88 ms lower** |
| 19 | **Vectorized 16-byte K/V staging for long sparse prefill** | FP16 attention replay **approximately 63.2 -> 41.3 ms/layer** | **Approximately 35% lower operator time**; historical width-two E2E **90.20 -> 94.24 TPS**, approximately **+4.5%**, not a fresh paired attribution |
| 20 | **Opt-in tensor-core sparse prefill: QK + FP16 PV** | FP16 attention replay **approximately 41.3 -> 20.1 ms/layer**; E2E **94.24 -> 99.18 TPS** | **Approximately 51% lower operator time**, **+5.2% E2E**. Combined with vector staging, actual attention GPU time falls **754.66 -> 238.13 ms**. |
| 21 | **Enable GEMM autotuning for the MTP head too** | E2E **99.58 -> 100.01 TPS** | Observed **+0.4%**; **too small to establish a conclusive independent gain** |
| 22 | **Exact FP8-to-FP32 bit conversion instead of repeated `ldexpf`** | Matched ten-request pools: E2E **99.44 -> 100.48 TPS** | **+1.05% measured E2E**; conversion is bitwise validated. Timing/acceptance variation means the entire gain cannot be assigned to conversion alone. |

## Experiments not enabled in the final configuration

| Experiment | Result | Decision |
|---|---|---|
| Head-size-256 specialization alone | E2E **41.14 -> 40.96 TPS** | No independent gain; subsequently combined with paired-channel accumulation |
| Reusable draft buffers, asynchronous head execution and hidden-state views | E2E **68.39 -> 67.98 TPS** | Reverted |
| Selected-logits graph variants | Saved roughly **14-16 ms TTFT**, but no demonstrated E2E gain | Optional experiment; not enabled |
| Dense-prefix cross-query K/V reuse | Attention **1,465.91 -> 1,467.91 ms** | Removed |
| Smaller, 32-row prefill tiles | Attention **1,465.91 -> 1,479.34 ms** | Removed |
| Expand tinygemm2 eligibility for large QKV projections | E2E **61.169 -> 61.132 TPS**; cuBLAS still won | Reverted |
| Eight-lane subwarp dot products | FP16 **approximately 70.5 -> 106.6 ms/layer** | Rejected |
| Eight interleaved rows instead of four | FP16 **approximately 63.2 -> 64.1 ms/layer** | Rejected |
| Early tensor-PV designs without the final staging/layout improvements | Approximately **64-106 ms/layer**, versus final **20.1 ms** | Replaced |
| MTP widths three and four | **89.38 / 85.64 E2E TPS**, versus width-two pooled **90.20** at that stage | Kept width two |

## Validation and limitations

- Final individual requests range from **98.22 to 104.92 E2E tokens/s**.
  The **100.48 TPS** result is aggregate throughput, not a per-request
  guarantee, and the margin above 100 is small.
- Tensor attention and GEMM tuning remain opt-in. FP16 tensor PV rounds
  probabilities to FP16 while retaining FP32 softmax and normalization;
  BF16 uses tensor QK with scalar FP32 PV. Outputs can change, and broader
  model-quality validation is still needed.
- Final attention validation passed 192 standalone numerical-reference
  checks and sixteen aligned/misaligned full-8K CUDA graph replay cases.
  The default vector path matched bitwise; tensor outputs passed the
  unchanged numerical tolerances.
- Exact FP8 conversion passed all 256 GPU bit-pattern comparisons.
  Thirty-six FP8 MTP QMoE and 24 NVFP4 cases matched the previous provider
  bitwise.
- New C++ regression tests from the later optimization iterations remain
  uncompiled/unrun; standalone numerical tests, graph replays and full
  requests provide the executed validation.
- Original model weights, configuration and installed runtime packages
  remain unchanged. Optimized libraries are staged in isolated runtimes.

## Peak memory usage

Measured separately with the final isolated runtime, width-two MTP,
base/MTP GEMM tuning and both tensor-attention flags, using the same
8,192-input / 512-output workload. The run includes model loading, one
warmup and three measured requests. All four requests completed with
512 output tokens and zero MTP failures.

| Phase | Sampled GPU device memory peak | Sampled host process RSS peak |
|---|---:|---:|
| Model loading | 76.21 GiB | 1.72 GiB |
| Warmup prefill, including cold tuning/capture | 100.06 GiB | 1.87 GiB |
| Warmup decode | 100.09 GiB | 2.42 GiB |
| Warmed prefill | 100.01 GiB | 3.35 GiB |
| Warmed decode | 100.04 GiB | 3.61 GiB |
| **Overall observed peak** | **100.09 GiB** | **3.61 GiB** |

GPU memory is device-wide NVML usage on otherwise idle GPU 0, including
weights, caches, CUDA context, allocator pools, workspaces and graphs;
it is not just live tensor storage. The idle NVML baseline is 0.60 GiB,
so the increase above baseline is approximately 99.49 GiB. Of the
140.40 GiB reported device capacity, approximately **40.31 GiB remains
at the observed peak**. This does not establish a minimum supported GPU
capacity: cache allocation and tuning behavior can change on another
device or configuration.

Sampling targets 20 ms, with a maximum observed gap of 127 ms. Short-lived
GPU allocations may be missed; these are sampled peaks, not exact CUDA
allocator high-water marks. Phase boundaries are received via stdout.
Host peak RSS uses the larger of sampled `/proc` VmHWM (3.61 GiB) and the
child-process wait high-water mark (3.59 GiB); both counters are retained
in the report. Host RSS is not virtual address space or total system RAM.
GPU and host peaks need not coincide and should not be summed as a
simultaneous peak.

This diagnostic run is separate from the ten-request throughput result.
The original memory sampler and raw samples remain local session artifacts.
For a fresh memory measurement, sample NVML device usage and process RSS
through loading, warmup and measured requests separately. Do not include
memory-instrumented request timings in the throughput comparison.

## Static-graph backend next to the paged Engine

A second Engine backend serves the same checkpoint from captured CUDA graphs
with a fixed-capacity KV cache and a device-side MTP(1) loop (model type
`qwen38_flash_graph`, PRs #2725-#2729). The table uses the protocol of
this document: one H200, batch one, greedy decoding, exactly **8,192 uncached
input tokens and 512 generated tokens**, with the prompt text of
`benchmark_qwen_mtp.py` (the same sentence repeated). The static-graph row
pools ten requests from two invocations of one warmup plus five measured
requests each, like the paged result above. TTFT is the time from
`begin_turn` to the `Engine.run` call that delivers the first token (fast
prefill included); decode counts the 511 tokens after the first token; E2E is
512 tokens divided by TTFT plus decode time.

| Metric | Paged Engine (final configuration above) | Static-graph Engine backend |
|---|---:|---:|
| **End-to-end throughput** | 100.48 tokens/s | **185.69 tokens/s** (185.79 and 185.59 per invocation; requests 185.07-185.92) |
| Decode throughput | 126.42 tokens/s | **252.50 tokens/s** |
| Mean time to first token | 1.05 s | **0.734 s** |
| Mean request time | 5.10 s | **2.76 s** |
| Draft acceptance | not reported above | 1.000 (every cycle committed two tokens) |

The same backend on natural text (windows of a long text pool, the vLLM
benchmark protocol: exact prompt length, 256 generated tokens, EOS ignored,
one warmup and three measured requests, decode only):

| Prompt tokens | Decode tokens/s | ms per MTP cycle | Draft acceptance | TTFT, median |
|---:|---:|---:|---:|---:|
| 2,048 | 249.0 | 7.70 | 0.91-0.95 | 0.298 s |
| 8,192 | 239.8 | 7.88 | 0.79-0.92 | 0.750 s |
| 32,768 | 231.5 | 8.07 | 0.88 | 3.150 s |

The 32k decode and the 8.07 ms cycle come from the earlier, six-run
development measurement on the same stack (median of the runs reported there);
the 2k and 8k rows were re-measured for this table on the stacked branch
(`tlwu/20261013/ga-b5-qwen38-qualification-tools`), where ms per cycle matched
the development run (7.71 / 7.88). Tokens per second follows draft acceptance;
ms per cycle is the steadier metric. Peak GPU memory of the baseline (`v12b`)
static-graph backend is 113.40 GiB (`116,123 MiB` `nvidia-smi`, 0.5 s samples,
2k-32k decode; `90,063 MiB / 87.95 GiB` static load), compared with the
100.09 GiB sampled above; the two samplers differ.

### Quantized and memory-reduced static-graph packages (`v12c`, `v12d`, and RC1)

Subsequent package and kernel optimizations (`build_package_engine_v12b.py`,
`qwen38_graph.cpp`, `qwen38_graph_prefill.cpp`, and `qwen38-flash-ops`) reduce
both static and peak GPU memory while increasing decode throughput and
preserving MMLU-Pro 800 accuracy (`664/800 = 83.000%` on `v12b` at `260.0 TPS`
median):

- **Optimized `v12c` (`INT8` KV + `FP8` Attn/SharedExp + `NVFP4` MTP MoE + dense weight sharing)**:
  - `pkg_v12b_engine_int8kv_fp8w_fp4moe` (BF16 LM-Head): **`77,237 MiB (75.43 GiB)`** static load, **`103,207 MiB (100.79 GiB)`** peak GPU memory (**`-12.61 GiB`** vs `v12b`), decode **`6.959 / 7.126 / 7.314 ms/cycle`** (**`265.1 / 252.2 / 243.4 TPS`** on 2k / 8k / 32k natural text), TTFT `0.101 / 0.155 / 0.296 / 0.761 / 3.182 s` (`128 / 512 / 2k / 8k / 32k`), and MMLU-Pro 800 **`668/800 (83.500%)`** ($p=0.54$ vs `v12b`) @ **`286.99 TPS` median** (`+10.4%` vs `v12b`).
  - `pkg_v12b_engine_int8kv_fp8w_fp4moe_fp8lm` (+ per-column FP8 LM-Head): **`76,635 MiB (74.84 GiB)`** static load, **`102,573 MiB (100.17 GiB)`** peak GPU memory (**`-13.23 GiB`** vs `v12b`), decode **`6.660 / 6.841 / 7.047 ms/cycle`** (**`275.4 / 263.8 / 252.9 TPS`** on 2k / 8k / 32k natural text), TTFT `0.100 / 0.153 / 0.293 / 0.756 / 3.170 s`, and MMLU-Pro 800 **`672/800 (84.000%)`** ($p=0.22$ vs `v12b`) @ **`287.71 TPS` median** (**`289.81 TPS`** with `--fuse-dense-ops`, `+11.5%` vs `v12b`).
- **Engine `v12d` (`P1 + P2 + P3 + P4`)**:
  - `P1`: `--prefill-chunk 2048` / `4096` reduces the dense-prefill BFC arena high-water mark by **`-13.93 GiB`** / **`-8.01 GiB`**.
  - `P2`: `--hc-quant fp8` quantizes `hc_tiled.data` (`1,225.00 -> 612.50 MiB` FP8, **`-612.50 MiB`** VRAM) with templated FP8 `HcBlock` / `HcPostBlock` and `UntileHcBf16`.
  - `P3`: `--host-map-embed` maps `model.embed_tokens.weight` (`1,212.50 MiB`) zero-copy over PCIe via `host_pool` (`cudaHostRegisterMapped`), saving **`-1.184 GiB`** GPU VRAM with 0% accuracy loss.
  - `P4`: Full initializer deduplication across `target`, `head_r2`, `dense`, and `head_kv` reduces session-private initializers from `177.84 MiB` to **`4.21 MiB`** (`68.80 GiB` device weight pool).
- **RC1 Unified Single-Graph Package (`qwen_flash_next_nvfp4_fp8_rc1`)**:
  - **Flat root layout & unified prefill + decode ONNX graphs (`model.onnx` + `mtp.onnx` + `embedding.onnx`)**: Extracts `model.embed_tokens.weight` (`[248320, 2560]` BF16, `1,212.50 MiB`) into a standard CPU `Gather` graph `embedding.onnx` + `embedding.onnx.data` (`0 MiB` GPU VRAM, zero-copy `qwen38_plane_embed_rows` UVA gather during decode) and executes both chunked prefill (`gpu_graph_id = "-1"`) and captured CUDA-graph decode (`gpu_graph_id = 71..74, 81..84`) through the exact same `model.onnx` and `mtp.onnx` graphs via sequence-length kernel dispatch (`rows <= 4` decode vs `rows > 4` prefill) and `Slice_last2` before `lm_head` / `MtpMoe`.
  - **`118.03 GiB` total package footprint (`-243 GiB` saved vs `361 GiB` `pkg_v12d_engine`)**: Compacts `source_weights/model.onnx.experts.0..7.data` from `63.28 GiB` to `7.03 GiB`, packs 447 small BF16/FP32 initializers (`45.54 MiB`) into `norm_and_rope.data`, and removes redundant `.safetensors` and `dense.data` copies while preserving 100% byte-for-byte initializer parity (`27/27 PASS` on `engine_behaviours.py --prefill fast` and `--prefill steps`, `6/6 PASS` bit-exact `engine_gate.py`).
  - **H200 Full-HBM throughput & quality (`MMLU-Pro 800` & `LiveCodeBench v6`)**: Achieves **`6.535 / 6.706 / 6.885 ms/cycle`** (**`290.2 / 277.5 / 268.4 TPS`** on 2k / 8k / 32k natural text), MMLU-Pro 800 **`663/800 (82.875%)`** at **`306.02 tok/s` median** (`98.41%` draft acceptance), and **LiveCodeBench v6 (`175` problems, `xhigh` thinking mode, greedy decoding)**: **`95/175 = 54.29%` `pass@1`** at `16,384` max new tokens (`43/43 = 100%` Easy, `39/52 = 75.0%` Medium, `13/80 = 16.3%` Hard; `+2.29 pp` / $p=0.503$ vs `v12c` `91/175 = 52.00%`; `73/175 = 41.71%` at `8,192` and `85/175 = 48.57%` at `12,288`) at **`270.14 / 272.12 tok/s`** median/mean decode and **`0.199 s`** median TTFT.

| Package / Configuration | LM Head | Prefill Chunk | Static Load GPU Memory | Peak GPU Memory (2k-32k) | Saved vs `v12b` Peak | Decode ms/cycle (2k / 8k / 32k) | Natural-Text Decode TPS (2k / 8k / 32k) | TTFT Median (`128 / 512 / 2k / 8k / 32k`) | MMLU-Pro 800 @ 8192 | MMLU Decode TPS (median) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| **`v12b` baseline** (`pkg_v12b_engine`) | BF16 | 8,192 | 90,063 MiB (87.95 GiB) | 116,123 MiB (113.40 GiB) | — | 7.713 / 7.879 / 8.068 | 249.0 / 239.8 / 231.5 | 0.105 / 0.159 / 0.298 / 0.750 / 3.150 s | 664/800 (83.000%) | 260.0 |
| **`v12c` Optimized** (`pkg_v12b_engine_int8kv_fp8w_fp4moe`) | BF16 | 8,192 | 77,237 MiB (75.43 GiB) | 103,207 MiB (100.79 GiB) | -12.61 GiB | 6.959 / 7.126 / 7.314 | 265.1 / 252.2 / 243.4 | 0.101 / 0.155 / 0.296 / 0.761 / 3.182 s | 668/800 (83.500%) | 286.99 |
| **`v12c` + FP8 LM** (`pkg_v12b_engine_int8kv_fp8w_fp4moe_fp8lm`) | FP8 | 8,192 | 76,635 MiB (74.84 GiB) | 102,573 MiB (100.17 GiB) | -13.23 GiB | 6.660 / 6.841 / 7.047 | 275.4 / 263.8 / 252.9 | 0.100 / 0.153 / 0.293 / 0.756 / 3.170 s | 672/800 (84.000%) | 287.71 (289.81 fused) |
| **`v12d` BF16 LM** (`pkg_v12d_bf16lm_engine`) | BF16 | 2,048 | 75,071 MiB (73.31 GiB) | 87,111 MiB (85.07 GiB) | -28.33 GiB | 7.185 / 7.364 / 7.291 | 262.8 / 258.2 / 263.1 | 0.112 / 0.158 / 0.303 / 1.247 / 5.241 s | 664/800 (83.000%) | 279.51 (303.28 opt) |
| **`v12d` FP8 LM (`chunk=4096`)** (`pkg_v12d_c4096_fixed`) | FP8 | 4,096 | 74,469 MiB (72.72 GiB) | 92,199 MiB (90.04 GiB) | -23.36 GiB | 6.918 / 6.846 / 7.035 | 279.2 / 268.0 / 255.4 | 0.104 / 0.157 / 0.299 / 0.947 / 3.942 s | 663/800 (82.875%) | 290.26 |
| **`v12d` FP8 LM (`chunk=2048`)** (`pkg_v12d_engine`) | FP8 | 2,048 | 74,469 MiB (72.72 GiB) | 86,477 MiB (84.45 GiB) | -28.95 GiB | 6.918 / 6.845 / 7.019 | 279.1 / 271.9 / 250.6 | 0.112 / 0.158 / 0.303 / 1.252 / 5.253 s | 663/800 (82.875%) | 290.26 |
| **RC1 Unified Single-Graph (`chunk=2048`)** (`qwen_flash_next_nvfp4_fp8_rc1`) | FP8 | 2,048 | **74,469 MiB (72.72 GiB)** | **86,477 MiB (84.45 GiB)** | **-28.95 GiB (-29,646 MiB)** | **6.535 / 6.706 / 6.885** | **290.2 / 277.5 / 268.4** | **0.112 / 0.158 / 0.303 / 1.252 / 5.253 s** | **663/800 (82.875%, $p=1.00$)** | **306.02** |

### 24 GiB & 32 GiB GPU QMoE CPU Offload on H200 (Offload v3 on Engine `v12c`, `v12d`, & RC1)

Combining `qwen_flash_next_nvfp4_fp8_rc1` (`-1,392.38 MiB` non-expert device weights vs `v12c`
offload) with `com.microsoft::QMoEOffload` (Offload v3, multi-draft chained MTP
`S=2..4`, and CPU/GPU expert overlap) adds `+288` GPU-resident QMoE expert
slots (`+6` slots/layer) while remaining strictly inside 24 GiB (`<= 24,576 MiB`)
and 32 GiB (`<= 32,768 MiB`) VRAM budgets across 2k-32k prefill and decode:

- **24 GiB GPU budget (`qwen_flash_next_nvfp4_fp8_rc1_offload_24g`, `s3648`: `3,648 + 96` physical slots = `14.84%` resident)**: **`24,241 MiB (23.67 GiB)`** peak GPU memory (`<= 24,576 MiB`), adaptive decode **`156.81 tok/s` median / `175.45 tok/s` peak** at 2k context (`168.20 tok/s` peak at `S=3`), **`133.14–141.73 tok/s`** at 8k (`173.90 tok/s` peak), **`132.39–135.63 tok/s`** at 32k (`165.14 tok/s` peak at `S=3`), MMLU-Pro decode **`158.08 tok/s` median / `173.03 tok/s` peak**, TTFT `0.537 / 0.868 / 1.286 / 2.797 s` (`128 / 512 / 2k / 8k`), and **`116.34–134.10 tok/s` median (`136.83 tok/s` peak)** at 2k (`59.17 GiB` cgroup `memory.peak`) under a **`64 GiB` enforced host RAM** budget (`100%` disjoint cold-expert residency, `20,928 / 20,928` cached).
- **32 GiB GPU budget (`qwen_flash_next_nvfp4_fp8_rc1_offload_32g`, `s6768`: `6,768 + 96` physical slots = `27.54%` resident)**: **`32,403 MiB (31.64 GiB)`** peak GPU memory (`<= 32,768 MiB`), adaptive decode **`196.48 tok/s` median / `208.35 tok/s` peak** at 2k context (`15.89 ms/cycle`), **`167.57 tok/s` median / `191.04 tok/s` peak** at 8k, **`155.74 tok/s` median / `160.87 tok/s` peak** at 32k (`S=3`; `150.75 / 167.74 tok/s` at `S=4`), TTFT `0.450 / 0.736 / 1.129 / 2.564 s` (`128 / 512 / 2k / 8k`), and MMLU-Pro 800 **`661/800 (82.625%)`** at **`173.05 tok/s` median / `190.28 tok/s` peak** (`486.2 tok/s` median / `756.4 tok/s` peak across 8 GPUs).
- **Bounded host RAM cgroup evaluation (`64 GiB` and `48 GiB` enforced `MemoryMax` on 32 GiB GPU)**:
  - **`64 GiB` enforced RAM (`ORT_QMOE_HOST_CACHE_MIB=57344`, `17,808 / 17,808` cold experts cached = `100%` disjoint)**: **`51.14 GiB`** cgroup `memory.peak` (`0` OOM), **`1.14 s` TTFT / `169.66 tok/s` median (`182.09–189.03 tok/s` peak)** at 2k, **`3.30 s` TTFT / `145.21 tok/s` median (`160.84 tok/s` peak)** at 8k, and **`17.55 s` TTFT / `134.02 tok/s` median (`137.42 tok/s` peak)** at 32k.
  - **`48 GiB` enforced RAM (`ORT_QMOE_HOST_CACHE_MIB=40960`, `15,456 / 18,096` cold experts cached = `85.4%`)**: **`45.29 GiB`** cgroup `memory.peak` (`0` OOM), **`1.40 s` TTFT / `123.41 tok/s` median (`147.40 tok/s` peak)** at 2k and **`3.59 s` TTFT / `114.42 tok/s` median (`162.42 tok/s` peak)** at 8k when page-cache hot, and **`54.5–75.1 tok/s`** under cold-storage `DROP_PAGE_CACHE=1` (no cold-storage baseline is recorded here, so no ratio is claimed).

### Consumer GPU Evaluation: Windows RTX 5090 (32 GB VRAM, 64 GB DDR5 RAM, Ryzen 9 9950X)

We ported and evaluated `qwen_flash_next_nvfp4_fp8_rc1_offload_32g` (`model.type = "qwen38_flash_graph"`, 4-row folded MTP `model_s4.onnx` + 3-draft `mtp.onnx`) on a consumer Windows 11 Pro workstation (**NVIDIA GeForce RTX 5090 32 GB GDDR7 `sm_120`** under WDDM, **AMD Ryzen 9 9950X 16-Core / 32-Thread AVX-512**, **64 GB DDR5 RAM**, Visual Studio 2026 MSVC 19.50, CUDA 13.2). Decode runs entirely inside a single captured 48-layer CUDA Graph via the zero-copy device-doorbell protocol (`DoorbellService` + `QMoEOffloadMergeFinalizeCombineKernel`) with `ORT_QMOE_HOST_CACHE_GIB=0` (direct `MapViewOfFile` over `source_weights\experts.data` with `MEMORY_PRIORITY_VERY_LOW` initial upload so `0 MiB` of GPU-resident experts pollute the Windows CPU Working Set).

Session 1 (the `75%` configuration) and four further optimizations (Phase 1–4, below) raised RTX 5090 decode throughput over the initial port by **`2.28x`–`2.47x`** (**`+128.4%`–`+147.1%`**; the summary table gives each context and configuration) and improved time to first token (TTFT) by **`1.11x`–`3.35x`** depending on prompt length ([TTFT by prompt length](#ttft-by-prompt-length-on-windows-rtx-5090)). TTFT covers the whole first-token path rather than prefill alone, and most of its improvement was already present after Session 1. The four Phase 1–4 optimizations are:

1. **WDDM Silent VRAM Spillover Elimination & Spare-Slot Reclamation (`qmoe_offload.cc`)**: Reclaiming unused `spare_slots_` (`768 -> 0` when promotions are off, or capped to `2/layer = 96` slots when enabled, saving **`1.77–2.02 GiB`** VRAM), enabling `use_arena_dequant` (`-600 MiB`), and scaling resident hot experts to `60%–62%` (`4,880–5,046` slots = `12.9–13.3 GiB`) keeps total `cudaMalloc` safely below the `~30.9 GiB` WDDM physical VRAM threshold. This eliminates silent WDDM paging to Shared GPU Memory over PCIe and cuts intra-cycle layer-to-layer GPU stalls (`gap_in`) by **88.6%** per cycle (`77.02 ms/cycle` / `1,640.3 us/layer` -> **`8.76 ms/cycle` / `184.9 us/layer`**).
2. **Single-Barrier Grouped Cold-Expert Dispatch, CCD Core Pinning & AVX-512 Prefetch (`qmoe_spin_pool.h`, `qmoe_shared_layout_avx512.cc`)**: Setting `ORT_QMOE_SHARED_GROUPED_CALLS=64` (`FC1=128`, `FC2=256`) dispatches all `~18.8` active cold experts per layer in **1 thread-pool barrier per layer** (`48` barriers/cycle instead of `144`), while `DiscoverWindowsBalancedCpuOrder` pins `SpinPool` workers round-robin across both Ryzen 9 9950X CCDs (`GetLogicalProcessorInformationEx` + `SetThreadAffinityMask`) and `Fp4Group4Avx512` issues `_mm_prefetch` (`_MM_HINT_T0`) for the next 4-row weight block.
3. **Cross-Layer Async Cold-Expert Promotion (`DoorbellService::PromoLoop`) & Prefill DMA Overlap (`PrefetchNextQMoEPrefillDmaPass`)**: During decode, `PromoLoop()` inspects lookahead layers `idx + 2..4` and enqueues asynchronous H2D DMA promotions onto a pre-allocated low-priority `promo_stream_` while intermediate layers execute on the CPU and GPU. During prefill, active-expert filtering (`expert_Zero_counts_host_[e] == 0`, cutting PCIe volume by `45%–76%`), single-contiguous-buffer DMA packing, and `PrefetchNextQMoEPrefillDmaPass` overlap next-pass/next-layer host staging and PCIe DMA with GPU MoE/attention compute.
4. **4-Row Batched PLE Control Staging & 2,048-Slot Hot N-Gram PLE Cache (`qwen38_plane.cu`)**: Batching the 40-word control H2D copy across all 4 folded draft rows and adding a `16 MiB` (`2,048 x 3,840` BF16) direct-mapped N-gram PLE row cache in `StagedPleWorkspace` cuts inter-cycle MTP draft + control overhead (`t1 + t2 + t3`) by **50.8%** (`13.67 -> 6.72 ms/cycle`).

#### Cycle-Level Telemetry Breakdown on Windows RTX 5090 (`ORT_QMOE_SHARED_CPU_TIMING=1`, 4-Row Folded MTP Target Cycle)

| Metric (48 Layers, 4-Row Folded MTP Target Cycle) | `SCALE_PCT=75`, `GROUPED=8` (WDDM Spillover) | `SCALE_PCT=50`, `GROUPED=8` (Zero Spillover) | **`SCALE_PCT=60..62`, `GROUPED=64`, Phase 1–4** |
|---|---:|---:|---:|
| **Physical GPU Slots (`resident + spare`)** | `6,103 + 768 = 6,871` (`18.12 GiB`) | `4,067 + 768 = 4,835` (`12.75 GiB`) | **`4,880..5,046 + 0..96` (`12.89–13.55 GiB`)** |
| **Cold Routes / Distinct Cold Experts per Cycle** | `1,395.4` (`72.7%`) / `835.3` (`17.4`/layer) | `1,577.5` (`82.2%`) / `952.1` (`19.8`/layer) | **`1,512.0` (`78.8%`) / `904.2` (`18.8`/layer, `2.33 GiB/cycle`)** |
| **CPU Cold-Expert Compute (`cpu_us` / cycle)** | `70.19 ms` (`~62 ms` w/ fused FC1->FC2) | `73.92 ms` (`~65 ms` w/ fused FC1->FC2) | **`~54.8 ms`** (fused FC1->FC2, 1 barrier/layer, CCD-pinned) |
| **Intra-Cycle Layer-to-Layer GPU Gap (`gap_in` / cycle)** | `77.02 ms` (`1,640.3 us/layer` WDDM stall) | `8.76 ms` (`184.9 us/layer`, `-88.6%`) | **`~8.8 ms` (`~185 us/layer`, zero WDDM spillover)** |
| **Inter-Cycle MTP Draft Head + Control (`t1 + t2 + t3`)** | `13.67 ms` (`10.47 + 0.08 + 3.12 ms`) | `9.44 ms` (`9.10 + 0.09 + 0.25 ms`) | **`6.72 ms` (`-50.8%` w/ batched 4-row PLE + N-gram cache)** |
| **Measured Cycle Time (`ctx=512` / `ctx=2048`)** | `137.22 ms` / `137.69 ms` | `63.38 ms` / `81.22 ms` | **`63.38–64.23 ms` / `61.94–63.40 ms`** |
| **Measured Decode Throughput (`ctx=512` / `ctx=2048`)** | `27.22 tok/s` / `27.95 tok/s` | `40.90 tok/s` / `46.33 tok/s` | **`54.47–58.93 tok/s` / `58.90–62.13 tok/s`** |

#### TTFT by prompt length on Windows RTX 5090

TTFT is what `bench_ttft_engine.py` reports: the time from submitting the prompt (`begin_turn`) to the return of the `Engine.run` call that delivers the first token. That interval includes every prompt chunk, the dense-to-packed hand-off, the batched head-KV prefill, the first draft and the enqueue of the first verify cycle, so the factors below are TTFT improvements, not isolated prefill speedups.

| Prompt tokens | Initial Port (Baseline) | Session 1 Opt (`75%` Scale) | **Phase 1–4 Final** | **TTFT improvement vs. baseline** | Session 1 / final |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.65 s | 1.97 s | 2.38 s | **1.11x** | 0.83x |
| 256 | 4.21 s | 2.27 s | 2.90 s | **1.45x** | 0.78x |
| 512 | 8.07 s | 3.21 s | 3.06 s | **2.64x** | 1.05x |
| 2,048 | 32.02 s | 9.48 s | 9.57–10.78 s | **2.97x–3.35x** | 0.88x–0.99x |
| 8,192 | 140.47 s | 42.16 s | 42.94–43.98 s | **3.19x–3.27x** | 0.96x–0.98x |

Each factor is the baseline (or Session 1) time divided by the final time, rounded to two decimals from the times shown. Where the final column is a range of recorded values, the factor range runs from the slowest to the fastest final value. A Session 1 / final value below `1.00x` means the final configuration's TTFT is higher than Session 1's: most of the improvement over the baseline was already present after Session 1. `check_results_ratios.py` recomputes these factors from the rows above, so they are derived rather than typed. They use the numbers reported here and are not independent GPU measurements.

#### Verified Windows RTX 5090 Benchmark & MMLU-Pro Summary

| Benchmark / Suite | Initial Port (Baseline) | Session 1 Opt (`75%` Scale) | **Phase 1–4 Final (`62%` Default / Cross-Layer Promo)** | Total Improvement vs. Baseline |
|---|---:|---:|---:|---|
| **`engine_behaviours.py` Functional & Determinism** | `26 / 26 PASS` | `26 / 26 PASS` | **`26 / 26 PASS` (`100%` turn-to-turn bit-exact)** | **100% compliant** |
| **`bench_engine_mtp.py` (`ctx=512` median decode)** | `23.85 tok/s` (`167.85 ms/cycle`) | `27.22 tok/s` (`137.22 ms/cycle`) | **`54.47 tok/s` (default) / `58.15–58.93 tok/s` (promo, `63.38–64.23 ms/cycle`)** | **`+128.4%` (`2.28x`) default / `+143.8%`–`+147.1%` (`2.44x`–`2.47x`) promo** |
| **`bench_engine_mtp.py` (`ctx=2048` median decode)** | `25.16 tok/s` (`152.93 ms/cycle`) | `27.95 tok/s` (`137.69 ms/cycle`) | **`58.90 tok/s` (default) / `60.70–62.13 tok/s` (promo, `61.94–63.40 ms/cycle`)** | **`+134.1%` (`2.34x`) default / `+141.3%`–`+146.9%` (`2.41x`–`2.47x`) promo** |
| **`mmlu_engine_eval.py` (`80`-shard 0 accuracy)** | `8 / 10 (80.0%)` | `8 / 10 (80.0%)` | **`8 / 10 (80.0%)`, `0` truncated** | **Exact parity** |
| **`mmlu_engine_eval.py` (Long-CoT decode `tok/s`)** | `24.64–26.34 tok/s` (`25.67` avg) | `31.34–35.23 tok/s` (`33.78` avg) | **`51.51–60.42 tok/s` (`56.15 tok/s` avg, `63.56 tok/s` peak)** | **`+118.7%` (`2.19x`)** |
| **`mmlu_engine_eval.py` (Total 10-item shard wall time)** | `189.51 s` | `106.62 s` | **`72.62 s` (`2.61x` faster than baseline, `-31.9%` vs Session 1)** | **`2.61x` end-to-end speedup** |
| **`bench_ttft_engine.py` (TTFT by prompt length)** | per-length table above | per-length table above | **per-length table above** | **TTFT improvement `1.11x`–`3.35x`, depending on prompt length (table above)** |

Read the comparison with these caveats:

- **Not a controlled A/B.** The paged result uses the optimized ORT branches
  and kernels documented here; the static-graph backend uses its own exported
  graphs, a different CUDA execution-provider build, and separately built
  custom-op libraries (https://github.com/tianleiwu/qwen38-flash-ops), on a
  different CUDA toolkit (13.0 versus 13.3.73). The paged run measures width-two
  MTP; the static-graph loop proposes one draft token per cycle.
- **The repeated-sentence prompt favors speculation.** Acceptance was 1.000 on
  it, so 252.5 tokens/s is close to the two-tokens-per-cycle ceiling of the
  7.9 ms cycle. Natural text accepts 0.79-0.95 and decodes at 231.5-249.0 tokens/s.
  The paged table does not control for acceptance either.
- **Scope of the static-graph backend:** greedy decoding only, one resident
  request (others queue), Linux and Windows with CUDA, no prefix cache, sessions
  up to 33,024 tokens. A continued or rewound turn re-ingests its whole history
  (0.3 s at 2k, 3.2 s at 32k on H200).
- **The fast prefill is tolerance level**, not bit-reproducible: streams of
  the same prompt can diverge after tens of tokens. MMLU-Pro (800 items, 8,192
  new tokens) scored 664/800 on this backend through `og.Engine`, within the
  noise of the other MTP(1) systems measured in the development report; this is
  not a model-quality claim for the paged configuration.
- **Reproduction** needs the custom-op libraries, the exported graphs and a
  CUDA EP that are not in this repository. `benchmark_qwen_mtp.py` requests a
  minimum generated-token count to force 512 tokens, which the static-graph
  backend rejects, so this directory's scripts do not run it. Use
  `tools/qwen38_graph/bench_engine_mtp.py` and
  `bench_ttft_engine.py` from #2729, for example
  `bench_engine_mtp.py <fixed-length package> --out r.json --contexts 8192
  --new 512 --reps 5 --text-pool <repeated-prompt-text file>`.

## Detailed records and reproduction

- [Optimization log and experiment details](../../../docs/Qwen38MtpEngramIssues.md)
- [Portable reproduction instructions](README.md)
- [Benchmark launcher](run_benchmark.sh)

Original raw reports, profiles and memory samples remain local session
artifacts and are not published with this branch. Use the checked-in
benchmark and overlay to generate fresh reports.
