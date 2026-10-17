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
`qwen38_flash_graph`, draft PRs #2725-#2729). The table uses the protocol of
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

### Quantized and memory-reduced static-graph packages (`v12c` and `v12d`)

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

| Package / Configuration | LM Head | Prefill Chunk | Static Load GPU Memory | Peak GPU Memory (2k-32k) | Saved vs `v12b` Peak | Decode ms/cycle (2k / 8k / 32k) | Natural-Text Decode TPS (2k / 8k / 32k) | TTFT Median (`128 / 512 / 2k / 8k / 32k`) | MMLU-Pro 800 @ 8192 | MMLU Decode TPS (median) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| **`v12b` baseline** (`pkg_v12b_engine`) | BF16 | 8,192 | 90,063 MiB (87.95 GiB) | 116,123 MiB (113.40 GiB) | — | 7.713 / 7.879 / 8.068 | 249.0 / 239.8 / 231.5 | 0.105 / 0.159 / 0.298 / 0.750 / 3.150 s | 664/800 (83.000%) | 260.0 |
| **`v12c` Optimized** (`pkg_v12b_engine_int8kv_fp8w_fp4moe`) | BF16 | 8,192 | 77,237 MiB (75.43 GiB) | 103,207 MiB (100.79 GiB) | -12.61 GiB | 6.959 / 7.126 / 7.314 | 265.1 / 252.2 / 243.4 | 0.101 / 0.155 / 0.296 / 0.761 / 3.182 s | 668/800 (83.500%) | 286.99 |
| **`v12c` + FP8 LM** (`pkg_v12b_engine_int8kv_fp8w_fp4moe_fp8lm`) | FP8 | 8,192 | 76,635 MiB (74.84 GiB) | 102,573 MiB (100.17 GiB) | -13.23 GiB | 6.660 / 6.841 / 7.047 | 275.4 / 263.8 / 252.9 | 0.100 / 0.153 / 0.293 / 0.756 / 3.170 s | 672/800 (84.000%) | 287.71 (289.81 fused) |
| **`v12d` BF16 LM** (`pkg_v12d_bf16lm_engine`) | BF16 | 2,048 | 75,071 MiB (73.31 GiB) | 87,111 MiB (85.07 GiB) | -28.33 GiB | 7.185 / 7.364 / 7.291 | 262.8 / 258.2 / 263.1 | 0.112 / 0.158 / 0.303 / 1.247 / 5.241 s | 664/800 (83.000%) | 279.51 |
| **`v12d` FP8 LM (`chunk=4096`)** (`pkg_v12d_c4096_fixed`) | FP8 | 4,096 | 74,469 MiB (72.72 GiB) | 92,199 MiB (90.04 GiB) | -23.36 GiB | 6.918 / 6.846 / 7.035 | 279.2 / 268.0 / 255.4 | 0.104 / 0.157 / 0.299 / 0.947 / 3.942 s | 663/800 (82.875%) | 290.26 |
| **`v12d` FP8 LM (`chunk=2048`, recommended)** (`pkg_v12d_engine`) | FP8 | 2,048 | **74,469 MiB (72.72 GiB)** | **86,477 MiB (84.45 GiB)** | **-28.95 GiB (-29,646 MiB)** | **6.918 / 6.845 / 7.019** | **279.1 / 271.9 / 250.6** | **0.112 / 0.158 / 0.303 / 1.252 / 5.253 s** | **663/800 (82.875%, $p=1.00$)** | **290.26** |

### 24 GiB & 32 GiB GPU QMoE CPU Offload (Offload v3 on Engine `v12c` & `v12d`)

Combining `pkg_v12d_engine` (`-1,392.38 MiB` non-expert device weights vs `v12c`
offload) with `com.microsoft::QMoEOffload` (Offload v3, multi-draft chained MTP
`S=2..4`, and CPU/GPU expert overlap) adds `+288` GPU-resident QMoE expert
slots (`+6` slots/layer) while remaining strictly inside 24 GiB (`<= 24,576 MiB`)
and 32 GiB (`<= 32,768 MiB`) VRAM budgets across 2k-32k prefill and decode:

- **24 GiB GPU budget (`v12d` `s3648`: `3,648 + 96` physical slots = `14.84%` resident)**: **`24,241 MiB (23.67 GiB)`** peak GPU memory (`<= 24,576 MiB`), adaptive decode **`156.81 tok/s` median / `175.45 tok/s` peak** at 2k context, **`141.73 tok/s`** at 8k (`173.90 tok/s` peak), **`122.40 tok/s`** at 32k (`135.63 tok/s` at `S=3`), TTFT `0.537 / 0.868 / 1.286 / 2.797 s` (`128 / 512 / 2k / 8k`), and `6/6` bit-exact `engine_gate.py` / `27/27` `engine_behaviours.py`.
- **32 GiB GPU budget (`v12d` `s6768`: `6,768 + 96` physical slots = `27.54%` resident)**: **`32,403 MiB (31.64 GiB)`** peak GPU memory (`<= 32,768 MiB`), adaptive decode **`196.48 tok/s` median / `208.35 tok/s` peak** at 2k context (`15.89 ms/cycle`), **`167.57 tok/s` median / `191.04 tok/s` peak** at 8k, **`150.75 tok/s` median / `167.74 tok/s` peak** at 32k (`155.74 tok/s` median at `S=3`), TTFT `0.450 / 0.736 / 1.129 / 2.564 s` (`128 / 512 / 2k / 8k`), MMLU-Pro 800 **`661/800 (82.63%)`** (`663/800 = 82.88%` on `v12c` 32 GiB), and **`189.03 tok/s`** (`169.66 tok/s` median / `182.09 tok/s` peak on 2k text, `51.14 GiB` peak RSS) under a **`64 GiB` enforced host RAM** budget.

Read the comparison with these caveats:

- **Not a controlled A/B.** The paged result uses the optimized ORT branches
  and kernels documented here; the static-graph backend uses its own exported
  graphs, a different CUDA execution-provider build, and separately built
  custom-op libraries (https://github.com/tianleiwu/qwen38-flash-ops), on a
  different CUDA toolkit (13.0 versus 13.3.73). The paged run measures width-two
  MTP; the static-graph loop proposes one draft token per cycle.
- **The repeated-sentence prompt favors speculation.** Acceptance was 1.000 on
  it, so 252.5 tokens/s is close to the two-tokens-per-cycle ceiling of the
  7.9 ms cycle. Natural text accepts 0.79-0.95 and decodes at 232-249 tokens/s.
  The paged table does not control for acceptance either.
- **Scope of the static-graph backend:** greedy decoding only, one resident
  request (others queue), Linux and CUDA only, no prefix cache, sessions up to
  33,024 tokens. A continued or rewound turn re-ingests its whole history
  (0.3 s at 2k, 3.2 s at 32k).
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
