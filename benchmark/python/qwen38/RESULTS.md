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
| 23 | **Increase small-N split-K parallelism**: maximum 32 -> 64 splits, vector CTA target 128 -> 256 | Fresh matched ten-request pools: E2E **99.69 -> 101.37 TPS**, decode **125.19 -> 127.85 TPS** | **+1.68% observed E2E**, **+2.12% decode**; three-row GEMV **8.71 -> 7.72 us/launch**. Separate ORT commit [`0a03b230a7`](https://github.com/microsoft/onnxruntime/commit/0a03b230a7); MTP acceptance also changed. |
| 24 | **Hoist FP8 MTP GEMV weight-row addressing and block scales** | Fresh matched twenty-request pools: E2E **100.38 -> 102.16 TPS**, decode **126.30 -> 129.10 TPS** | **+1.77% observed E2E**, **+2.21% decode**; ten-route gate/up **114.44 -> 64.65 us**, down **53.36 -> 32.84 us**. Retained locally, not pushed. |

### Latest local follow-up: MTP scale hoisting

FP8 GEMV resolves each weight row once and loads one scale per 128-element
block rather than once per lane element. The FP16/BF16 weight rounding,
lane FMA sequence and warp reduction are preserved. Other block sizes
keep a generic path. Draft width, scheduling and token selection are
unchanged.

Forty exact measured requests compare against the published split-K
runtime in two reversed-order batches. All four candidate invocations
exceed 100 E2E TPS (**102.51, 101.15, 103.91, 101.12**), pooling at
**102.16 TPS**, versus fresh control **100.38 TPS**. Acceptance still
varies, so the entire E2E gain is not assigned to kernel savings.
Thirty-six full-QMoE graph cases match control bitwise; four C++ tests
pass, including 432 new projection combinations covering scale types,
FP16/BF16, layouts, K tails, generic blocks, GEMV/GEMM and graph replay.
Original runtime libraries remain unchanged; peak memory is not remeasured.

### Latest local follow-up: dense split-K

The original published 100.48-TPS result above remains unchanged. The
split-K follow-up uses the same exact workload, flags and width-two
overlay, with control/candidate then candidate/control ordering and five
measured requests after one excluded warmup per invocation. Candidate
invocations reach **100.99 and 101.76 E2E TPS**, pooling at **101.37 TPS**.
TTFT remains **1.054 s**; mean request time falls **5.136 -> 5.051 s**.

Fresh 64-token decode profiles confirm the targeted N=324, K=10240 vector
GEMV uses 64 instead of 32 splits. Its three-row mean GPU duration falls
**11.3%**, independently of request-wall timing. All 36 base-projection
graph comparisons pass (maximum absolute FP16 difference **0.0001221**),
and all 11 targeted C++ tests pass, including FP16/BF16, scalar/vector
kernels, higher split counts, workspace sizing, CUDA-graph replay and
forced/tuned dispatch.

MTP verification rounds total **2,233 -> 2,195** and accepted drafts
**2,872 -> 2,911** across the pools. Thus the full E2E gain is not solely
kernel savings. Changed reduction partitions can change floating-point
outputs; no general quality or complete greedy-sequence-equivalence claim
is made. The original isolated runtime is preserved. Source and tests are
in separate ORT commit [`0a03b230a7`](https://github.com/microsoft/onnxruntime/commit/0a03b230a7)
on `asonawane/qwen-38-flash-100-e2e`; peak memory has not been remeasured
for this candidate.

### Latest local follow-up: bounded sparse-indexer state replay

The opt-in `ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1` path limits sparse
indexer rollback copies to the committed key and auxiliary-buffer prefixes.
The default remains the existing full-capacity replay. Two reversed-order
Engine batches compare separate control/candidate packages with byte-identical
ORT and GenAI libraries; each pool has ten exact 8,192-input / 512-output
requests, width-two MTP, production overlay and sparse-prefill tensor-core
flags.

| Configuration | Requests | E2E TPS | Decode TPS | Mean TTFT |
|---|---:|---:|---:|---:|
| Control | 20 | 101.117 | 127.354 | 1.051 s |
| Bounded replay | 20 | 102.803 | 130.119 | 1.053 s |

The observed pooled gains are **1.67% E2E and 2.17% decode**. In a separate
matched 512-token Nsight profile, `ReplayStateUpdatesKernel` falls from
**54.56 ms to 3.68 ms** across 100 calls (about **545.6 to 36.8 us/call**,
93.3% lower). MTP rounds decrease from 4,462 to 4,421 and acceptance rises
from 73.58% to 74.49%, so the request-level difference is not attributable
solely to the replay kernel. All 20 requests complete 512 tokens with zero
MTP failures.

Four focused CUDA fixed-state tests pass, including active key/buffer prefix
updates and sentinel checks that inactive tails remain untouched. Paired
output sequences are not identical; independent repeats within each variant
also diverge early. This supports no claim of exact sequence equivalence or
model-quality equivalence. The change is retained as an opt-in experiment;
the default behavior is unchanged.

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
| Paired-row NVFP4 expert-weight reuse | Fresh paired/reversed pools: **100.17 -> 99.12 E2E TPS**, **1.05% lower**; 72 bitwise operator cases passed | Rejected; kernel and opt-in switch removed |
| Eight-column NVFP4 block geometry | Fresh paired/reversed pools: **101.18 -> 99.69 E2E TPS**, **1.47% lower**; 216 geometry-sweep bitwise cases passed | Rejected; original 16-column geometry retained |
| Two-way NVFP4 K-loop unrolling | Fresh paired/reversed pools: **102.29 -> 100.83 E2E TPS**, **1.43% lower**; 72 bitwise graph cases passed and operator screening improved | Rejected; unroll directive removed |
| FP8 MTP gate/up projection + SiLU fusion | Fresh paired/reversed pools: **102.62 -> 102.58 E2E TPS**, no demonstrated gain; 36 bitwise graph cases passed | Rejected; separate activation retained |
| Eight-warp FP8 MTP GEMV blocks | Fresh paired/reversed pools: **101.90 -> 100.68 E2E TPS**, **1.20% lower**; 36 bitwise graph cases passed | Rejected; four-warp blocks retained |
| Widen split-K GEMV eligibility to N=4096 for MTP input projections | N=2560, K=2560, M=1/2/3: GEMV **10.08-11.10 us**, cuBLAS **7.20-7.71 us** under graph-replay autotuning | Rejected at microbenchmark stage; cuBLAS wins all six tested shapes, so no full-request gain claimed |
| Same-warp paired FP8 gate/up projection + SiLU | Twenty exact paired/reversed requests: **101.680 -> 101.848 E2E TPS**, observed **+0.165%**; 36 bitwise graph cases passed | Rejected as inconclusive; candidate used 11 fewer draft rounds and shape-matched gate/up profile times were worse than the retained trace |
| Graph-only adjacent shared-expert SiLU fusion using existing ScaledSiLU | Twenty exact paired/reversed requests: **101.857 -> 100.874 E2E TPS**, observed **-0.965%**; 48 base pairs and one MTP pair fused | Rejected; 16 eager/graph numerical checks pass fixed tolerance but are not universally bitwise-equivalent; candidate required 19 more draft rounds |
| Sigmoid + HyperConnectionPreMix fusion | All-row: **102.125 -> 102.105 E2E TPS**, TTFT about **16 ms higher**. Decode-only, forty measured requests: **101.526 -> 102.280 TPS**, observed **+0.74%** | Not retained: first batch **+1.46%**, independent repeat only **+0.024%**. Matched small-row kernel sums improve **11-13%**; 80 bitwise cases per variant pass |
| Four-K-lane NVFP4 K-packed GEMV with a 32-column CTA tile | Eight exact Engine requests per variant: **104.508 -> 103.539 E2E TPS** and **132.793 -> 131.208 decode TPS** | Rejected: pooled E2E fell **0.93%**; per-call gate/up profile time rose **39.58 -> 40.78 us**, while down fell **20.91 -> 20.36 us**. One targeted MTP-window parity test passed; full-request sequence identity is not established. |

## Dense INT4 screening (synthetic operators, not E2E results)

Two timing-order-reversed screens compare the retained FP16 runtime with
its existing symmetric INT4 MatMulNBits path, FP16 activations, blocks
64/128 and rows 1/2/3. All 48 quantized-output comparisons pass against
explicit FP16 dequantization (rtol=0.02, atol=0.002); graph replay is
bitwise stable. The table shows block-128 GPU kernel-sum speedups after
a 256-MiB cache flush, as ranges across the two screens:

| Projection K -> N | M=1 speedup | M=2 speedup | M=3 speedup |
|---|---:|---:|---:|
| 2560 -> 7168 | 2.05-2.08x | 1.20-1.21x | 1.21-1.23x |
| 2560 -> 6144 | 1.99-2.01x | 1.14-1.16x | 1.69-1.73x |
| 6144 -> 2560 | 1.61-1.65x | **0.62x (slower)** | 1.15-1.16x |
| 2560 -> 248320 vocabulary head | 1.97x | 1.69x | 1.37x |

The vocabulary head takes approximately 292-294 us in FP16 versus
148-149/173-174/214-215 us in INT4 for rows 1/2/3. Packed weights plus
FP16 scales occupy 3.88x fewer bytes at block 128; this is not a measured
total GPU-memory reduction. Synthetic quantization introduces roughly
9-10.4% relative output L2 error versus original FP16, distinct from
kernel correctness. No actual model weights are quantized, no quality
or MTP-acceptance result is established, and no E2E speedup is claimed.
Selective calibrated INT4 experiments are warranted; blanket conversion
is not recommended given the two-row attention-output regression.

### Sparse indexer live-key radix selection: retained opt-in

The eligible short-QSA indexer can reuse its bounded radix selector on
the existing scored keys, scanning only live blocks instead of merging
the padded capacity. Exact early termination when a radix bucket is
fully selected avoids unnecessary finer passes. Scores, stable ties,
state updates and output counts are preserved; weights are unchanged.

| Comparison | Control E2E TPS | Candidate E2E TPS | Observed change |
|---|---:|---:|---:|
| First batch, ten requests per variant | 100.546 | 101.753 | +1.20% |
| Independent reversed-order repeat, ten per variant | 100.844 | 101.540 | +0.69% |
| Pooled twenty per variant | 100.695 | 101.646 | +0.94% |

Pooled decode throughput is **126.875 -> 128.494 TPS (+1.28%)**.
Mean TTFT is 1.057019 -> 1.060150 s. Speculative rounds also vary
(4470 -> 4442), so the entire E2E gain is not assigned to the kernel.
All forty measured Engine requests generate exactly 512 tokens with
zero MTP failures. This is a matched improvement, not a new absolute
throughput record.

Matched 30-replay profiles at 2048 live blocks measure selection
**35.456/37.424/42.144 -> 22.592/25.344/27.184 us** for one/two/three
rows, roughly 32-36% lower. Eleven merge kernels become one selector.
The measured package passes 336 FP16/FP32 and 168 BF16 bitwise graph
cases; 336 disabled-mode cases also match control. The final build
passes those 504 enabled graph cases and four expanded C++ tests in
each mode. Three additional unpaired final-build Engine requests pass;
their throughput is not included in the gain estimate.

**Retained locally, opt-in and not pushed.** With the rebuilt CUDA
provider, set `ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=1` before the first
eligible call. The default remains off; long-context throughput is not
established. The isolated measured and final-validation runtime
packages are preserved separately.

### Sparse indexer local merge fusion: rejected

Combining the first five hierarchical top-K merge levels in shared
memory reduced merge launches from eleven to seven, but did not improve
the exact Engine workload: ten requests per variant measured
**101.784 control -> 101.322 candidate E2E TPS (-0.45%)**.
All 336 FP16/FP32 merge-boundary graph cases matched control bitwise.
Matched 30-replay profiles at 2048 live blocks measured merge GPU time
**35.392/37.504/42.048 -> 39.792/40.271/41.088 us** for one/two/three
rows. Fewer launches did not overcome slower local merge work.
The candidate is not retained; original models and measured runtimes
are unchanged.

### Actual-weight INT4 MTP vocabulary head: Engine experiment

Only the MTP draft vocabulary head is changed to symmetric INT4,
block 128, in an isolated model. Target graph/weights and runtime remain
unchanged. Forty exact Engine requests compare original and candidate
in two reversed-order batches:

| Comparison | Control E2E TPS | INT4 draft-head E2E TPS | Observed change |
|---|---:|---:|---:|
| First batch, ten requests per variant | 101.414 | 102.472 | +1.04% |
| Independent repeat, ten per variant | 103.036 | 102.884 | -0.15% |
| Pooled twenty per variant | 102.218 | 102.677 | +0.45% |

Pooled evaluated-draft acceptance is 74.78% control versus 74.39% INT4;
draft rounds are 4410 versus 4435. Every request completes with exactly
512 tokens and zero MTP failures. A separate profile confirms INT4
draft-head kernels and the original FP16 target-head kernel.

**Not promoted to the retained configuration:** the E2E gain does not
clearly repeat. This is uncalibrated round-to-nearest quantization;
there is no broader quality or real-hidden-state logit-agreement
validation. Engine is used for all throughput requests. The isolated
candidate remains available for follow-up; original models are untouched.

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

## Detailed records and reproduction

- [Optimization log and experiment details](../../../docs/Qwen38MtpEngramIssues.md)
- [Portable reproduction instructions](README.md)
- [Benchmark launcher](run_benchmark.sh)

Original raw reports, profiles and memory samples remain local session
artifacts and are not published with this branch. Use the checked-in
benchmark and overlay to generate fresh reports.
