# Qwen3.8 Flash NVFP4 Optimization Results

The latest memory candidate reaches **99.79 independent E2E TPS**,
**133.33 decode TPS**, and **79.95 GiB sampled peak**. See the
[short optimization summary](README_MEMORY.md) for the selected changes,
portable graph preparation and important context/memory limits.
The sections below are a chronological experiment log: descriptions such
as "local", "uncommitted" and "not pushed" record status at measurement
time. Successful runtime changes and default-off diagnostic paths are now
included on `asonawane/qwen-38-flash-100-e2e`; rejected model exports,
binary snapshots and local raw evidence are not published.

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

## INT2 target-expert screening (initial export invalid; corrected results below)

**The initial INT2 results below are INVALID as an assessment of INT2
quantization.** The experimental exporter retained the source concatenated
`[gate | up]` rows, whereas `swiglu_fusion=1` requires interleaved
`[g0,u0,g1,u1,...]` rows. Kernel-versus-fallback parity did not detect this
shared semantic error. All 48 candidate gate/up weights and scales have
since been interleaved, and the exporter now follows the repository builder.
The original NVFP4 model is untouched. Historical JSONs are preserved;
their throughput and repetition observations cannot establish the quality
or performance of correctly converted INT2.

Independent source-byte and PyTorch routed-SwiGLU reference checks now pass
for layers 0, 23 and 47, rows one/three, and the first ten experts
(six cases; maximum absolute error approximately 1.9e-6 to 3.8e-6).
A corrected exact 8K/512 smoke completes with zero MTP failures at
103.319 E2E / 132.256 decode TPS. That **one measured request** was not a
paired throughput gain or a model-quality validation. The subsequent
corrected multi-prompt, throughput and memory evaluation is reported below.

An isolated candidate quantizes all 48 target routed-expert layers from
the original BF16 `Qwen/Qwen3.8-Flash-Next` checkpoint, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, using the repository's symmetric
signed-scale INT2 quantizer, block size 64 and FP16 scales. Non-MoE
projections, tokenizer, Engram and FP8 MTP remain unchanged. This is not
whole-model INT2 and is not a requantization of NVFP4 weights.

The current INT2 GEMV does not accept this model's top-10 expanded rows.
An experimental ORT flag, `ORT_QMOE_INT2_LARGE_EXPERTS=1`, extends packed
grouped execution/prepacking to 512 experts; grouped kernels serve both
prefill and decode without increasing the dense-fallback scratch limit.
Six FP16 regressions pass against the fallback, covering projection biases,
one/three/32 rows, blocks 64/128 and CUDA graph capture/replay. Actual
first-layer weights also pass eager-fallback versus packed-graph comparison.

The invalid initial export's ten measured requests per variant, in
reversed-order batches, used the
same isolated provider, exact 8,192/512 Engine workload, width-two MTP,
standard overlay, both tensor flags and bounded indexer replay:

| Variant | Pooled E2E TPS | Decode TPS | Mean TTFT | Draft acceptance |
|---|---:|---:|---:|---:|
| Retained NVFP4 model | 101.792 | 128.490 | 1.053 s | 73.92% |
| INT2 target experts (**INVALID layout**) | 101.575 | 128.919 | 1.077 s | 81.28% |

In that invalid run all requests completed 512 tokens with zero MTP
failures, but **all ten INT2 outputs repeated an 86-token sequence exactly**
and had the same output
hash. None of the ten NVFP4 outputs shows that exact periodic pattern.
Higher draft acceptance therefore does not establish quality or usefulness.
E2E regresses 0.21%, and the 0.33% decode difference is not a demonstrated
gain. The earlier rejection of INT2 on this evidence is withdrawn because
the exported model was incorrect; corrected INT2 still requires broader
quality evaluation before deployment. The original model is untouched. The candidate
package still contains copied baseline external data, including unused
NVFP4 expert bytes, so its directory size is not a compact INT2 size result.

The isolated candidate provider SHA256 is
`c4e32a6920b52e7cb661fdae695ab44ef7a71ba75cb53dc1934e13974f77fe7b`.
Model-conversion scripts, all request JSONs, manifests and the pooled
comparison remain local experiment artifacts.

### Corrected INT2 and mixed expert evaluation

All three candidates use the original BF16 source checkpoint, interleaved
gate/up rows, integer block size 64 and FP16 scales. Only the 48 target
routed-expert layers change; non-MoE target weights, Engram and FP8 MTP
remain identical to the retained model. `G2/D4` means INT2 gate/up and
INT4 down; `G4/D2` is the reverse. INT4 export bytes and scales match the
repository quantizer for the first eight experts of every converted
projection. Both mixed candidates also pass six independent source-byte
and PyTorch routed-SwiGLU checks across layers 0/23/47 and rows one/three;
maximum absolute errors are 1.9e-6 to 3.8e-6.

**Bounded quality gate:** render the packaged chat template with
`enable_thinking=False`, greedy width-two Engine generation, up to 128
output tokens, and fresh requests. The eight known-answer prompts cover
geography, arithmetic, sorting, Python evaluation, extraction,
translation, JSON and contextual arithmetic. Two additional prompts
check an idempotency explanation and Python code generation. The
predeclared gate requires at least seven correct known answers, preservation
of every baseline-correct answer, no empty output, no 8-token ngram
occurring three times, and no run of twelve identical tokens.

**All four variants score 8/8, with no empty/repetitive outputs or MTP
failures.** All open-ended explanations are coherent, and all generated
functions correctly call `max` on the nonempty input list. This is a
small smoke screen, **not perplexity or broad model-quality validation**.
Teacher-forced logit comparison was not performed: the current Engine
Python bindings do not expose logits, and Generator rejects the paged
recurrent-state configuration.

The corrected full workload uses the same isolated provider and flags
for all variants: both tensor-attention flags, bounded indexer replay,
`ORT_QMOE_INT2_LARGE_EXPERTS=1` and the standard width-two tuning overlay.
Radix selection and the dequantization scratch override remain unset.
Eight invocations run in order
`NVFP4, G2/D4, G4/D2, INT2, INT2, G4/D2, G2/D4, NVFP4`,
each with one excluded warmup and five measured requests. Thus each
variant pools ten exact uncached 8,192-input / 512-output requests.
Controls bracket reversed candidate order; this is not a separate
adjacent ABBA experiment for each candidate.

| Target experts | Pooled E2E TPS | Change vs control | Decode TPS | Mean TTFT | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| Retained NVFP4 | 102.460 | -- | 129.590 | 1.054 s | 74.35% |
| Corrected INT2 gate/up + down | 103.365 | +0.88% | 132.279 | 1.090 s | 82.04% |
| INT2 gate/up + INT4 down | 94.226 | -8.04% | 117.508 | 1.085 s | 71.94% |
| INT4 gate/up + INT2 down | 94.962 | -7.32% | 118.342 | 1.074 s | 70.60% |

All 48 requests including warmups complete exactly 512 tokens with zero
MTP failures. None of the forty measured outputs has an exact full-output
period of 128 tokens or less. The long throughput prompt itself is
repeated raw text and is **not** a quality prompt. Corrected INT2 batches
reach 103.257 and 103.473 E2E TPS versus controls 102.403 and 102.517.
The observed 0.88% E2E / 2.08% decode improvement is small, and differing
draft acceptance/output trajectories prevent assigning it entirely to
faster kernels. INT2 TTFT increases approximately 36 ms.

Memory is measured separately through loading, one warmup and one exact
measured request per variant with the same runtime/flags. Device-wide
NVML sampling targets 20 ms on otherwise idle GPU 0; maximum observed
sample gaps are 80-89 ms. Host peaks use the larger of `/proc` VmHWM and
child-process wait RSS.

| Target experts | Sampled GPU peak | Host process peak RSS | GPU change vs control |
|---|---:|---:|---:|
| Retained NVFP4 | 100.09 GiB | 2.90 GiB | -- |
| Corrected INT2 gate/up + down | 92.18 GiB | 1.82 GiB | -7.90% |
| INT2 gate/up + INT4 down | 109.95 GiB | 2.31 GiB | +9.85% |
| INT4 gate/up + INT2 down | 127.69 GiB | 2.21 GiB | +27.57% |

The current integer path retains raw initializers in addition to
prepacked weights (`is_packed=false` for mixed/INT2 projections in
`QMoE::PrePack`), so smaller quantized source bytes do not directly
translate into the same GPU-memory saving. Mixed projections increase
both weight storage and the measured peak. These are sampled device
peaks, not exact allocator high-water marks or minimum GPU capacities;
GPU and host peaks need not coincide. The older 3.61-GiB host result below
used three measured requests and a different diagnostic runtime, and
must not be substituted for this matched current comparison.

**Outcome:** corrected all-INT2 target experts are a promising
memory-saving experimental candidate, not the quality rejection implied
by the invalid initial export. The mixed candidates are not retained as
performance/memory improvements for this runtime. Keep the original
NVFP4 deployment baseline until broader held-out quality validation is
complete; the small observed speed gain alone is insufficient reason to
switch.

**Later quality update:** the expanded 64-case screen below finds
reproducible answer regressions for corrected INT2 as well as simulated
INT4 activations. The memory result remains valid, but the candidate is
not recommended over NVFP4 on the current quality evidence.

Local evidence is preserved in the experiment session as
`corrected_expert_evaluation_summary.json`, `corrected_full_*.json`,
`quality_*.json`, `source_mapping_g*.json` and `peak_*_results.json`
with raw memory CSVs and logs. Runnable model directories are
`/home/asonawane/flash/qwen_38_flash_int2_moe_engine`,
`/var/tmp/asonawane-qwen38-g2d4-engine` and
`/var/tmp/asonawane-qwen38-g4d2-engine`; each contains its own regular
external-data files and quantization manifest. Copied unused NVFP4 bytes
remain in these packages, so directory sizes are not compact-model size
measurements. No original model or installed runtime was modified.

### INT2 weights with simulated INT4 MoE activations

A subsequent **quality-only** experiment uses the corrected INT2 model
unchanged and a separately built CUDA provider with
`ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS=1`. Each contiguous 64-element
activation group is quantized/dequantized dynamically with FP32
`max(abs(group))/7` scales, symmetric signed codes `[-7,7]`, nearest
ties-even rounding, and FP16/BF16 dequantized values. QDQ is applied both
before gate/up GEMM and after SwiGLU before down GEMM. Routing, attention,
residuals, caches, non-MoE weights and FP8 MTP remain unchanged.

This **does not execute native INT4 activation arithmetic or retain
packed INT4 activations**. Expert GEMMs still consume FP16/BF16; no
native W2A4 speed or memory benefit is measured or claimed. The mode is
default-off, bypasses GEMV for affected nodes, and fails explicitly if
the packed grouped path is unavailable rather than silently skipping QDQ.

Fourteen focused CUDA tests pass: six FP16/BF16 independent-reference
and graph-replay cases at rows 1/3/32, two unsupported-path rejection
cases, and six existing large-expert disabled-mode regressions. Tests
include zero groups, outliers, ties-even inputs, bias handling, unchanged
original input buffers and a measurable difference from the unquantized
reference. Six additional actual-weight cases at layers 0/23/47 and
rows 1/3 pass the existing source/PyTorch reference tolerances; maximum
absolute output error is 0.000411. QDQ thresholding can magnify small
activation-rounding differences, so this is not bitwise numerical parity.

The same ten chat-template prompts and frozen answer-preservation,
nonempty-output and repetition gates compare flag-off and flag-on runs
using the **same new provider**:

| Variant | Known answers correct | Empty/repetitive outputs | MTP failures | Bounded quality gate |
|---|---:|---:|---:|---|
| INT2 weights, FP16 activations (flag off) | 8/8 | 0 | 0 | Pass |
| INT2 weights, simulated INT4 activations (flag on) | 8/8 | 0 | 0 | Pass |

Both explanations are coherent and both code outputs correctly use
`max` on the nonempty list. Every baseline-correct known answer is
preserved. This is a small smoke screen, **not broad held-out quality
validation or perplexity evidence**. The new flag-off control differs
from the earlier retained-provider run in JSON formatting (plain JSON
instead of a fenced JSON block), while preserving the correct object;
end-to-end bitwise parity across the provider rebuild is not claimed.
The existing disabled-mode operator parity regressions pass.

The screening provider SHA256 is
`3f239a94cc2e1a5b9cddf8710ce618410a186fe6b8cf7e7b7bdf8d420a14c113`.
The retained INT2/NVFP4 provider is untouched. Local evidence is
`int4_activation_quality_summary.json`, `quality_int4_act_control.json`,
`quality_int4_act_candidate.json` and `int4_act_source_mapping.json`,
with associated logs, in the same experiment session.

**Outcome:** group-64 symmetric INT4 activation simulation passes this
bounded INT2-MoE quality screen. Broader held-out quality checks should
precede native-kernel investment; this result does not establish that
native W2A4 is supported or faster on the H200.

### Expanded activation quality screen: both INT2 candidates fail

The initial 8/8 smoke result does not generalize to the subsequent
**64-case custom answer-key suite**, frozen before running all variants.
There are eight new cases in each of arithmetic, word problems, Python
evaluation, logical reasoning, extraction, structured JSON, multilingual
instructions and long-context retrieval. The latter embeds 450 records
and queries different positions; prompts range from 29 to 6,341 tokens.
This is a wider custom screen, **not a standardized held-out benchmark
or perplexity measurement**.

The predeclared absolute gate is at least 90% correct (58/64), with no
empty/repetitive output. The paired gate additionally requires preserving
all INT2-A16-control-correct answers when enabling simulated INT4.
Grading is strict against the frozen answer key: correct arithmetic
with extra explanation still fails an explicit only-integer instruction.
No generated code is executed.

All variants use the same screening provider, greedy generation,
`enable_thinking=False`, width-two MTP, up to 128 generated tokens,
fresh requests, and identical optimization flags. Because GPU 0 was
occupied, the quality-only experiment uses idle physical GPU 6. Variant
order is NVFP4, INT2 A16, INT2 simulated A4, then the reverse order.
There are 384 requests total, but only **64 distinct cases per variant**:
the repeat is a reproducibility check, not additional independent samples.

| Target expert weights / activations | First score | Repeat score | Accuracy | Absolute gate |
|---|---:|---:|---:|---|
| Retained NVFP4 / FP16 | 58/64 | 58/64 | 90.63% | Pass |
| Corrected INT2 / FP16 | 50/64 | 50/64 | 78.13% | Fail |
| Corrected INT2 / simulated INT4 | 52/64 | 52/64 | 81.25% | Fail |

| Category (8 cases each) | NVFP4 | INT2 A16 | INT2 simulated A4 |
|---|---:|---:|---:|
| Arithmetic | 6 | 4 | 6 |
| Word problems | 4 | 2 | 2 |
| Python evaluation | 8 | 6 | 6 |
| Logical reasoning | 8 | 7 | 7 |
| Extraction | 8 | 8 | 8 |
| Structured JSON | 8 | 7 | 7 |
| Multilingual | 8 | 8 | 8 |
| Long-context retrieval | 8 | 8 | 8 |

Every request completes with zero MTP failures and no empty/repetitive
output. Both INT2 variants repeat all token sequences exactly. NVFP4
changes one incorrect word-problem answer from 44 to 49 across repeats
(correct answer 46); no case's correctness changes.

Relative to NVFP4, INT2 A16 loses nine previously correct cases and
recovers one; simulated A4 loses seven and recovers one. Enabling
simulated A4 relative to INT2 A16 loses two correct cases and recovers
four, so its higher aggregate score **does not satisfy answer
preservation**. Examples:

- Python `len({i % 3 for i in range(10)})` is correctly answered as 3
  by both NVFP4 and INT2 A16, but simulated A4 answers 2 in both runs.
- Sorting `[14,-3,14,0,8]` is correct under NVFP4; both INT2 variants
  place 0 before -3.
- The 420-part worker word problem is correct under NVFP4; both INT2
  variants answer 108.
- INT2 A16 answers a travel-time question correctly as 6; simulated A4
  produces an explanation truncated at 128 tokens instead of the
  requested integer. This is an instruction-following/length failure,
  not evidence that its intermediate arithmetic is wrong.

**Updated outcome:** do not promote either INT2 candidate over NVFP4
on this quality evidence, and do not treat the initial smoke pass as a
deployment-quality result. The observed INT2 memory saving is real for
the tested configuration, but does not compensate for these answer
regressions. Native W2A4 kernel work is not justified by this screen
alone. A next accuracy investigation should isolate INT2 weight error
(for example with higher-precision sensitive projections or calibrated
quantization) before reducing activation precision further.

No original BF16 full-model reference is evaluated, so these findings
establish differences relative to the retained NVFP4 model, not absolute
quality loss versus BF16. Local evidence is
`int4_activation_extended_suite.json`,
`int4_activation_extended_summary.json`,
`extended_quality_*.json` and `extended_quality_repeat_*.json`, with
associated logs. The suite SHA256 is
`69e25f742e00f9ca26a47ab86f58b97d56bd36515f60b4ce92c37126b1c90526`.
No throughput or memory numbers are inferred from these quality runs.

### INT2 block-size 32 accuracy experiment: not retained

An independent candidate requantizes all 48 target routed-expert layers
from the original BF16 checkpoint using INT2 block size **32** instead
of 64, with FP16 scales and FP16 activations. Gate/up rows are interleaved
correctly; non-MoE target tensors, Engram and FP8 MTP remain unchanged.
Simulated INT4 activations are disabled.

GPU export is bitwise checked against the repository quantizer for the
first eight experts of every one of the 96 projections. ONNX checker
passes. Six actual-source byte and independent PyTorch routed-SwiGLU
cases at layers 0/23/47 and rows 1/3 pass, with maximum absolute output
error 3.81e-6. The existing packed grouped path supports block 32 without
a runtime kernel change.

The unchanged frozen 64-case suite runs twice on GPU 6 using the same
screening provider, flags and generation options as the preceding
block-64 experiment. A fresh NVFP4 control follows the block-32 repeat.

| Variant | Correct answers | Repeat | Accuracy | Frozen 58/64 gate |
|---|---:|---:|---:|---|
| Fresh retained NVFP4 control | 58/64 | -- | 90.63% | Pass |
| INT2 block 64, preceding same-provider comparison | 50/64 | 50/64 | 78.13% | Fail |
| INT2 block 32 | 49/64 | 49/64 | 76.56% | Fail |

Block-32 repeats every token sequence exactly. All 192 new requests
complete with zero MTP failures and no empty/repetitive output.
Relative to block 64, block 32 loses four correct answers and recovers
three; relative to the fresh NVFP4 control it loses nine and recovers
none. Reasoning improves from 7/8 to 8/8, but Python falls from 6/8 to
5/8 and structured-output tasks from 7/8 to 6/8. Arithmetic and word
problem scores remain 4/8 and 2/8. Correct computations with extra
explanation still fail explicit only-integer instructions; this remains
a strict custom task screen, not standardized quality/perplexity evidence.

**Outcome:** reducing the INT2 weight block size does not improve the
measured task score. Keep NVFP4 and investigate calibrated quantization
or selective higher precision rather than assuming smaller groups solve
the accuracy regression. Throughput and GPU/host peak memory are
deliberately **not benchmarked**, because the candidate fails quality.
Its expert weight/scale data contains 37,748,736,000 bytes (35.16 GiB)
versus 33,973,862,400 bytes (31.64 GiB) for block 64; these are stored
expert-data bytes, not live GPU-memory measurements. The package still
contains unused copied NVFP4 tensors and is not compact.

The runnable experiment is
`/var/tmp/asonawane-qwen38-int2-b32-engine`, with validation and evaluation
in its manifest. Local evidence is `int2_b32_quality_summary.json`,
`int2_b32_source_mapping.json`, `extended_quality_int2_b32.json`,
`extended_quality_int2_b32_repeat.json` and
`extended_quality_int2_nvfp4_b32_control.json`, with export/quality logs.
The original model, prior candidates and retained runtimes are untouched.

### Activation-aware INT2 calibration and compact storage

The updated goal is **smaller on-disk model size with NVFP4 accuracy
preserved**, not maximum throughput. The agreed gate preserves every
NVFP4-correct answer on the frozen 64-case suite and on a separate
32-case validation set. Sixteen different calibration prompts are
frozen independently; no prompt overlaps the calibration and validation
sets. These remain custom suites, not a standardized benchmark or an
accuracy guarantee on arbitrary tasks.

A separately built opt-in diagnostic provider captures the first 128
prefill rows of inputs/router logits for all 48 NVFP4 QMoE layers on
each calibration prompt (768 files, approximately 580 MiB). Capture
disables CUDA graphs and MTP, synchronizes copies, and is never used
for throughput measurement. Target routing determines expert-specific
activation second moments. Down-projection inputs are reconstructed
using the original BF16 gate/up weights and FP16 SwiGLU, with at most
32 routed samples per observed expert. Unobserved experts use an
explicit layer-global prior, and channel importance has a 1% mean floor.
This coverage limitation is recorded, not silently treated as calibrated
expert-specific evidence.

The exporter fits signed INT2 block-64 scales/codes with an
activation-second-moment-weighted error objective, searches both scale
signs and clipping factors, and performs least-squares scale refinement.
It preserves the repository quantization result whenever fitting does
not improve the measured group objective. This is a **diagonal
activation-aware approximation**, not full-covariance GPTQ or
quantization-aware fine-tuning. Aggregate weighted fitting error drops
29.34%, but that proxy does not establish model accuracy.

The new package is genuinely compact: only referenced non-MoE target
initializers are copied, shared initializer offsets in GenAI configuration
and MTP graphs are updated, and unused baseline NVFP4 expert data is
omitted. All unchanged target external ranges are byte-verified against
baseline. Six calibrated source-byte/PyTorch routed-SwiGLU cases pass,
with maximum absolute output error 1.91e-6. The target ONNX checker
passes and the full Engine runs both quality suites.

| Metric | NVFP4 reference | Calibrated INT2 |
|---|---:|---:|
| Real package files, including Engram/MTP/tokenizer | 122.96 GiB | 91.31 GiB |
| Frozen suite | 58/64 | 48/64 |
| Separate validation | 27/32 | 24/32 |
| Previously correct answers lost | -- | 10 frozen + 4 separate |

**The package shrinks 25.74%, but the same-accuracy gate fails.**
No empty/repetitive outputs or MTP failures occur. Validation uses the
same retained screening provider as preceding quality comparisons,
FP16 activations, width-two MTP and GPU 6, with diagnostic capture unset.
The separate NVFP4 control is fresh; the frozen control is the preceding
same-provider/GPU/suite control. Evaluation is one run per suite, not a
statistical equivalence claim.

**Outcome:** do not promote this candidate. Activation-aware diagonal
fitting alone does not rescue INT2 on this workload. Further experiments
would need stronger calibration (full covariance/broader coverage) or
selective higher precision, followed by the same untouched accuracy
gates. On-disk size reduction is measured, not inferred from weight bits;
no throughput or GPU-memory benchmark is run for this rejected candidate.

The runnable package is
`/var/tmp/asonawane-qwen38-int2-calibrated-engine`.
Local evidence includes `calibrated_int2_summary.json`,
`int2_calibrated_source_mapping.json`, `quality_calibrated_frozen.json`,
`quality_calibrated_separate.json`, `quality_nvfp4_separate.json`,
calibration captures/moments, and their manifests.
Calibration-suite SHA256:
`0452417e41bcec2204eb12dc813a078cf61e64890e089644bf8b1f2336ff220b`.
Separate-validation SHA256:
`f831e524731c829905e8ca36c19f4a32ffa5e4f2e156349d57dd765eb25134c4`.
The diagnostic capture provider SHA256 is
`277c2bc93079ee04bb7d4e2fec8a50f420c649622aec08c7abaab404828949ea`.
Original models and retained runtime packages remain untouched.

### Selective INT2 in twelve low-error layers: accuracy gate fails

To prioritize smaller storage while preserving NVFP4 quality, a
conservative candidate replaces only **12 of 48 target MoE layers** with
the preceding calibrated INT2 weights. The other 36 retain their exact
NVFP4 weights, attributes and global scales; activations remain FP16.
Layer selection is frozen using calibration data only: rank by the
maximum gate/up or down activation-weighted normalized squared weight
error, including all 512 source experts in the reference-energy
normalization, then select the lowest twelve. This diagonal proxy is not
a causal sensitivity ablation and does not use validation answers.

Selected zero-based layers are
`0, 1, 15, 31, 40, 41, 42, 43, 44, 45, 46, 47`.
The compact package copies only referenced initializers, remaps shared
configuration/MTP offsets, and preserves ordinary regular external files.
The target ONNX checker passes, all copied external ranges are verified
against their source-copy SHA256 values, and full Engine evaluation
confirms the 12 INT2 / 36 NVFP4 composition is runnable.

| Metric | Fresh NVFP4 reference | Selective INT2 |
|---|---:|---:|
| Actual package files | 122.96 GiB | 115.04 GiB |
| Frozen suite | 58/64 | 51/64 |
| Separate validation | 26/32 | 26/32 |
| Baseline-correct answers lost | -- | 7 frozen + 1 separate |

**Size drops 6.44%, but accuracy preservation still fails.** Equal scores
on the separate set conceal one regression and one recovery. For example,
NVFP4 correctly answers Python `len({i // 4 for i in range(15)})` as 4,
while the selective candidate answers 3. Seven frozen-suite regressions
include arithmetic, worker/capacity word problems and structured-output
format/extraction failures. No empty/repetitive output or MTP failures
occur in the 192 requests.

Fresh controls and candidate use the same screening provider, physical
GPU 6, unchanged frozen64/separate32 prompts, FP16 activations, greedy
thinking-disabled width-two generation and existing flags. The fresh
separate NVFP4 score is 26/32 rather than its earlier 27/32; conclusions
use paired case-level preservation against the current control, not
substitution of the earlier score. This remains a one-run custom screen,
not standardized quality or statistical-equivalence evidence.

**Outcome:** keep NVFP4; the calibration-error ranking does not identify
a twelve-layer INT2 subset that meets the same-accuracy goal. Do not
tune the subset to these validation failures while continuing to call
the same sets held out. Further work needs calibration-only causal
layer/projection sensitivity, stronger covariance-based quantization,
or a different compression method, with fresh untouched validation
before acceptance. Throughput and GPU-memory benchmarks are omitted
because this candidate fails the required accuracy gate.

The runnable experiment is
`/var/tmp/asonawane-qwen38-selective-int2-12-engine`.
Local evidence is `selective_int2_layer_selection.json`,
`selective_int2_summary.json`, `quality_selective12_*.json` and
`quality_nvfp4_selective_*.json`, with the package copy records/manifest
and build/ranking logs. Original models and retained runtimes are untouched.

## Calibrated INT3 numerical screen in INT4 storage

Screened all 48 target MoE layers using signed eight-level weights
`[-4,3]`, block64, calibrated FP16 scales, and unchanged FP16 activations.
Calibration reused only the sixteen calibration prompts and their
activation moments; no evaluation answers were used for fitting.
HF gate/up rows were interleaved before quantization. Attention, routing,
Engram, non-MoE tensors and FP8 MTP were unchanged.

This is **not native INT3**: codes are carried in the supported INT4
container, offset by eight, and QMoE uses `expert_weight_bits=4`.
No runtime/provider changes or new environment switches were needed.
Existing comparison flags and isolated screening provider were retained.

Six actual-source byte/PyTorch routed-SwiGLU reference cases passed
on layers 0/23/47 with one and three input rows, maximum absolute error
`3.8147e-6`. All 60,397,977,600 packed weight bytes were independently
scanned: every nibble is in `[4,11]`, decoding only to `[-4,3]`.
The target ONNX checker and full Engine execution passed.

Fresh paired controls on physical GPU6 used greedy chat, width-two MTP
and at most 128 generated tokens, with identical configuration/providers:

| Suite | NVFP4 correct | INT3 correct | Lost control-correct answers | Recoveries |
|---|---:|---:|---:|---:|
| Existing frozen64 | 58/64 | 53/64 | 5 | 0 |
| Existing separate32 | 26/32 | 26/32 | 1 | 1 |
| Fresh INT3 validation32 | 25/32 | 26/32 | 0 | 1 |

All 256 requests completed without MTP failures, empty answers or
repetition failures. The fresh32 suite was frozen before generation and
is prompt-disjoint from calibration and both existing suites, but uses
new instances of the same custom task families; it is not a standardized
benchmark. Its SHA256 is
`58f53b38b64e2e21d5c735087781c66545c70736ef3cc75cb6127ba5454a8dd8`.
Neither model meets the previously frozen 90% absolute threshold on
either 32-case suite; baseline-relative preservation is reported
separately.

**Rejected for the same-accuracy objective.** The fresh suite result is
encouraging but does not negate six regressions on existing suites.
INT3 frozen64 improves over calibrated INT2's earlier 48/64, without
establishing accuracy equivalence. Native three-bit packing, throughput
and GPU-memory work were not undertaken.

The independent regular-file screen package is
128,245,186,147 bytes (119.44 GiB), versus NVFP4's
132,025,315,338 bytes (122.96 GiB). The roughly 2.86% package reduction
comes from the INT4/block64 representation and compact packaging,
**not three-bit storage**. Original models/runtimes remain untouched.

Local evidence is `int3_screen_summary.json`, `int3_source_mapping.json`,
`quality_int3_*.json`, `quality_nvfp4_int3_*.json` and
`int3_fresh_validation_suite.json`. The runnable experiment and rejected
manifest are in `/var/tmp/asonawane-qwen38-int3-screen-engine`.

## Conservative NVFP4 expert pruning without recovery training

Removed 16 of 512 routed experts in each of the 48 target layers,
leaving 496 experts with top10 routing. Engram and FP8 MTP were unchanged.
All retained NVFP4 weights, scales, global scales and biases are copied
without requantization. Router columns are sliced in exactly the same
retained-expert order, with private reshape constants updated to 496.
No runtime kernel changes or recovery training were used.

Selection was frozen using 64 broader calibration-only prompts across
software, math/science, multilingual, structured-data, reasoning,
long-record, writing and practical topics. Four independent capture runs
collected 3,072 input/router files. Coverage is 387-487 of 512 experts per
layer; **all unobserved experts were protected**, not assumed expendable.
The calibration SHA256 is
`0743635342eadeb44607ad1d0dd96a61e88e2b782ddcc1e3786cb1fe966a2ecd`.

The score minimizes worst-category normalized routing-weighted squared
expert-output energy. Output energy uses BF16 source reconstruction with
at most sixteen routed samples per expert; capture samples the first
128 prefill rows. This is a contribution proxy, not causal ablation or
proof of expert redundancy, and does not model replacement-expert
behavior or cancellation between experts. No validation answers informed
selection.

Independent checks verified 66,129,291,264 bytes of retained expert data
and router columns against the baseline. Nine CUDA reference cases on
layers 0/23/47 with 1/3/17 rows matched a baseline with removed experts
masked before routing; maximum absolute error was `7.6294e-6`.
Target ONNX checking and full Engine execution passed.

Paired greedy-chat accuracy on GPU6, width-two MTP, including a
reverse-order repeat:

| Suite (both runs) | NVFP4 correct | Pruned correct | Lost control-correct answers per run |
|---|---:|---:|---:|
| Existing frozen64 | 58/64 | 58/64 | 0 |
| Existing separate32 | 26/32 | 26/32 | 0 |
| Fresh pruning32 | 25/32 | 24/32 | 1 |

Fresh validation was frozen before generation, prompt-disjoint from all
previous suites and calibration, with SHA256
`ce265793121002b280a76fb6ea6587a7b3b22b31005467d48a3fc6e1ed323227`.
It uses new instances of existing custom task families, not a
standardized benchmark. The first fresh regression is `pruning_fresh_11`, a Python sorted-list
evaluation. The reverse-order repeat instead loses `pruning_fresh_20`,
a JSON sorting case; the original Python regression does not repeat.
Thus the aggregate fresh score repeats but the failed case varies.
Equal scores on the established suites do not establish preservation
on new questions.

All **512 requests** completed without MTP failures, empty answers or
repetition failures. Engram/MTP data and other copied non-graph files
were independently hash-verified unchanged. MTP graph external references
and shared-initializer configuration were remapped to the compact
target data, without changing their weight values.

**Rejected for the strict same-accuracy goal.** This candidate is closer
than the low-bit quantization candidates on these screens, but still
loses a baseline-correct answer in each fresh run. No throughput or
GPU-memory claims are made. A next experiment could use recovery
distillation or a smaller pruning budget; neither is started, and any
new selection must not be tuned to these fresh-suite failures while
calling that suite held out.

The independent compact package is approximately **120.97 GiB** versus
**122.96 GiB**, saving **1.99 GiB**, about **1.62%** of the full package.
Excluding unchanged Engram data, the remainder falls from **75.27 GiB**
to **73.28 GiB**, approximately **2.64%**. Removing 3.125% of experts
does not remove 3.125% of all package bytes.

Local evidence is `pruning_expert_selection.json`,
`pruning_mapping_validation.json`, `pruning_summary.json`,
`quality_pruned16_*.json`, `quality_nvfp4_pruning_*.json` and
`pruning_fresh_validation_suite.json`. The runnable candidate is
`/var/tmp/asonawane-qwen38-pruned16-engine`, with `pruning_manifest.json`.
Original models and retained runtimes remain untouched.

## NVFP4 expert offloading feasibility prototype

Built a separate model using a CPU-only `flash.offload::Nvfp4ExpertStage`
custom operator and the existing CUDA `QMoE`. The expert initializers
remain CPU-side; each invocation gathers the sorted union of the
router's top10 experts and copies only those compact tensors to CUDA.
The compact router preserves original logits and expert ordering.
GPU QMoE still performs all expert computation. This is **not** the
upstream FP16/BF16 CPU-execution offloading PR, not disk-backed expert
lookup, and not a persistent expert cache.

All NVFP4 weights, scales, biases, Engram and FP8 MTP weight files were
independently hash-verified unchanged. No quantization, pruning or
training was applied. The standalone custom library builds with
`-Wall -Wextra -Werror`; original repositories' runtime packages are
unchanged. Only the separate candidate config loads the custom library.

Four isolated CUDA layer0 reference cases with 1/3/17/128 rows passed
against original NVFP4, maximum absolute error `7.6294e-6`.
Profiling verifies CPU placement of the staging operator. Synthetic
tests verify exact router/expert slice ordering, scales/biases, changing
active sets (10/27/470 experts), and explicit invalid-index rejection.

The dynamic expert union is bounded by 512 per invocation; ordinary
1-3 row decode stages at most 10-30 experts, while prefill can select all
512. This prototype does not promise a smaller fixed per-layer cache
capacity. Allocation/reuse is managed by ORT, so live memory and arena
high-water marks must be measured rather than inferred from active counts.
Layer-local symbolic expert dimensions are required because different
layers have different active sets. CUDA graph capture is explicitly
disabled on both models: CPU execution is not supported by existing
whole-session CUDA capture.

### Measured memory and execution cost

Physical GPU6, one warmup and three measured requests per model,
**exactly 8192 input / 32 output tokens**, width-two MTP, greedy,
GEMM autotune and CUDA graphs disabled on both:

| Variant | Sampled peak GPU GiB | Sampled peak host RSS GiB | Aggregate E2E TPS | Measured request seconds |
|---|---:|---:|---:|---|
| Original NVFP4 | 99.31 | 1.92 | 20.49 | 1.565 / 1.570 / 1.550 |
| CPU-stage NVFP4 | 39.77 | 59.31 | 0.85 | 37.808 / 36.490 / 39.050 |

The prototype saved **59.54 GiB**, about **59.95%** of sampled peak GPU
memory, at roughly **24.2x slower** E2E execution in this test.
GPU memory and process RSS were sampled every 0.5 seconds across load,
warmup and execution. Host RSS is not the complete system page-cache
footprint. Concurrent quality work on GPU5 shared host resources, so
these timings are feasibility evidence, not isolated production numbers.
They must not be compared directly to the earlier graph-enabled
8K/512 ~100 TPS benchmark.

Four short known-answer probes passed 4/4 in both variants and showed
sampled GPU peaks of 95.25 GiB versus 27.75 GiB. These short probes
underestimate the peak for the full 8K workload.

### Quality and outcome

Fresh matched graph-disabled controls on GPU5 scored **58/64** in both
models on the existing frozen64 suite, with **zero baseline-correct
regressions**, no MTP failures and no empty/repetitive outputs across
128 requests. One already-incorrect arithmetic output had different
tokens. This is a single paired custom-suite screen, not a new heldout
benchmark or universal accuracy guarantee.

**Feasible for GPU-memory reduction, not deployment-ready.** The
uncached per-invocation host gather/transfer incurs substantial latency.
A next implementation should investigate a persistent bounded GPU
expert cache, reusable pinned staging and asynchronous transfers, with
prefill handling and graph integration qualified separately. No such
cache or native runtime integration is claimed here.

The independent candidate is
`/var/tmp/asonawane-qwen38-nvfp4-offload-engine`; its
`offload_manifest.json` records the experimental status.
Local evidence is `nvfp4_offload_summary.json`,
`nvfp4_offload_8k_memory.json`, `nvfp4_offload_layer_both.json`,
`nvfp4_expert_stage_synthetic.json`,
`quality_nvfp4_offload_frozen.json` and
`quality_nvfp4_offload_control_frozen.json`.
The staging library, builder, tests and benchmark scripts remain
session artifacts; this is not an upstream production feature.

## Persistent 128-expert GPU cache and batched staging

Continued the NVFP4 offloading prototype with **128 persistent expert
slots per target layer**, reusable pinned host staging for sixteen
experts, and CUDA-side outputs. CPU-side exact weights remain unchanged.
Cache hits reuse packed GPU weight records; misses upload packed
records. All expert computation still runs in the existing CUDA QMoE.
The cache never removes experts from routing and does not requantize.

Three variants were screened at the same 128-slot budget:

| Prototype | Sampled peak GPU GiB | 8K/32 E2E TPS |
|---|---:|---:|
| Original uncached host stage | 39.77 | 0.85 |
| LRU cache with individual device copies | 55.71 | 2.39 |
| Frequency-admitted cache | 55.80 | 2.23 |
| LRU cache with batched CUDA scatter, first run | 57.87 | 2.91 |
| LRU cache with batched CUDA scatter, repeat | 57.87 | 2.57 |

Frequency admission was rejected: it reduced measured hit rate rather
than improving it for this workload. Batched scatter replaces many
per-expert/per-field device copies with one contiguous H2D transfer and
one scatter kernel per batch. It has additional reusable device staging
memory, explaining the higher peak than individual-copy LRU.
Synchronization remains at each staging batch and invocation: this is
not a claim of asynchronous overlap with future layer computation.

The final repeat measured original NVFP4 immediately before cached
NVFP4 on GPU6, without concurrent quality work:

| Variant | Peak GPU GiB | Peak host RSS GiB | Aggregate E2E TPS |
|---|---:|---:|---:|
| Original NVFP4, matched graph-disabled control | 99.31 | 1.92 | 19.70 |
| Selected batched LRU cache | 57.87 | 57.20 | 2.57 |

Same exact **8192 input / 32 output**, width-two MTP, one warmup and
three measured requests, greedy, CUDA graphs/GEMM autotune disabled.
Measured cached request times were 12.485/12.369/12.448 seconds.
The selected cache saves **41.44 GiB** of sampled peak GPU memory and
is about **3.04x faster than the original uncached prototype**, but
remains **7.65x slower than its matched GPU-resident control**.
The earlier 2.91 TPS run demonstrates variability; 2.57 TPS is the final
paired-repeat result, not a guaranteed performance floor.

Persistent cache records total about 15.86 GiB across all 48 layers,
in addition to compact outputs, device staging, Engram, MTP and execution
workspaces. The 128-slot budget bounds cache residency, not total GPU
allocator high-water marks. Cache statistics record hits, misses and
actual uploaded bytes; broad prefill still selects many experts outside
the cache. A 128-slot cache cannot eliminate those transfers.

Synthetic byte-order, scale/bias, changing-set and eviction tests passed,
including explicit invalid-index rejection. Six actual-weight layer0
CUDA cases (including repeated decode after 128-row execution) matched
the GPU-resident reference within maximum absolute error `7.6294e-6`.
The cache/scatter library builds with strict C++ warnings and SM90 CUDA.
Weight files were independently hash-verified unchanged against the
already-verified offload package.

On existing frozen64, the selected cache and matched graph-disabled
NVFP4 control both score **58/64**, with **zero regressions and identical
generated token outputs for all 64 cases**. There were no MTP failures
or empty/repetitive answers. This remains one existing-suite screen,
not new heldout general-quality certification.

The separate candidate
`/var/tmp/asonawane-qwen38-nvfp4-cache128-engine` now selects
`libnvfp4_expert_cache_scatter.so` in its configuration, with an explicit
experimental `cache_manifest.json`. Other libraries retain the earlier
LRU and rejected frequency candidates for reproducibility.
Original model/runtime and uncached candidate remain unchanged.
Evidence is `nvfp4_cache_summary.json`,
`nvfp4_cache_scatter_repeat_8k_memory.json`,
`quality_nvfp4_cache_scatter_frozen.json`,
`nvfp4_cache_scatter_layer_both.json` and cache statistics in the
corresponding benchmark logs.

**Still not production-ready.** Next work should focus on reducing
prefill transfer demand and enabling useful transfer/compute overlap,
or evaluate a larger GPU-resident expert budget with user approval.
Graph capture, concurrent invocation stress, other GPUs and the exact
8K/512 production benchmark are not qualified by these tests.

## GPU-resident memory reduction screens

The new target is **below 80 GiB sampled peak GPU memory while retaining
approximately 130 decode tokens/s**, with the original 8,192-input /
512-output prompt, width-two MTP and unchanged NVFP4 weights. These
screens ran on otherwise idle H200 GPU 6. The initial screens below did
not meet the memory target; the later chunk-eight candidate is recorded
separately. The original model/runtime remains unchanged.

| Configuration | Sampled GPU peak (GiB) | Decode TPS | E2E TPS |
|---|---:|---:|---:|
| Fresh resident control | 100.09 | 135.63 | 106.00 |
| Requested exact-size arena + MoE row tile 128 | 100.09 | 128.18 | 37.53 |
| Scheduler limit 1,024 tokens | 127.29 | 127.63 | 82.04 |
| Additional row-major scale-copy bypass | 100.09 | 127.31 | 100.30 |
| Requested 6 GiB exact-size decoder arena | 100.09 | 129.44 | 101.08 |
| No memory patterns + explicit 40-block cache | **97.11** | **127.71** | **101.03** |
| Same 40-block configuration, GEMM tuning disabled | 97.05 | 119.45 | 95.49 |

Timings in this table are from memory-instrumented runs, not a stable
performance ranking. Most runs used one warmup and three measured
requests; the requested 6 GiB arena diagnostic used one measured request.
Cold loading, prefill, tuning, graph capture and decode are included in
the sampled memory peak.

### Two audit corrections

The extra scale-copy condition was redundant: CUDA QMoE already returns
early from `PrePack` for row-major NVFP4 block scales on the dequant
fallback path. The isolated rebuilt provider passed all **22 existing
offline row-major NVFP4 tests**, but neither loaded memory nor peak memory
fell. Removed the redundant source conditions; there is no new
scale-elimination implementation or 7 GiB saving to claim.

The fine-grained arena screens did **not** establish that exact-size
allocation is ineffective. GenAI supplies `default_memory_arena_cfg`
using `UpdateCUDAProviderOptionsWithValue`, but the current ORT
implementation handles only `user_compute_stream` and silently ignores
the arena pointer. Consequently the requested `max_mem` and numeric
`arena_extend_strategy` settings in those screens were not applied.
The tiling and scheduler settings were independent of that pointer.
This needs an API/configuration fix and a new allocation audit before
fine-grained arena results can be interpreted.

A separate diagnostic used the standard CUDA `gpu_mem_limit` setting
instead. Verbose arena logs confirmed an **80 GiB per-session arena
limit**, but the exact workload failed during warmup prefill at layer 1
QMoE requesting another **4,124,476,672 bytes**. This limit is not a
device-wide cap: the GenAI global allocator and auxiliary sessions have
separate pools. The global CUDA arena used default settings and grew to
11 GiB in this diagnostic. A successful per-session cap alone would not
prove a below-80-GiB total peak.

### Limited successful reduction, not target attainment

Disabling memory patterns in decoder/MTP and explicitly selecting 40
paged-cache blocks reduced the observed peak by **2.98 GiB**, to
**97.11 GiB**. These settings were screened together; no individual
attribution is established. The original exact prompt hash, 512 output
tokens and zero MTP failures were checked for every benchmark request.
The fixed cache is a workload-specific screen, not a general replacement
for the model's 262K context capacity or arbitrary request batching.

Separate unsampled three-request throughput runs measured:

| Configuration | Decode TPS | E2E TPS |
|---|---:|---:|
| Adjacent unchanged resident control | 127.20 | 92.70 |
| No memory patterns + 40 blocks | 126.60 | 100.54 |

Decode throughput differed by less than 0.5% in this small paired screen;
it does not establish a guaranteed 130 TPS rate. Fresh paired frozen-64
quality runs scored **58/64 each**, lost no control-correct answers, and
produced **identical token sequences on all 64 requests**, with no MTP,
empty-output or repetition failures. This is limited observed accuracy
evidence, not universal preservation or a new untouched held-out suite.
The long 8K/512 throughput runs were not token-identical between
configurations; each configuration also produced four distinct output
hashes across its own four runs. Consequently the short-suite parity
must not be generalized to deterministic long-context output parity.
Disabling GEMM tuning did not meaningfully improve peak memory and
slowed decode, so that variant is rejected.

The 97.11 GiB run targeted 20 ms sampling and had a maximum gap of
215 ms. The redundant-scale run had a maximum gap of about 2.04 seconds.
All GPU peaks remain sampled device-wide observations, not exact
allocator high-water marks. The sampler now reports the selected GPU
and records the optional row-tiling flag rather than hardcoding GPU 0.

Evidence remains in local session artifacts:
`gpu_resident_memory_{baseline,no_scale_copy,arena_audit,standard_cap,no_pattern,no_tune}_*`,
`gpu_resident_{control,no_pattern}_throughput.json`, and
`quality_resident_{control,no_pattern}_frozen.json`.
The original model configuration and retained runtimes are untouched.
These initial experiments did not satisfy the below-80-GiB goal.

### Corrected arenas and selected-logits continuation

Fixed three tightly coupled runtime issues before continuing:

- CUDA pointer options now set/get `default_memory_arena_cfg`; unsupported
  pointer updates fail explicitly instead of silently returning success.
- The CUDA provider factory copies the arena configuration, preserving
  its lifetime after GenAI releases the temporary configuration object.
- GenAI forwards arena controls to its global CUDA allocator init session
  and verifies the pointer round-trip, reporting an explicit error with
  an older incompatible ORT library.

Actually applying same-as-requested allocation with zero tolerated dead
bytes exposed a BFC allocator bug: an exactly sized allocation was split
into an allocation and a zero-byte chunk, causing a `RegionFor` failure
during model loading. The split now requires a nonempty remainder.
All **15 BFC arena tests**, **3 focused CUDA C API tests** (including
configuration lifetime), and **22 offline row-major NVFP4 tests** passed.
ORT core/CUDA and GenAI core/CUDA builds passed. These runtime changes
are uncommitted; no installed or retained baseline runtime was replaced.

Created a separate selected-logits model. Both main and MTP graphs use
the existing `logits_indices` contract: a Gather before the LM head
computes only the requested logits rows, including every MTP verification
row. Hidden-state outputs and expert weights remain unchanged. This is
not removal of MTP, expert pruning, quantization or offloading.
Independently verified **132,024,168,449 bytes in 10 unchanged files**
against the original package, including external weights and tokenizer.
Only main/MTP graph structure and candidate configuration changed.

| Configuration | Sampled GPU peak (GiB) | Decode TPS | E2E TPS |
|---|---:|---:|---:|
| Corrected arenas, original full-logits graphs | 95.15 | 129.52 | 102.42 |
| Corrected arenas, row tile 32 | 93.21 | 124.67 | 19.16 |
| Corrected arenas, selected-logits graphs | 91.37 | 128.93 | 102.37 |
| Selected logits, deterministic MoE tactics | 87.70 | 121.68 | 94.21 |
| Selected logits, deterministic MoE + row tile 8 | 85.36 | 54.78 | 12.38 |
| Selected logits, scheduler chunk 256 | 88.68 | 127.03 | 51.67 |
| Selected logits, scheduler chunk 32 | 84.34 | 128.48 | 15.13 |
| Selected logits, scheduler chunk 8, first screen | **79.44** | **129.11** | **17.78** |
| Selected logits, scheduler chunk 8, three-request repeat | **79.44** | **134.88** | **18.12** |

All configurations retain the full 8,192-token input, 512-token output,
width-two MTP and CUDA graphs. Scheduling chunks changes how prefill
is executed, not the request's context or output length. Exact-size
arenas, disabled memory patterns and a workload-specific 40-block cache
are shared by these screens. Deterministic-tactic and row-tiling variants
are not selected. Table timings are memory-instrumented; one measured
request was used for row-tile 8, scheduler chunk 32 and the first
chunk-eight screen, and three for other rows.

**The chunk-eight candidate satisfies the observed memory/decode targets,
but has a severe prefill latency tradeoff.** An independent five-request
run without the sampler measured **134.40 decode TPS**, **18.10 E2E TPS**
and **24.48 seconds mean TTFT**. A subsequent three-request run using
the persisted candidate configuration with **no overlay** measured
**131.95 decode TPS**, **17.95 E2E TPS** and **24.65 seconds mean TTFT**.
The unchanged control's independent five-request run measured
129.71 decode TPS, 102.63 E2E TPS and 1.05 seconds mean TTFT.
This is not a replacement for the original approximately 100 E2E TPS
configuration if prefill latency/end-to-end throughput must be retained.
The unchunked selected-logits variant measured 128.82 decode/102.34 E2E
TPS independently, but still used 91.37 GiB peak.

Fresh paired quality runs for chunk eight:

| Existing suite | Control correct | Candidate correct | Lost control-correct |
|---|---:|---:|---:|
| Frozen 64 | 58 | 58 | 0 |
| Separate 32 | 26 | 26 | 0 |
| Earlier INT3 validation 32 | 26 | 26 | 0 |

All 128 requests preserved every fresh control-correct answer. Frozen64
and earlier-INT3-32 token streams matched exactly; three separate32
requests had token differences without answer regressions. No MTP,
empty-output or repetition failures were observed. These are existing
observed suites, not new untouched held-out data or universal accuracy
proof. Long benchmark outputs are not required to be bit-identical.

The two chunk-eight memory runs observed the same peak:
**85,301,460,992 bytes = 79.443 GiB**, leaving only **0.557 GiB**
below 80 GiB. The three-request repeat's maximum sampling gap was
131 ms despite a 20 ms target. These are device-wide sampled peaks
through loading, warmup/prefill/capture and decode, **not a hard allocator
cap or proof that every transient stays below 80 GiB**. An actual 80 GiB
device, larger batch/context, different prompt, concurrent workload or
other GPU remains unqualified.

The experimental candidate is persisted at
`/var/tmp/asonawane-qwen38-selected-logits-engine`, including chunk-eight
settings in its own `genai_config.json`. It requires the isolated
`nvfp4-arena-python` runtime, not the previous installed runtime.
The baseline package/configuration is untouched. Local evidence:
`gpu_resident_memory_selected_chunk8{,_repeat}_{results,benchmark}.json`,
`selected_chunk8{,_persistent}_throughput.json`,
`quality_selected_chunk8{,_control}_{frozen,separate,fresh32}.json`,
`selected_logits_weight_hashes.json`, and the arena build/test logs.

## Experimental memory-efficient NVFP4 prefill

Implemented the default-off `ORT_QMOE_NVFP4_PACKED_PREFILL=1` route in
ORT's packed-prefill pipeline. It reads the original row-major NVFP4
weights/scales and dequantizes only a shared-memory tensor-core tile;
it does not materialize full activation-dtype expert weights. Original
weight bytes, CPU Engram, FP8 MTP, width-two speculation and decode GEMV
are unchanged. The selected-logits graph and corrected exact-size
arenas remain necessary for the following measurements.

The first scalar 32x64 tile reached 83.23 GiB / 55.32 sampled E2E TPS.
Increasing it to 64x128 did not improve latency. Nsight Systems then
attributed 4.55 seconds, 85.8% of prefill GPU kernel time, to the packed
GEMMs. Reusing the existing eight-element FP4 decoder, sharing scales
across each vector and padding shared strides substantially reduced
that overhead. Smaller 32x128 and 64x64 tiles were slower and were not
retained. The final kernel uses 64x128x64 tiles, FP16/BF16 operands and
FP32 accumulation, with the existing 256 MiB packed scratch limit.

All screens below retain **8,192 uncached input / 512 generated tokens**,
one excluded warmup and three measured requests. TPS in this table is
from memory-instrumented runs, not independent throughput validation:

| Packed-prefill configuration | Sampled peak GPU memory | Decode TPS | E2E TPS | TTFT |
|---|---:|---:|---:|---:|
| Vectorized, scheduler 8,192 | 83.23 GiB | 125.84 | 92.23 | 1.49 s |
| Vectorized, scheduler 2,048 | 80.38 GiB | 131.11 | 94.26 | 1.53 s |
| Vectorized, scheduler 1,024, initial screen | 79.79 GiB | 133.93 | 90.13 | 1.87 s |
| Final bounded-dispatch build, scheduler 1,024 | 79.79 GiB | 134.10 | 87.46 | 2.04 s |

Two separate five-request, unsampled invocations of the retained final
build measured **87.13 / 85.43 E2E TPS**, **133.43 / 129.40 decode TPS**
and **2.05 / 2.04 seconds TTFT**. The first ran while quality checks used
other GPUs; the second ran without concurrent session quality work.
All warmup and measured requests generated exactly 512 tokens with zero
MTP failures. Do not substitute the initial 90.13 sampled figure for
these independent results.

A fresh three-request flag-off check with the final runtime and normal
8,192-token scheduling measured **103.27 E2E TPS**, **130.40 decode TPS**
and **1.04 seconds TTFT**. It confirms that the fast dense route remains
available; its previously measured selected-logits peak was 91.37 GiB,
not a new memory measurement of this invocation.

Fresh paired flag-off/flag-on controls using the same selected graph,
1,024-token scheduling and final runtime preserved all individually
control-correct answers:

| Existing observed suite | Control correct | Packed correct | Lost correct | Token-output differences |
|---|---:|---:|---:|---:|
| Frozen64 | 58/64 | 58/64 | 0 | 2 |
| Separate32 | 26/32 | 26/32 | 0 | 2 |
| Earlier INT3 validation32 | 26/32 | 26/32 | 0 | 0 |

No empty answers, repetition failures or MTP failures were observed.
These 128 cases are already observed custom suites, not new untouched
held-out data, long-context quality qualification or proof of universal
accuracy equivalence. CUDA provider build, 38 focused NVFP4 tests and
14 shared integer-prefill regression tests passed. Packed graph replay
is tested exactly; dense numerical comparison sessions are uncaptured.

The final sampled device-wide peak is **85,678,948,352 bytes =
79.7947 GiB**, only **0.2053 GiB below 80 GiB**. The largest sample gap
was 127 ms despite a 20 ms target. This does not establish a hard cap,
qualification on an actual 80 GiB GPU, arbitrary prompts/context/batches,
or safe concurrency.

**Outcome:** substantial improvement over chunk-eight's approximately
18 E2E TPS below 80 GiB, but **the combined below-80 / 100-E2E target
is not met**. Keep this experimental path opt-in. No model, installed
runtime or existing persisted chunk-eight configuration was replaced.

The retained isolated runtime is `nvfp4-packed-prefill-final-python`.
Reproduction requires the selected-logits graph, corrected arena
overlay with `engine.dynamic_batching.max_scheduled_tokens=1024`,
40 cache blocks, no decoder/MTP memory patterns, and
`ORT_QMOE_NVFP4_PACKED_PREFILL=1`, in addition to the standard tensor
attention/bounded-indexer flags. Local evidence:
`gpu_resident_memory_packed_prefill_final_scheduler1024_{results,benchmark}.json`,
`packed_prefill_final{,_sequential}_independent.json`,
`quality_packed_prefill_{control,candidate}_*.json`,
`nvfp4_packed_tile64_profile.{nsys-rep,sqlite}`,
and the final build/test logs.

### Follow-up: asynchronous packed-prefill pipeline

Profiled the retained vectorized build with 1,024-token scheduling
before changing it. Nsight Systems attributed **1.147 seconds /
58.2%** of prefill GPU kernel time to 768 packed-GEMM launches.
Sparse tensor attention contributed 0.249 seconds and gated delta net
0.121 seconds. These are kernel totals, not request wall time. Nsight
Compute hardware counters were inaccessible under this account's
permissions; no counter-based occupancy or bandwidth claim is made.

Implemented asynchronous global-to-shared input and packed-weight
prefetch, overlapping the next reduction tile's transfers with current
tile dequantization/tensor-core computation. Activation dtype, FP32
accumulation, original scales/weights and decode GEMV remain unchanged.
A compact layout double buffers only input and packed bytes while
reusing one dequantized weight tile and aliasing the final FP32 epilogue.
It uses 44 KiB dynamic shared memory. Copy waits and block barriers
protect producer/consumer handoff and storage reuse.

The initial double-buffer implementation reduced packed-GEMM kernel
time to **0.915 seconds**. Compact storage reduced it further to
**0.882 seconds**, **23.1% below** the matched original profile.
The other dominant kernel totals were essentially unchanged. K32 and
32x64 tile screens did not improve the measured prefill latency and
were not retained. The final tile remains 64x128x64.

Sequential unsampled five-request runs on physical GPU6, with one
excluded warmup each and identical graph/configuration, measured:

| Metric | Non-pipelined control | Retained pipeline |
|---|---:|---:|
| E2E TPS | 83.54 | **90.52** |
| Decode TPS | 131.29 | **132.03** |
| Mean TTFT | 2.237 s | **1.786 s** |
| Mean request time | 6.129 s | **5.656 s** |

The measured E2E improvement is **8.35%**. It includes ordinary
request/decode variability; the separate kernel profiles support the
prefill improvement mechanism. The control was slower than earlier
85-87 TPS controls, so do not attribute the whole difference from
previous reports to this change.

The final retained memory run observed **85,689,434,112 bytes =
79.8045 GiB**, with a maximum sampling gap of **111 ms**. Its sampled
timings were 87.71 E2E / 126.42 decode TPS and 1.795 seconds TTFT;
independent throughput above is reported separately. An additional
1,280-token scheduler screen observed 79.9568 GiB and only about
44 MiB sampled headroom, so it was not selected. Neither configuration
is a hard memory cap or qualification for actual 80 GiB hardware.

Fresh flag-off/flag-on paired quality runs using the retained provider
preserved all control-correct answers: frozen64 **58/58**, separate32
**26/26**, and earlier INT3 validation32 **26/26**. Two of 128 output
token streams differed, with zero lost correct answers, MTP failures,
empty answers or repetition failures. These remain previously observed
custom suites, not universal/long-context accuracy qualification.

The CUDA provider build and **54 focused/regression tests** passed:
40 NVFP4 tests including long K/N at Qwen dimensions, FP16/BF16 dense
numerical parity, tails, empty experts, graph replay and fallback
routing, plus 14 shared integer-prefill tests.

The opt-in retained runtime is `nvfp4-packed-pipeline-retained-python`,
provider SHA256
`5fb4bc3b2514f7c6bfc523128ab4df3d44d271b57bc0ba1fe0f604e2bd7dc962`.
It requires the existing 1,024-token corrected-arena overlay and
`ORT_QMOE_NVFP4_PACKED_PREFILL=1`. Original runtime/model/configuration
and the earlier candidate remain retained. Local evidence:
`nvfp4_packed_{final,compact}1024_profile.*`,
`gpu_resident_memory_pipeline_retained1024_{results,benchmark}.json`,
`pipeline_retained_{matched_control,independent}.json`,
`quality_pipeline_{control,candidate}_*.json`, and
`nvfp4_packed_pipeline_retained_manifest.json`.

**The combined below-80-GiB / 100-E2E-TPS objective is still unmet.**
At the independently observed 132.03 decode TPS, 100 E2E TPS would
require TTFT near 1.25 seconds, versus 1.79 seconds measured. Further
prefill latency reduction and substantially more memory headroom are
still required; the pipeline is an experimental improvement, not a
production deployment recommendation.

### Follow-up: bounded indexer capacity and larger packed workspace

The experimental candidate already reserves only 40 KV blocks of
256 tokens each, but its decoder and MTP indexer graphs still declared
65,536 compressed-key slots for the original 262K context.
Created a **separate** graph candidate at
`/var/tmp/asonawane-qwen38-bounded-indexer-engine`:

- Changed `PackedSparseAttentionIndexer.state_capacity` from 65,536
  to 2,560 for twelve decoder nodes and one MTP node, and matched
  their input/output key-state shapes.
- Set configured context/search length to 10,240 tokens, consistent
  with the existing 40-block pool. The actual Engine request ceiling
  is **9,217 total tokens**, including input and output, due to cache
  reservations. Request budgets of 9,218 and 10,241 are rejected
  explicitly; a 9,217-token budget is accepted. The exact 8K/512
  workload remains unchanged.
- Retained all initializer protobuf bytes and independently hashed
  every byte of all three external weight files against the source.
  No expert pruning, requantization, offloading, MTP removal or
  altered input/output lengths were introduced.

The decoder ONNX checker passed. The MTP checker reports the same
pre-existing `SimplifiedLayerNormalization` opset22 registration error
on both source and candidate; it is not reported as a checker pass.
The actual ORT runtime validated and executed both graphs successfully.
Weight files are independent regular-file copies because this ONNX
checker rejects symlinks and multi-link files. Other unchanged assets
are linked, and must not be edited through the candidate.

Additional kernel changes pair round-to-nearest FP32-to-FP16/BF16
conversions, hoist each expert's global scale out of the reduction
loop, and consolidate the redundant end-of-K barrier into the next
iteration's copy-wait barrier. A final barrier still protects the
shared FP32 epilogue. These preserve the original arithmetic format.
The last two changes did not independently establish a throughput
gain in the memory-instrumented screens.

Raised the scratch bound **only for the opt-in NVFP4 route** from
256 to 384 MiB; integer prefill retains 256 MiB. A 4,096-row Qwen
chunk can then complete as one packed MoE pass instead of two,
reducing repeated expert-weight traffic.

All screens retain 8,192 uncached input / 512 generated tokens,
one excluded warmup and three measured requests. These timings are
memory-instrumented, not independent throughput validation:

| Bounded-indexer screen | Sampled peak | E2E TPS | Decode TPS | TTFT |
|---|---:|---:|---:|---:|
| Pipeline, scheduler 2,048, scratch 256 MiB | 79.25 GiB | 96.94 | 133.97 | 1.47 s |
| Paired conversion, scheduler 3,072, scratch 256 MiB | 79.56 GiB | 94.99 | 132.12 | 1.52 s |
| Scheduler 4,096, scratch 256 MiB | 79.67 GiB | 96.94 | 133.16 | 1.44 s |
| Scheduler 3,072, scratch 384 MiB | 79.59 GiB | 97.72 | 132.46 | 1.38 s |
| Scheduler 4,096, scratch 384 MiB | 79.95 GiB | 99.53 | 133.07 | 1.30 s |
| Persisted 4,096 configuration, no overlay | 79.95 GiB | 99.82 | 133.43 | 1.30 s |

Two separate, unsampled five-request invocations of the final
4,096-token candidate measured:

| Metric | Invocation 1 | Invocation 2 | Pooled ten requests |
|---|---:|---:|---:|
| Independent E2E TPS | 99.64 | 99.93 | **99.79** |
| Decode TPS | 133.10 | 133.57 | **133.33** |
| Mean TTFT | 1.299 s | 1.297 s | **1.298 s** |

All warmup and measured requests generated exactly 512 tokens with
zero MTP failures. This is **approximately 100 E2E TPS**, not a strict
at-least-100 result or per-request guarantee.

Fresh paired unbounded/bounded-indexer controls used the same final
provider, flags and 4,096-token scheduler. All 128 output token streams
matched exactly: frozen64 **58/58 correct**, separate32 **26/26**,
and earlier INT3 validation32 **26/26**. All 110 individually
control-correct answers were preserved; zero MTP, empty-answer or
repetition failures occurred. These remain observed custom suites,
not untouched held-out or universal long-context qualification.
The final provider build and **54 numerical/regression tests** passed.

An additional fresh **default-off dense NVFP4** control run also
preserved every individually correct answer: 58/58, 26/26 and 26/26
across the same suites. Three token streams differed from that dense
control without answer regressions. Thus the token-identical comparison
above establishes bounded/unbounded indexer parity with the same packed
provider, not universal bit parity with the original dense route.

Both final memory runs observed **85,840,429,056 bytes =
79.9451 GiB**, leaving only **0.0549 GiB / approximately 56 MiB**
below 80 GiB. The persisted run's maximum sampling gap was 120 ms;
the earlier run's was 126 ms. Device-wide samples can miss shorter
transients. **Do not interpret this as safe deployment on actual
80 GiB hardware**, a hard cap, concurrency qualification, or
support for the original 262K context. The 2,048-token screen has
more sampled headroom, but its independent throughput has not been
validated in this iteration.

The new candidate persists 4,096-token scheduling, width-two MTP,
40 blocks, corrected arenas and no decoder/MTP memory patterns.
It requires `ORT_QMOE_NVFP4_PACKED_PREFILL=1` and the isolated
`nvfp4-packed-pipeline-workspace384-python` runtime; provider SHA256:
`d10df8dc0f2ea92a8fe888770d73b69ca94512271d0c9ae505364b657c742586`.
The original model, selected-logits/chunk-eight candidate, installed
runtimes and earlier retained providers remain unchanged.

Local evidence:
`build_bounded_indexer_candidate.py`,
`build_bounded_indexer_candidate_verified.log`,
`gpu_resident_memory_bounded_indexer_workspace384_4096_{results,benchmark}.json`,
`gpu_resident_memory_bounded_indexer_persistent_{results,benchmark}.json`,
`bounded_indexer_workspace384_independent_{1,2}.json`,
`quality_bounded4096_{control,candidate}_*.json`, and the candidate's
`memory_candidate_manifest.json`.

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
