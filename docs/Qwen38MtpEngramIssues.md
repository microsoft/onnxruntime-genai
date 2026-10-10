# Qwen3.8 Flash Next: MTP and Engram Issues

For portable build and benchmark commands, see the
[Qwen3.8 reproduction README](../benchmark/python/qwen38/README.md).
The [results table](../benchmark/python/qwen38/RESULTS.md) summarizes the
measured gains. Machine-specific paths below describe the original local
experiments, not prerequisites for the portable benchmark.

## Summary

The Qwen3.8 Flash Next model can be exported and executed on CUDA. The corrected one-draft MTP path is lossless and avoids target replay after rejected drafts. Two performance limitations remain:

1. The MTP output projection can dominate coding and long-context workloads.
2. The 48 GB Engram lookup runs on CPU inside the main decoder session, preventing CUDA graph capture for the target model.

## Implementation status

The following changes are now implemented:

- multimodal pipeline snapshot and rewind delegation,
- PLE token/convolution state snapshots,
- shared indexer logical rewind,
- graph-metadata-based MTP hidden-state sizing,
- correct Qwen3.8 MTP indexer and sequence-length bindings,
- standalone `engram.onnx` export,
- `model.engram` configuration parsing,
- CPU Engram token-history state,
- explicit CPU-to-CUDA Engram embedding staging,
- decoder `engram_embeddings` input,
- chunked prefill support for the staged Engram input.
- isolated Q/gate projection selection during MTP export,
- compact first-token GatedDeltaNet transition capture,
- two-slot convolution and PLE convolution state,
- accepted-prefix Engram token-history commit,
- rejection handling that commits verify row zero without replaying the target.

Validation completed:

- the migrated split model is token-identical to the original model with CUDA graph capture disabled,
- greedy baseline and greedy MTP outputs are token-identical on the short validation path,
- the split target session loads with CUDA graph capture enabled,
- focused Python exporter tests and native MTP configuration tests pass.
- corrected FP16 MTP logits have cosine similarity approximately 0.999989 to PyTorch with the same top-1 token,
- corrected INT4 MTP has the same top-1 token as PyTorch,
- a 64-token natural-language test improved from 40.0 to 84.8 tok/s (2.12x) at 70.3% acceptance while remaining token-identical,
- the full coding-context sweep is not universally token-identical because sequence-2 target verification can choose a different argmax from two sequential target forwards; non-identical rows are not valid speed measurements.

Remaining limitation:

- repeated target CUDA graph replay still encounters an ORT CUDA host-copy/capture-stream conflict after the first captured decode step. The split itself is functional, but graph replay needs a dedicated stream-safe staging/capture integration fix.

### Baseline decision

Experimental hybrid n-gram drafting, logits-free MTP state advancement, and adaptive fallback were evaluated before direct prefix commit was available and reverted. Direct prefix commit is now the active optimization because it preserves the target token sequence and removes the extra 48-layer target forward on every rejection.

The active split model package therefore defaults to ordinary target decoding. MTP correctness fixes remain in the source so the feature can be revisited without reintroducing state-alignment bugs.

### Baseline optimizations

The active baseline path includes:

- removal of redundant per-token Engram CUDA synchronizations,
- fixed 1K chunked prefill in the production model package,
- optional `adaptive_chunking` support (512 tokens through 32K, then 1K), disabled by default because different chunk sizes are not token-identical for this recurrent model,
- chunk-streamed Engram embeddings with asynchronous preparation of the next CPU chunk,
- an optional bounded decode-time Engram result cache,
- eight CPU intra-op threads for the Engram session.

Measured improvements for the active fixed-1K package:

| Context | Before | After |
|---:|---:|---:|
| 1K decode | 33.2 tok/s | 38.7 tok/s |
| 216K decode | 32.8 tok/s | 34.9 tok/s |
| 216K prefill | 1,326 tok/s | 1,331 tok/s |

CUDA graph capture remains blocked by `GroupQueryAttention` input 6 (`total_sequence_length`), which is registered as `OrtMemTypeCPUInput`. Even when every graph node is assigned to CUDA, ORT inserts a `MemcpyToHost` for this dynamic host scalar.

Relevant upstream work:

- ONNX Runtime PR #32718 prototypes partitioned CUDA graph capture with eager CPU partitions and device copies.
- ONNX Runtime PR #32071 integrates CUDA kernel workspace with activation memory patterns.
- ONNX Runtime GenAI PR #2597 (merged) implements CPU embedding offload with pinned host staging and persistent CUDA buffers for Engine models.
- ONNX Runtime GenAI PRs #2605 and #2616 improve cross-device multimodal buffer ownership and feature copies.

## Current model configuration

### Separate NVFP4 Engine benchmark

The local `qwen_38_flash_nvfp4_engine` package differs from the INT4 export
described below. Its target graph uses NVFP4 QMoE, and its Engram table is in a
separate CPU session. CUDA graphs were verified for this Engine configuration.
Its MTP graph uses FP8 QMoE and cannot initialize with an ORT build configured
with `onnxruntime_USE_FP8_QMOE=OFF`; the initial measurements below disable MTP.

FP8 QMoE support is present in upstream commit
`51f986229cad6d2a176d7e1a7383fd6e920dad71` (PR #32887). The local ORT branch
already contains its development series and additional fused FP8 kernels, so
the upstream squash was not cherry-picked. Reconfiguring the existing build
with `onnxruntime_USE_FP8_QMOE=ON` and rebuilding the CUDA provider succeeded.
The isolated FP8 provider's SHA256 is
`416a90c383626d2b3b0aacdaa396b065559b31eab0f1488d77a8083e9e935aff`;
installed conda packages remain unchanged.

MTP model and Engine initialization succeeded after the rebuild. Initially, the benchmark's
`min_generated_tokens=512` prevents drafting: Engine's
`Request::DraftTokenValidationError` rejects speculation until the turn passes
its minimum generated-token count. That run completed 512 target passes but
proposed zero drafts, and the benchmark rejected it. Removing the minimum
while preserving the original EOS settings ended the first turn after only
23 tokens, so that run was also rejected. An EOS sentinel of -1 was rejected
by GenAI's vocabulary validation and is not retained. Neither rejected run is
included in the throughput table.

GenAI was then updated to allow speculation below the turn's minimum generated
token count. Before target argmax or top-k selection, each verification row,
including the correction/bonus row, applies the EOS floor at its own logical
sequence position rather than the sequence length including all staged drafts.
CPU and CUDA search accept this explicit logical length; ordinary generation
retains the existing default. Repetition-penalty, no-repeat-ngram, and guidance
restrictions remain unchanged. Three focused Engine tests passed, including
greedy and sampled cases crossing the minimum and a bonus row below the floor.

The rebuilt GenAI libraries were staged alongside the FP8 ORT provider in an
isolated Python package. The exact 8,192-input / 512-output MTP benchmark then
passed with the original EOS settings and `min_generated_tokens=512`. Loaded
ORT and GenAI library paths were verified. The draft width was seven, with
1,006 accepted tokens out of 3,677 proposed across 527 measured rounds:
27.36% of all proposals, or 67.29% of the 1,495 evaluated proposals. Later
drafts after the first rejection are not evaluated. Mean accepted drafts per
round were 1.91; target forward passes fell from 1,536 in the three baseline
requests to 530. There were zero MTP failures.

The same rebuilt runtime without MTP measured 74.20 decode TPS and 54.57
end-to-end TPS. MTP measured 80.70 decode TPS and 57.93 end-to-end TPS:
8.77% and 6.16% improvements respectively. This does not meet the 100
end-to-end TPS target. The seven-token chain performs substantially more
draft work than the accepted prefix warrants; width-one measurements follow
below, with widths two through four still untested. Full-model lossless output equivalence remains
unverified because unchanged baseline runs also exhibit greedy nondeterminism.

The benchmark uses one H200, batch size 1, exactly 8,192 input tokens and 512
output tokens, greedy decoding, one warmup and three measured requests without
prefix-cache reuse. Model loading and warmup are excluded. End-to-end TPS
includes uncached prefill and request completion; decode TPS counts the 511
tokens after the first token.

| Configuration | Decode TPS | End-to-end TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Original, CUDA graphs off, 2K prefill budget | 56.76 | 30.89 | 7.57 | 16.57 |
| CUDA graphs on, 8K prefill budget, cache utilization 0.05 | 65.21 | 38.92 | 5.32 | 13.16 |
| Plus vectorized row-major NVFP4 dequantization and sparse softmax reductions | 66.11 | 41.14 | 4.72 | 12.45 |
| Plus head-size-256 specialization alone | 65.70 | 40.96 | 4.72 | 12.50 |
| Plus paired-channel sparse value accumulation and head-size-256 specialization | 66.38 | 43.18 | 4.16 | 11.86 |
| Plus grouped tiled sparse-attention prefill | 66.21 | 50.11 | 2.50 | 10.22 |
| Grouped tiled prefill, independent unprofiled repeat | 66.58 | 50.34 | 2.50 | 10.17 |
| Grouped tiled prefill and split decode | 70.65 | 52.66 | 2.49 | 9.72 |
| Grouped split decode, independent unprofiled repeat | 70.66 | 52.65 | 2.49 | 9.72 |
| Plus 32-block hierarchical indexer tiles and warp sorting | 73.27 | 54.09 | 2.49 | 9.47 |
| Indexer change, independent unprofiled repeat | 73.28 | 54.08 | 2.49 | 9.47 |
| Plus padding-aware hierarchical merges | 74.17 | 54.58 | 2.49 | 9.38 |
| Padding-aware merges, independent unprofiled repeat | 74.10 | 54.53 | 2.49 | 9.39 |
| FP8 enabled and GenAI EOS-floor fix, MTP disabled | 74.20 | 54.57 | 2.50 | 9.38 |
| Same runtime, MTP enabled, draft width 7 | 80.70 | 57.93 | 2.51 | 8.84 |
| Same runtime, MTP enabled, draft width 1 | 90.34 | 62.80 | 2.50 | 8.15 |
| Draft width 1, independent unprofiled repeat | 102.43 | 68.39 | 2.50 | 7.49 |
| Draft width 1, short-query attention split tuning, stable repeat | 103.92 | 69.06 | 2.50 | 7.41 |
| Plus bounded hierarchical indexer scoring | 107.87 | 70.79 | 2.50 | 7.23 |
| Bounded indexer scoring, independent unprofiled repeat | 107.96 | 70.81 | 2.50 | 7.23 |

Reducing `speculative.max_draft_tokens` to one improved throughput. Both
invocations retained one warmup and three measured requests, exact 8,192/512
token counts, the original EOS/minimum settings, and the same isolated runtime.
One draft per round and zero MTP failures were verified. The first invocation
had substantial measured variability (73.97-104.43 decode TPS); its slow first
measured request is retained in the aggregate, not discarded. The independent
repeat ranged from 100.05 to 104.09 decode TPS and averaged 68.39 end-to-end
TPS, 18.06% above width seven and 25.34% above the same-runtime non-MTP
baseline. Draft acceptance was 78.23% in the first invocation and 80.24% in
the repeat. The reason for the first invocation's slow request was not profiled.
Width one crosses 100 decode TPS in the repeat, not 100 end-to-end TPS:
prefill still costs about 2.50 seconds, and total request time is 7.49 seconds
versus the 5.12-second target.

#### MTP runtime overhead and vocabulary-projection experiments

A graph-node-aware Nsight decode capture with width one recorded 188
`cudaStreamSynchronize` calls, 56 graph launches, and 400 `cudaMallocAsync` /
`cudaFreeAsync` calls. Synchronization accounted for 63.8% of CUDA API time,
but that includes waits for GPU execution and is not a removable latency
budget. Captured kernel totals included approximately 25.16 ms across 88
large vocabulary-projection-shaped GEMMs; those were not independently
attributed to target versus head. Instrumented wall timings are excluded.

Two runtime changes were tested together: use the existing reusable device
draft buffers and asynchronous head execution for width one, and bind contiguous
target hidden-state rows directly instead of allocating/copying a packed tensor.
CPU and noncontiguous inputs retained their fallback. This measured 67.98
end-to-end TPS, not an improvement over the previous 68.39. Both runtime
changes were removed; the final Engine drafting implementation is unchanged.

Selected-logits graph variants insert Gather before each target/head LM-head
MatMul. Target prefill projects the final row rather than all 8,192 rows;
target verification retains all required verification rows, while the head
projects only the last row per request. Optional
`model.mtp.inputs.logits_indices` configuration was added with parser and
projection tests, without inheriting the target's binding into a head that
does not declare it.

The two small opt-in graphs, `model.selected_logits.onnx` and
`mtp.selected_logits.onnx`, are alongside the original external weights.
Original graphs, configuration, and weights are unchanged; an overlay selects
the variants. An initial separate-directory experiment was rejected by external
data path validation and was abandoned, not bypassed. The target variant passes
ONNX checker. The full checker does not support the original MTP graph's
SimplifiedLayerNormalization schema, so the added Gather was schema-checked
and the variant validated by ORT loading and execution. Reversing only the
selection edits reproduces each original graph byte-for-byte, including its
initializer metadata.

| Experiment | Decode TPS | End-to-end TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Previous width-one independent repeat | 102.43 | 68.39 | 2.497 | 7.486 |
| Reusable draft buffers / asynchronous head / hidden-state views | 101.59 | 67.98 | 2.501 | 7.532 |
| Runtime experiment plus selected logits | 102.02 | 68.34 | 2.483 | 7.492 |
| Selected logits alone, runtime experiment removed | 100.86 | 67.84 | 2.481 | 7.548 |

Every experiment used one warmup and three measured requests, with exact
8,192/512 counts, one draft per round, zero MTP failures, and verified isolated
ORT/GenAI library paths. The final rebuilt runtime passed 29 focused
MTP lifecycle/config/selected-logits tests. Selected logits consistently saved
about 14-16 ms of TTFT but did not demonstrate an end-to-end gain; decode
variation remains confounded by the existing full-model greedy nondeterminism.
The optional graph variants/configuration support remain available for
experiments, but are not enabled in the original model configuration.
These experiments do not improve the best demonstrated 68.39 end-to-end TPS.

#### Short-query sparse-attention split tuning

Width-one MTP target verification processes two packed rows. The existing
grouped-attention occupancy heuristic selected 22 splits on H200, compared
with 32 for a single-token decode. For grouped queries of at most eight
tokens, the occupancy target now uses four SM waves instead of two, retaining
the 32-split and candidate-length caps. Longer prefill queries and fallback
attention retain their existing heuristic; scratch allocation and launch
continue to use the same split count.

The CUDA provider rebuilt successfully. Standalone numerical-reference
validation passed 128 cases: FP16/BF16, 1/2/4/8 query tokens, 24 query heads
and two KV heads, 2,051 selected entries, causal/noncausal attention, softcap,
head sinks, and slot mappings. Additional C++ reference coverage for 2/4/8
query tokens and 2,048 cached positions was added, but not executed in this
iteration.

| Measurement | Decode TPS | End-to-end TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Split tuning, first invocation | 93.11 | 64.14 | 2.495 | 7.983 |
| Fresh control, original split heuristic | 90.84 | 63.06 | 2.493 | 8.119 |
| Split tuning, independent repeat after control | 103.92 | 69.06 | 2.496 | 7.414 |

The first candidate invocation contained a 76.07 decode-TPS request; the
fresh control contained a 74.67 decode-TPS request. Both are retained.
The candidate repeat ranged from 102.19 to 105.14 decode TPS. Compared
with the earlier stable width-one repeat (68.39 end-to-end TPS), this is
about a 1% E2E gain, not the much larger gain implied by comparing only
against the slow fresh control. Prompt/token counts, zero MTP failures,
one draft per round, and isolated provider paths were checked.

Graph-node-aware captures confirmed two-token attention grids changed from
`(6, 2, 22)` to `(6, 2, 32)`. Average grouped-attention duration decreased
from 160.61 to 114.19 microseconds (about 29%). Captures contain different
launch counts (552 versus 562 for two-token attention), so these are
per-launch averages rather than an equal-work total-time comparison.
One-token grids remain `(6, 1, 32)`. Instrumented wall timings are not
throughput measurements. The retained provider SHA256 is
`0bea036b4ac63db93fc75612adb2effeb6824ee8a356ba1d5c1bfd48b743ac1a`;
installed packages and original model files remain unchanged.

#### Bounded hierarchical indexer scoring

The hierarchical QSA path already covers two-token MTP verification, but its
scoring launch was sized for the full 65,536-block state capacity. With around
2,100 live blocks, most of its 4,096 two-row CTAs only wrote 32 padding keys
despite launching 1,024 threads each.

Scoring now launches at most 128 CTAs per query row. All threads cooperatively
initialize the fully padded suffix, and CTAs score only causally visible tiles
using a grid-stride loop. The last partially visible tile still explicitly
pads its inactive entries. Metadata is evaluated once per CTA, and each warp
reuses its query vector across tiles. Scoring arithmetic, stable sorting,
hierarchical merge layout, workspace sizing, overflow rejection and output
ordering are unchanged. This applies only to the existing hierarchical path
(at most 64 packed query rows); ragged prefill and other shapes are unchanged.

The provider rebuilt successfully. Standalone A/B validation passed 84 packed
cases across FP16/FP32/BF16, plus the existing eight two-request cases. All
selection indices/counts and state/buffer/length outputs matched bitwise.
Coverage includes 1/2/4/8/64 packed rows, compression-boundary state updates,
stable ties, zero live blocks, tile/merge boundaries, capacities 2,051 and
65,536, multiple grid-stride iterations, almost-full state and overflow
rejection. BF16 cases use explicit input/output casts to accommodate Python's
tensor interface. A persistent C++ FP16/BF16 large-capacity regression was
added but not compiled/run: the configured unit-test target would rebuild
825 C++ objects, so this iteration used the isolated numerical checks and
exact Engine benchmarks instead.

| Measurement | Decode TPS | End-to-end TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Fresh control, previous attention-split provider | 92.74 | 63.94 | 2.497 | 8.007 |
| Bounded scoring, first invocation after control | 107.87 | 70.79 | 2.495 | 7.232 |
| Bounded scoring, independent repeat before control | 107.96 | 70.81 | 2.497 | 7.230 |
| Fresh control, reverse-order repeat | 91.96 | 63.59 | 2.495 | 8.051 |

Each invocation retained one warmup and all three measured 8,192-input/
512-output requests with width-one MTP, greedy search, original graphs,
zero MTP failures and verified isolated ORT/GenAI libraries. Both fresh
controls contained slow requests (76.62 and 75.03 decode TPS); neither was
discarded. The two candidate invocations ranged from 105.98 to 109.18 and
106.56 to 108.67 decode TPS. Compared with the previous stable attention-split
repeat (103.92 decode / 69.06 E2E TPS), the new repeat is about 3.9% faster
in decode and 2.5% faster end-to-end, saving about 0.18 seconds. Do not
interpret the outlier-affected control aggregates as a clean 11% gain.
Full-model greedy outputs remain nondeterministic. Candidate draft acceptance
was 676/855 and 675/857, versus 670/862 and 663/868 for the fresh controls;
acceptance differences also affect throughput.

Graph-node-aware profiling confirmed two-row score grids changed from 4,096
to 256 CTAs, with mean kernel duration decreasing from 49.30 to 6.39
microseconds (about 87%). One-row grids changed from 2,048 to 128 CTAs,
with means decreasing from 24.07 to 3.48 microseconds. Captures contain
different launch counts (562 versus 606 two-row launches); these are
per-launch comparisons, not equal-work total-time comparisons. The retained
build and isolated provider both have SHA256
`ee5ad509e2d27be01d8d55a9a18f7fd8c6a64c136c93ddae888aaf52d8c788fb`.
Installed packages, original model graphs and weights remain unchanged.

#### Dequantization time attribution

Dequantization is not the majority of measured GPU kernel time. The
graph-node-aware width-one decode trace before bounded scoring contained no
standalone dequantization launches. Fused NVFP4 MoE GEMV kernels accounted
for 104.39 of 683.89 milliseconds (15.3%) of summed kernel time; FP8 MoE
GEMV accounted for 15.09 milliseconds (2.2%). These kernels include weight
reads, unpacking/scaling, dot products and other work, so the trace does not
isolate the cost of dequantization within them. Grouped sparse attention
accounted for 9.4%, and hierarchical indexer scoring/merging for 7.2%.

The earlier grouped-tiled prefill trace attributed 157.39 of 2,466.51
milliseconds (6.4%) to standalone NVFP4 dequantization, versus 1,470.17
milliseconds (59.6%) to sparse attention. This is a historical prefill
capture, not a new full-request MTP profile. All percentages refer to summed
GPU kernel durations, not end-to-end wall time or a hardware-counter
breakdown of fused instructions.

#### Latest-runtime prefill profile and MTP width-two sweep

A fresh graph-node-aware first-token capture used the bounded-indexer provider
(`ee5ad509e2d27be01d8d55a9a18f7fd8c6a64c136c93ddae888aaf52d8c788fb`),
original model graphs and width-one MTP. Profiling began immediately before
the measured request's first Engine run and stopped after its first token;
model loading and the warmup request were outside the capture.

| Category | Summed GPU kernel time (ms) | Share of summed kernel time |
|---|---:|---:|
| Sparse attention | 1,465.92 | 59.47% |
| Causal convolution | 221.96 | 9.01% |
| NVFP4 dequantization | 157.31 | 6.38% |
| Chunked gated delta net | 116.95 | 4.74% |
| QSA state update | 48.07 | 1.95% |
| QSA ragged scoring | 27.49 | 1.12% |

Total summed kernel time was 2,464.79 ms. All 12 target prefill attention
launches used grid `(6, 8192, 1)` and averaged 122.16 ms. The capture also
includes first-token MTP-head work, including one single-query attention
launch. Instrumented TTFT (7.65 seconds) is not a throughput result;
unprofiled TTFT remains about 2.50 seconds. This confirms that attention,
not dequantization, remains the main prefill optimization target.

Width two was tested with a separate overlay differing from width one only
in `speculative.max_draft_tokens`. The original model configuration was not
changed. Both widths used the same isolated provider/GenAI binaries,
8,192-token prompt/hash, 512-token minimum/maximum, batch one, greedy search
and CUDA graphs. Each independent invocation had one excluded warmup.

| Unprofiled measurement | Measured requests | Decode TPS | E2E TPS | Mean request time (s) |
|---|---:|---:|---:|---:|
| Width one, first comparison | 3 | 107.97 | 70.84 | 7.228 |
| Width two, first comparison | 3 | 124.72 | 77.62 | 6.596 |
| Width two, reverse-order repeat | 3 | 100.49 | 67.48 | 7.588 |
| Width one, reverse-order repeat | 3 | 106.19 | 70.07 | 7.307 |
| Width one, extended comparison | 5 | 100.27 | 67.44 | 7.592 |
| Width two, extended comparison | 5 | 112.03 | 72.52 | 7.060 |
| Width one, all three invocations pooled | 11 | 103.87 | 69.05 | 7.415 |
| Width two, all three invocations pooled | 11 | 111.63 | 72.34 | 7.078 |

Pooled throughput is total generated tokens divided by total elapsed time,
not an average of invocation TPS. All slow requests are retained: width two
reached as low as 74.27 decode TPS in the reverse repeat and 84.57 in the
extended comparison; width one reached 75.93 in the extended comparison.
Across all requests, width one ranged from 75.93 to 109.98 decode TPS and
width two from 74.27 to 125.82. The first width-two result alone would
overstate the gain. Across 11 requests per width, width two improved E2E
throughput by about 4.8%, saving about 0.34 seconds per request; the cause of
the slow requests and full-model greedy nondeterminism remain unresolved.

| Pooled speculative metric | Width one | Width two |
|---|---:|---:|
| Rounds | 3,141 | 2,447 |
| Proposed drafts | 3,141 | 4,889 |
| Evaluated drafts | 3,141 | 4,269 |
| Accepted drafts | 2,471 | 3,171 |
| Accepted / proposed | 78.67% | 64.86% |
| Accepted / evaluated | 78.67% | 74.28% |
| Accepted drafts per round | 0.787 | 1.296 |
| Target forward passes | 3,161 | 2,461 |
| Draft forward passes | 3,141 | 4,889 |

Width two trades about 56% more draft passes for about 22% fewer target
passes. Some final rounds contain only one proposal because the remaining
token budget reserves a correction/bonus token; this accounts for proposed
drafts being slightly below twice the round count. Both widths completed
every request with zero MTP failures; width-two histograms confirm acceptance
of two drafts in a round. Runtime paths and provider SHA256 were checked.
Width two remains an opt-in benchmark candidate, not a default change.

The next kernel experiment should target prefill sparse attention. The
current CTA shares K/V across four heads but still processes query tokens
independently. Measure selected-index overlap between nearby query rows
before attempting cross-query K/V reuse; causal boundaries and different
selected sets must remain correct. A prefill tensor-core dot-product path
is another candidate if reuse cannot be exploited economically. Neither
has been implemented or validated by this profiling/sweep iteration.
Even the first width-two result (6.60 seconds) remains above the 5.12-second
target; pooled width-two latency is about 7.08 seconds.

#### Prefill reuse investigation and six-head grouping

An isolated diagnostic provider captured selected indices/counts from all
12 target sparse-attention layers for the same 8,192-token prompt. The
temporary capture hook synchronized and copied selections to session
artifacts; it was removed before all performance experiments. Its cold,
instrumented request is not a throughput measurement.

Adjacent query pairs have substantial set overlap, but very little
rank-aligned overlap. For queries 2,048-8,191, eliminating duplicate loads
over a pair's union could theoretically save 34.0-41.8% of selected-row
loads across the layers. However, rank-aligned reuse saves only 0.14-0.36%;
identical 64-row tiles save only about 0.012%. These are load-count upper
bounds, not measured bandwidth or execution-time savings. The first 2,048
queries select the entire causal prefix, in different score orders.

Three prefill experiments were built, numerically validated and benchmarked:

| Experiment | Full-prefill attention GPU time, 12 layers (ms) | Mean unprofiled TTFT (s) | Disposition |
|---|---:|---:|---|
| Existing four-head, 64-row tiles | 1,465.91 | 2.495 | Control |
| Guarded dense-prefix paired queries | 1,467.91, including coverage check | 2.499 | Removed |
| Four-head, 32-row prefill tiles | 1,479.34 | 2.514 | Removed |
| Six-head, 64-row prefill tiles | 1,183.31 | 2.220 | Retained |

The paired-query experiment verified duplicate-free complete causal coverage
before sharing K/V between two queries. Other rows used the original kernel.
Its remaining attention work took 1,317.83 ms, paired attention took
149.95 ms, and coverage checks took 0.13 ms. It passed reference checks
but did not reduce prefill time. Smaller tiles also failed to improve time.
Neither experiment nor the diagnostic capture logic remains in source.

The retained change groups six query heads per CTA for eligible prefill
instead of four, sharing each 64-row K/V tile across six warps. The model's
12:1 GQA ratio allows six heads to remain within one KV head. It uses
192 threads instead of 128, reducing prefill head-group grid X from six
to four. This reuses K/V across more heads, not across query tokens.
Selection order, scoring and softmax arithmetic are unchanged.

Six-head dispatch requires the existing grouped-path eligibility, more than
64 query tokens, one candidate split, a GQA ratio divisible by six, and
sufficient shared memory for six-head logits. All other calls retain
four-head grouped or fallback dispatch, including decode and MTP verification.
The existing split/scratch heuristic is unchanged because the new branch
only executes unsplit prefill. Kernel loads use the actual block size.

Graph-node-aware profiling confirmed 12 launches with grid `(4, 8192, 1)`
and 192-thread CTAs, averaging 98.61 ms, versus 122.16 ms for the original
four-head kernel (about 19.3% lower attention duration). Each experiment
passed 96 standalone FP16/BF16 reference cases covering dense and irregular
selection, duplicate/invalid entries, odd packed request boundaries, masks,
softcap, head sinks, slot mappings and two-token verification.

An additional attention-only replay used captured selections from layers
zero and eleven, deterministic synthetic Q/K/V and the actual
8,192-query/24-head/2-KV-head shape. Inputs/outputs were device-bound, with
two warmups and five timed calls per case. Mean synchronized operator times:

| Replay | Four heads (ms) | Six heads (ms) |
|---|---:|---:|
| Layer zero selections, FP16 | 122.43 | 98.91 |
| Layer eleven selections, FP16 | 122.43 | 98.96 |
| Layer zero selections, BF16 | 142.12 | 103.76 |
| Layer eleven selections, BF16 | 142.08 | 103.80 |

All four replay outputs matched bitwise (zero maximum absolute difference).
These are isolated operator timings, not full-model throughput. A persistent
C++ six-head prefill reference regression was added but not compiled/run;
the editor does not discover these C++ tests, and the configured test target
still requires a broad rebuild. The standalone cases and full-shape replay
provide the executed numerical validation for this iteration.

Exact GenAI Engine width-two 8,192/512 benchmarks retained every slow sample:

| Measurement | Requests | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|---:|
| Six-head candidate, first invocation | 3 | 102.59 | 71.10 | 2.220 | 7.201 |
| Fresh four-head control repeat | 5 | 107.71 | 70.69 | 2.499 | 7.243 |
| Six-head candidate repeat | 5 | 102.51 | 71.11 | 2.215 | 7.200 |

Each invocation had one excluded warmup. Both candidate invocations confirm
about 0.28 seconds lower TTFT (11.4% in the fresh comparison). The fresh
E2E aggregates differ by only 0.6% because candidate decode included a
60.97-TPS request, versus a 76.01-TPS slow control request. Do not infer
decode improvement or report only fast requests: this change targets
prefill, and occasional decode slowdowns remain unresolved. Full-model
greedy nondeterminism also remains. All requests had exact token counts,
zero MTP failures and verified isolated ORT/GenAI library paths.

The retained build and isolated six-head provider have SHA256
`fae7a9fff0dc664843f732e85ec1b7b0c3d8dbb3f27713b041483aeda4b73358`.
Installed packages and original graphs/configuration/weights are unchanged.
The next priorities are identifying slow-decode causes and a larger
prefill attention arithmetic improvement, such as a validated tensor-core
path; direct rank-aligned cross-query tile reuse is not justified by the
measured selection layout.

#### Width-two slow-decode and verification profiling

The six-head provider was used unchanged with original graphs, width-two MTP,
the same 8,192/512 greedy workload and isolated library-path checks. An
opt-in benchmark diagnostic records Engine-run start/end times, emitted-token
counts and cumulative speculative counters per step. GPU-zero clocks,
power, temperature and utilization were sampled every 200 ms. Neither
device clocks nor system configuration was changed.

Eight measured requests after one excluded warmup reproduced a slow request:

| Diagnostic request | Decode TPS | Ordinary Engine-step median (ms) | Final Engine step (ms) | Final emitted-token transition |
|---|---:|---:|---:|---|
| Seven faster measured requests | 115.69-126.44 | About 18 | Shape-dependent | 510/511 to 512 |
| Eighth measured request | 83.56 | 18.00 | 1,898.21 | 511 to 512 |

The slow request had 225 speculative rounds and 285 accepted drafts, versus
214-233 rounds and 278-297 accepted drafts in the faster requests. Its
ordinary-step 90th percentile was 18.48 ms and maximum was 18.82 ms. The
large excess was concentrated in the last, non-speculative target step,
not steady-state verification. Its extra step added a target forward pass
but no draft proposal or speculative round.

Three additional full-decode captures used graph-node tracing and separate
CUDA-profiler ranges. In the second trace the final 510-to-512 step took
2,579.08 ms; ordinary steps still had an 18.17-ms median. The first trace's
wall-clock decode duration also contains substantial profiler-start overhead
outside Engine::Run. None of the profiled wall timings are throughput
measurements; opt-in per-step timing and telemetry also have some overhead.

The slow tail trace contains 96 `populateRandomBufferKernel` launches,
totaling 421.23 ms, from 4.437 to 6.917 seconds into the captured GPU span.
They occur with many grouped-GEMM candidate kernels and event synchronizations.
The other two traces contain no random-profiler initialization launches.
A new graph capture starts only after this tuning work, at 6.957 seconds;
the late graph-instantiation API takes 18.49 ms, not seconds. Graph creation
alone therefore does not explain this stall.

The source confirms a wasted-autotuning path in CUDA QMoE:

- `moe_quantization.cc` prepares grouped-GEMM tactics when
  `!use_fp8_fused && !use_packed_int`, without excluding `use_fp4_gemv`.
- On a non-capturing call, `profileTactics` tunes distinct actual row-count
  buckets for both projections. `GemmProfilerBackend::prepare` fills its
  workspace with random data before running candidate GEMMs.
- Later, `!use_fp4_deep_gemm && use_fp4_gemv` runs the FP4 GEMV projections
  and returns before the grouped-GEMM runner consumes those tactics.

Thus an unseen small-row bucket can pay grouped-GEMM tuning during a tail
step even though real inference uses GEMV. A warmup request does not
necessarily exercise every one-/two-/three-row tail shape. The confirmed
next fix is to bypass grouped-GEMM tactic/workspace preparation for the
actual FP4-GEMV execution path, preserving native/DeepGEMM/dense fallbacks
and routing scratch. It has not been implemented in this profiling
iteration. Regression validation should exercise cold row-count buckets,
check unchanged outputs, confirm profiler kernels disappear, and retain
all exact-request timings rather than hide stalls with extra warmup.
This identifies the reproduced stall, not every historical slow request.

Telemetry does not support sustained thermal or clock throttling as the
cause of this reproduced stall. The slow unprofiled request had median/max
SM clocks of 1,980 MHz, minimum 1,905 MHz, memory clock 3,201 MHz and
maximum temperature 51 C. Faster requests had similar clocks/temperatures.
Occasional software-power-cap event samples appeared in both fast and slow
requests; 200-ms sampling cannot rule out shorter events.

For width two, normal target verification uses three query rows. The clean
third trace shows grouped attention grid `(6, 3, 30)`, averaging 119.16
microseconds across 2,807 launches. Single-/two-row launches also occur in
draft-head preparation and tails; launch counts include eager/capture work.
Of 3,770.63 ms summed GPU kernel time in that trace:

| Kernel category | GPU time (ms) | Share |
|---|---:|---:|
| Fused NVFP4 MoE GEMV | 656.73 | 17.42% |
| Grouped sparse attention | 345.37 | 9.16% |
| FP8 MoE GEMV | 164.06 | 4.35% |
| Hierarchical indexer merge | 121.60 | 3.23% |
| Hierarchical indexer scoring | 27.78 | 0.74% |

Numerous dense projection GEMMs are additional large costs. CUDA API
synchronization time includes waiting for GPU work and must not be added
to GPU kernel time or interpreted as independently removable overhead.
Next after the autotuning fix: profile MTP projection nodes specifically,
and compare 30 versus 32 candidate splits for three-row verification with
correctness and exact-request measurements. No inference-source/model
changes were made by this profiling iteration.

#### Bypassing unused grouped-GEMM tuning for FP4 GEMV

CUDA QMoE now derives `run_fp4_gemv` from the actual standalone execution
predicate, `!use_fp4_deep_gemm && use_fp4_gemv`, and uses it both for dispatch
and to exclude grouped-GEMM tactic/workspace preparation. The shared routing
scratch allocation remains active; only unused runner workspace becomes zero.
Native grouped-GEMM, dense fallback, DeepGEMM and FP8 paths keep their existing
profiling behavior. Optional GEMV-specific autotuning is unaffected.

The CUDA provider builds successfully. Twenty-four cold-session NVFP4
operator cases match the previous provider bitwise, covering FP16/BF16,
with/without projection biases, one/two/three input rows, and hidden sizes
1,280/2,560/4,352. This validates routing/output behavior after removing unused
workspace as well as short verification shapes. These are executed standalone
operator checks, not a compiled C++ unit-test run.

Nsight confirms zero `populateRandomBufferKernel` launches across the 24
cold operator cases (48 actual FP4 GEMV launches) and each of three complete
width-two decode traces (21,888/21,696/21,504 FP4 GEMV launches). Grouped-GEMM
prefill is still allowed to tune; these checks target cold GEMV calls and
decode only, not all tuning during model startup.

Eight measured diagnostic requests after one excluded warmup achieved
119.32 decode / 78.78 E2E TPS with timing instrumentation enabled. Final
Engine steps took 15.67-72.63 ms, and no decode Engine step exceeded
92.90 ms. This includes one-, two- and three-token final emissions. Before
the fix, the equivalent eight-request diagnostic had a 1,898.21-ms final
step and an 83.56-TPS request. The three candidate decode traces had final
steps of 15.83-88.61 ms with no multi-second Engine stall. Their wall-clock
TPS remains invalid for throughput comparisons because of profiler
range/report overhead.

A fresh uninstrumented sequential candidate/control comparison retained all
five measured requests per provider, with one excluded warmup each:

| Provider | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) | Decode TPS range |
|---|---:|---:|---:|---:|---|
| GEMV tuning bypass | 121.67 | 79.79 | 2.217 | 6.417 | 114.33-125.32 |
| Previous six-head provider | 109.17 | 74.27 | 2.213 | 6.893 | 75.19-129.77 |

The candidate was 7.42% better E2E in this five-request comparison, saving
0.476 seconds per request on average. The control again had a slow request,
whereas the candidate did not. This is principally a tail-latency/tuning
fix, not evidence of faster steady-state GEMV arithmetic. Small samples,
fixed candidate-first order and existing full-model greedy nondeterminism
limit the precision of the throughput comparison. Finite tests do not
prove all possible causes of slow requests have been eliminated.

All 21 measured candidate/control requests (eight diagnostic, three
profiled, five candidate plain and five control plain) passed exact
8,192-input/512-output, prompt hash, width-two overlay, loaded-library-path
and zero-MTP-failure checks. No original model/configuration/weights or
installed package was changed. The new built/staged provider SHA256 is
`930005721d919d1c66c364b26c1ff10f23ac8a9fb8539e1d7006d7a62f49a9d4`.
The standalone reference runner, sequential tmux script, traces and
assertion-based result analyzer are retained as session artifacts. The
installed GenAI library still has its original SHA256
`c2df796387e8894268038ec690b0db12f2fe37be901ba55dc059a600fe7b9f27`.

#### Base-only profiling and twelve-head prefill reuse

The next iteration focuses on the base model, without changing MTP algorithms,
proposal width, graphs or weights. Fresh base-only GenAI Engine profiles use
the GEMV-tuning-bypass provider, the same 8,192/512 greedy contract and CUDA
graphs. MTP is disabled by an additive overlay for attribution only.

Summed GPU kernel times (not host launch spans or end-to-end timings):

| Category | Base prefill (ms) | Prefill share | Base decode share |
|---|---:|---:|---:|
| Sparse attention | 1,183.12 | 54.29% | 13.25% |
| Causal convolution | 221.94 | 10.18% | 0.87% |
| Standalone NVFP4 dequantization | 157.26 | 7.22% | 0% |
| Gated delta net | 116.70 | 5.35% | 2.13% |
| Dense `nvjet` GEMMs | 89.62 | 4.11% | 27.82% |
| Dense GEMV | 0 | 0% | 5.20% |
| NVFP4 MoE GEMV | 0 | 0% | 11.42% |
| Indexer kernels | 83.08 | 3.81% | 5.02% |

The prefill capture contains 2,179.35 ms summed kernel time. Decode captures
the beginning of generation through 64 emitted tokens and includes eager
and capture work, not exclusively graph replay. Its summed kernel time is
794.14 ms. Dense GEMMs/GEMV therefore remain a substantial base-only decode
target; these measurements no longer include draft-head work.

A separate graph-disabled, cold-request ORT node diagnostic identifies actual
base projection shapes. It reaches ORT's million-event cap and truncates
before request completion, so it is used only for node/shape metadata.
Host node spans include launch/tuning overhead and are not GPU attribution.
Large repeated FP16 projections include 36 `qkv_z_proj` weights of shape
`(2560, 16384)`, 48 attention-output projections `(6144, 2560)`, and the
single vocabulary projection `(2560, 248320)`. The original graph remains
unchanged; quantizing these requires accuracy and compatible-kernel validation.

The measured prefill candidate extends the existing grouped kernel from six
to twelve query heads per CTA when the GQA ratio is divisible by twelve,
query count exceeds 64, no candidate splitting is active, and shared memory
fits. A 384-thread block shares each selected K/V tile across all twelve
query heads belonging to one KV head; Qwen's grid X decreases from four to
two. Six-head eligibility/fallback is preserved, and decode/short verification
still uses four heads. Candidate resolution, dot-product ordering, softmax
and output accumulation are unchanged.

The experimental provider passed 96 FP16/BF16 standalone reference checks.
Captured 8,192-query layer-zero/layer-eleven attention replay showed:

| Captured selections | Six-head FP16 (ms) | Twelve-head FP16 (ms) | Six-head BF16 (ms) | Twelve-head BF16 (ms) |
|---|---:|---:|---:|---:|
| Layer zero | 98.85 | 70.48 | 103.71 | 88.55 |
| Layer eleven | 98.91 | 70.45 | 103.72 | 88.53 |

All four outputs are bitwise identical to their six-head references.
The persistent prefill C++ regression now targets twelve-head dispatch;
it has not been compiled/run. Production dispatch also keeps the prior
six-head fallback; its rebuilt provider passes dense-prefill FP16/BF16
reference checks before full-model profiling. These isolated timings are
not an end-to-end speedup claim.

Full-model graph-node profiling confirms twelve target attention launches
with grid X two and 384-thread CTAs. Summed attention time is 840.96 ms
versus 1,183.12 ms for six heads, saving 342.16 ms (28.92%). A fresh
uninstrumented base-only comparison, five measured requests per provider
with one excluded warmup, confirms that gain outside the profiler:

| Base-only provider | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Six-head + GEMV tuning bypass | 75.37 | 56.94 | 2.211 | 8.992 |
| Twelve-head + GEMV tuning bypass | 75.30 | 59.17 | 1.868 | 8.654 |

TTFT improves by 343.65 ms (15.54%); E2E throughput improves by 3.91%.
Base-only decode is unchanged within measurement variation, as intended.
Every measured request is retained. Integrated width-two MTP with the same
twelve-head base provider, unchanged speculative settings and five measured
requests, achieves 120.65 decode / 83.84 E2E TPS, 1.871-s TTFT and 6.107-s
mean request time. That integrated result is a compatibility/workload check,
not a fresh paired MTP speedup attribution; the direct gain comes from the
base-only comparison and attention trace/replay.

All 18 measured requests (ten base-only plain, five integrated MTP plain,
and three base-only profiler requests) passed exact token counts, prompt
hash, graph-enabled workload overlay and isolated library-path checks.
Base-only requests have zero draft proposals; integrated requests have
verified width-two proposals and zero MTP failures. Full-model greedy
outputs remain nondeterministic, so full-model bitwise equivalence is not
claimed. Sequential full-shape replay and control-first base benchmark
ordering, and the small request count, limit precise throughput estimates.

The retained built/staged provider SHA256 is
`10194174fcba522a331c07a178751a499c04c6e6155ff9f22ff9474166fa30ab`.
Original model files, installed packages and MTP configuration remain
unchanged. Temporary full-shape reference arrays are removed after their
bitwise comparison; captured selections, replay hashes/timings, scripts,
bounded node analysis and Nsight reports remain session artifacts.

The next base-model opportunity is prefill causal convolution: the current
general kernel assigns a channel to each thread and loops serially over
all tokens, explaining its high long-prefill cost. Temporal parallelism
must preserve initial/final-state alias safety and speculative state-update
capture. A separate final-state update after all output reads is a candidate
design; it is not implemented in this iteration. MTP-specific tuning is
deferred until the base-model work is evaluated.

#### Temporally parallel base-model prefill convolution

Long single-request `VarlenCausalConvWithState` calls now use temporal tiles
of 32 tokens with 128 channel-contiguous threads per block. This targets
Qwen's `(8192, 10240)` input, four taps, and dilation one or three. The
specialization requires at least 256 tokens; short requests, decode,
ragged batches, other tap/dilation combinations, and inputs exceeding
CUDA's grid-Y limit retain the existing kernels.

The convolution output kernel reads old state but never updates final
state. A second kernel on the same stream updates final state after all
temporal output tiles finish reading, so initial/final-state aliasing
remains safe during eager execution and CUDA graph replay. Since eligible
inputs are longer than the complete carry state, final state comes entirely
from the input suffix. Tap accumulation order, bias, SiLU, compact capture
counts/clamping and inactive-capture behavior are unchanged. No model or
MTP settings are modified.

Forty standalone cases compare bitwise with the previous twelve-head
provider and independently check final-state and compact-capture outputs.
Small cases also check a CPU convolution reference. Coverage includes
FP16/BF16/FP32, 255/256/257-token dispatch boundaries, 129-channel partial
tiles, bias/no bias, SiLU/no activation, both dilations, capture disabled,
inactive and clamped counts, and separate/in-place final state.
Four full-shape FP16 cases exercise both dilations and activation/aliasing
combinations. The same forty cases pass with CUDA graphs enabled, after
restoring initial state between replays. A persistent C++ boundary/capture
regression is added but not compiled/run: the editor discovers no tests
in that C++ file. Executed validation is the standalone numerical and
graph-replay suite plus model benchmarks.

At the full `(8192, 10240)` shape, synchronized operator timings fall from
5.52-5.98 ms to 0.30-0.36 ms, including final-state writes. Model-level
Nsight prefill confirms the same improvement across all 37 convolutions:
221.88 ms before versus 12.26 ms after (94.47% lower). The new trace contains
37 output kernels totaling 12.20 ms plus 37 final-state kernels totaling
0.059 ms. Sparse attention and other base-model kernels remain unchanged.

Fresh exact 8,192/512 uninstrumented comparisons use five measured requests
per provider and one excluded warmup per invocation; all slow samples are
retained:

| Workload/provider | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Base only, previous twelve-head provider | 75.49 | 59.28 | 1.868 | 8.638 |
| Base only, parallel convolution | 75.04 | 60.45 | 1.660 | 8.470 |
| Fixed width-two MTP, previous provider | 120.34 | 83.64 | 1.875 | 6.122 |
| Fixed width-two MTP, parallel convolution | 121.87 | 87.40 | 1.665 | 5.858 |
| Fixed width-two MTP, final-build repeat | 121.09 | 86.99 | 1.666 | 5.886 |

The direct base comparison confirms 207.91 ms lower TTFT and 1.98% higher
E2E throughput. Base decode varies by about 0.6%, not a decode gain from
this prefill-only specialization. The paired width-two comparison has
210.63 ms lower TTFT and 4.50% higher E2E throughput; some request-duration
variation also comes from nondeterministic outputs/speculative acceptance.
Do not attribute the entire E2E delta to convolution arithmetic.

The final-build repeat follows a safety-only widening of the channel-index
guard to avoid overflow at extreme channel counts. Its forty graph-enabled
operator cases again match baseline outputs bitwise, with full-shape timings
of 0.30-0.35 ms. Pooling all ten candidate width-two requests across both
invocations gives 121.48 decode / 87.19 E2E TPS, 1.665-s TTFT and 5.872-s
mean request time. These are sequential small-sample measurements, not a
guarantee that every workload or long-run request achieves that throughput.

All 26 measured requests in this iteration (ten base-only, fifteen integrated
width-two and one prefill-profile request) pass exact token counts, prompt
hash, graph-enabled overlays and isolated loaded-library-path assertions.
Base-only requests make zero draft proposals; integrated requests have
verified width-two proposals and zero MTP failures. Original model files,
weights, MTP configuration and installed packages are unchanged.

The retained built/staged provider SHA256 is
`49e9953f6e387920bb2230c6c19070cdba71e9022b95162d7739b293e373dba0`.
Scripts, timings, model traces and the assertion-based analyzer are retained
as session artifacts; temporary standalone reference arrays are removed
after validation. Further base-model priorities are dense decode projections
(about 33% of summed base-only decode GPU time in the earlier profile) and
the remaining sparse-attention prefill arithmetic. MTP-specific optimization
remains deferred.

#### Base-decoder dense GEMM tuning

The base decoder's existing optional `ep.cuda.enable_gemm_auto_tune` policy
compares cuBLAS, small-N GEMV and SM90+ tinygemm2 per eligible FP16/BF16
shape. The default remains cuBLAS. This iteration evaluates the policy
without changing inference source, graph weights, quantization or MTP
algorithms. The current convolution-optimized CUDA provider is unchanged.

Thirty-six synthetic FP16 projection cases cover twelve actual base
weight shapes at M one, two and three. They run with device-bound inputs,
constant weights and CUDA graph replay. Tuned outputs match cuBLAS within
the validation tolerance; maximum absolute difference is 0.00024414.
They are not bitwise identical in every case. Observed isolated improvements
include router/indexer/gate projections, but timings include Python launch
overhead and reuse a single weight matrix in cache. They are screening
measurements, not full-model speedups or accuracy validation.

The actual base-only 64-token decode trace confirms 6,432 tinygemm2
launches totaling 26.83 ms. With sixteen columns per CTA, grids X six,
32 and 40 correspond to N 96 (36 linear-attention gate projections),
512 (48 expert routers) and 640 (12 indexer projections); their launch
counts are 2,412/3,216/804. No small-N GEMV launches appear in this trace.
Larger projection shapes retain cuBLAS; the large `(2560, 16384)` QKV
matrix exceeds tinygemm2's current 32-million-weight-element eligibility
cap, and `(2560, 248320)` vocabulary weights are also ineligible.
Do not infer that all dense projection work is now optimized.

The integrated overlay explicitly sets MTP GEMM auto-tuning to zero,
because its session options otherwise inherit unset entries from the base
decoder. Applying only the decoder entry would unintentionally tune the
MTP head too. The overlay keeps CUDA graphs, 8,192 scheduled tokens,
utilization factor 0.05 and draft width two unchanged. The relevant
additive configuration is:

```json
{
  "model": {
    "decoder": {
      "session_options": {"ep.cuda.enable_gemm_auto_tune": "1"}
    },
    "mtp": {
      "session_options": {"ep.cuda.enable_gemm_auto_tune": "0"}
    }
  }
}
```

First occurrences of eligible shapes perform real tuning and synchronization
outside graph capture, with results cached process-wide. Unseen row counts
can therefore add latency later; the prior fix for *unused MoE GEMM*
profiling does not eliminate legitimate dense-MatMul tuning. Numerical
changes may also affect greedy outputs or speculative acceptance. Keep
this policy opt-in and validate the deployment's workload/accuracy rather
than globally forcing small-N GEMV. Model-loading and excluded warmup
are not part of the reported steady request throughput.

Fresh five-request comparisons after one excluded warmup per invocation:

| Measurement | Policy | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---|---:|---:|---:|---:|
| Base, step-timing diagnostic | cuBLAS | 75.33 | 60.64 | 1.659 | 8.443 |
| Base, step-timing diagnostic | Tuned | 76.11 | 61.15 | 1.659 | 8.373 |
| Base, plain repeat | cuBLAS | 75.37 | 60.66 | 1.661 | 8.440 |
| Base, plain repeat | Tuned | 75.98 | 61.04 | 1.662 | 8.388 |
| Width two, step-timing diagnostic | cuBLAS | 120.25 | 86.57 | 1.665 | 5.914 |
| Width two, step-timing diagnostic | Base tuned | 124.88 | 88.98 | 1.662 | 5.754 |
| Width two, plain repeat | cuBLAS | 119.39 | 86.16 | 1.662 | 5.942 |
| Width two, plain repeat | Base tuned | 121.84 | 87.42 | 1.663 | 5.857 |

Base-only decode improves 1.04% in the diagnostic and 0.81% in the
uninstrumented reverse-order repeat. The latter saves 52.73 ms per request
and improves E2E by 0.63%. Integrated width-two E2E improves 2.79% in the
diagnostic and 1.46% in the plain reverse-order repeat. Some integrated
variation comes from greedy-output/speculative-acceptance differences;
do not claim all of that delta as a kernel speedup. Timing instrumentation
also adds overhead; use the plain repeat for the conservative result.
Both invocation orders were tested and every measured sample retained.
This is a modest opt-in improvement, not a large projection breakthrough.

The five measured tuned base diagnostic requests had maximum decode Engine
steps of 69.20-71.36 ms. The tuned width-two diagnostic had maximum steps
of 80.34-140.03 ms; no multi-second tuning stall occurred in these requests.
These finite tests do not eliminate the cold/new-shape tuning caveat.

All forty measured comparison requests pass exact 8,192/512 token counts,
prompt hash, overlay, library-path and zero-MTP-failure assertions.
The separate 64-token base trace finishes the same 512-output request,
has the expected isolated library paths and verifies actual tuned launches.
The initial integrated tuned invocation rejected a misnested MTP override
before loading a model; its JSON nesting was corrected and parser-checked,
then that invocation was rerun successfully. No measured request was
discarded as part of that recovery.

The policy is retained as additive session overlays:
`qwen_base_gemm_tune_overlay.json` for base-only work and
`qwen_width2_base_gemm_tune_overlay.json` for unchanged-width-two integration,
in the session artifact directory. Apply the JSON with `Config.overlay`
before creating the model. The inference source, original model/config
and installed packages remain unchanged; the provider SHA256 remains
`49e9953f6e387920bb2230c6c19070cdba71e9022b95162d7739b293e373dba0`.
Standalone references are removed after comparison; scripts, traces,
per-step records, checks and result summaries persist.

#### Large-QKV tinygemm2 eligibility experiment (rejected)

An isolated provider temporarily raised tinygemm2's weight-element limit
from 32 to 64 Mi elements, admitting the base linear-attention QKV matrix
`(K=2560, N=16384)` at M one, two and three. It did not change weights,
quantization, kernel geometry, or MTP behavior. All three tuned standalone
outputs matched the previous provider bitwise. This validates the selected
dispatch outputs, not the accuracy of a forced tinygemm2 path.

Default-logger verbose tuning confirms cuBLAS wins for every tested row
count. Session-level verbose logging alone does not expose these messages,
because the tuner uses `LOGS_DEFAULT`.

| Rows | cuBLAS tuning time (us) | tinygemm2 tuning time (us) | Selected |
|---|---:|---:|---|
| 1 | 22.048 | 43.392 | cuBLAS |
| 2 | 21.760 | 43.808 | cuBLAS |
| 3 | 21.280 | 45.152 | cuBLAS |

These are the tuner's CUDA graph replay measurements with L2 flushing and
hot activations, not Python microbenchmark wall times. Simply admitting
the larger matrix adds tuning work without selecting a faster kernel.

Fresh base-only comparisons use the same opt-in base GEMM tuning overlay,
8,192 input and 512 output tokens, one excluded warmup and five measured
requests per provider. The experimental provider ran before the control;
all samples are retained.

| Provider | Decode TPS | E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Retained 32-Mi limit | 76.144 | 61.169 | 1.6593 | 8.3702 |
| Experimental 64-Mi limit | 76.073 | 61.132 | 1.6580 | 8.3753 |

The 0.094% decode and 0.061% E2E decreases are too small to interpret as a
meaningful regression, but there is no demonstrated gain. Token counts,
prompt hash, overlays, loaded isolated-library paths, zero draft proposals
and zero MTP failures pass assertions for all runs. No additional integrated
MTP benchmark is needed for a rejected eligibility-only change.

The cap is restored to 32 Mi elements; the existing eligibility regression
expectations remain unchanged. The retained isolated runtime and optional
base-only tuning policy are unaffected. The CUDA provider rebuild succeeds,
and the source file has no remaining diff. Its rebuilt SHA256 is
`b2445fc9d40c853d44186d654ceeb4b9653f477242054926b4354c1e23b50eff`,
not the retained isolated binary's `49e9953...` hash. A separately staged
rebuilt runtime matches the retained runtime bitwise for the three large-QKV
row counts; this is targeted validation, not a full-model benchmark of
the relinked binary. The retained deployment runtime is not replaced.
Temporary reference arrays are removed; benchmark scripts, logs and
assertion-checked summaries persist. Large dense projections need a
different kernel geometry or a separately accuracy-validated representation
change, rather than this eligibility expansion. MTP-specific optimization
remains deferred.

#### Interleaved sparse-attention prefill dot products

The twelve-head grouped prefill path now interleaves four independent
candidate-row dot products per warp. Each row retains its original FP32
channel accumulation and warp reduction order; K/V staging, online softmax,
value accumulation, head sinks and candidate filtering are unchanged.
Only the existing twelve-head long-prefill dispatch uses four-row
interleaving. Six-head/four-head fallbacks and short decode/verification
continue to use one row at a time. Tail rows are guarded before shared
memory accesses and logits writes.

Two alternatives were screened and rejected. Redistributing four rows
across eight-lane subwarps passed tolerance checks but took approximately
106.6 ms FP16 / 132.2 ms BF16 per full-8K attention layer, versus the retained
70.5 / 88.5 ms baseline. Eight independent full-warp interleaved rows were
also slower than four (approximately 64.1 / 89.5 ms versus 63.2 / 85.5 ms).
The retained implementation does not use tensor cores or lower-precision
softmax and does not modify model weights.

The final CUDA provider builds successfully. Ninety-six standalone
FP16/BF16 reference checks cover twelve-head ragged prefill, causal and
noncausal masks, duplicate/invalid candidates, current and past K/V,
invalid slots/blocks, empty selections, head sinks, softcap and unchanged
fallback/width-two verification paths. Four captured-selection full-8K
FP16/BF16 cases match the previous provider bitwise both eagerly and
with CUDA graph replay. Graph-replay means are:

| Selection / dtype | Previous (ms) | Four interleaved rows (ms) |
|---|---:|---:|
| Layer 0 / FP16 | 70.446 | 63.234 |
| Layer 11 / FP16 | 70.413 | 63.199 |
| Layer 0 / BF16 | 88.514 | 85.544 |
| Layer 11 / BF16 | 88.527 | 85.561 |

The full-model prefill profile confirms all twelve attention launches use
the four-row specialization. Their summed GPU time falls from 840.90 ms
in the earlier convolution-optimized profile to 754.66 ms, a 10.3% reduction.
Fresh full-shape replays above are the isolated comparison; the two
full-model profiles were captured in separate iterations. Instrumented
wall-clock request timings are not throughput results.

The C++ twelve-head regression is extended to selected widths divisible
by four and widths with one/two/three-row remainders. Those new C++
regression cases have not been compiled or run: editor test discovery
finds no tests and the available old binary does not list this suite.
The executed numerical checks use standalone ORT sessions with the final
isolated CUDA provider.

Fresh exact 8,192/512 request comparisons use opt-in base GEMM tuning and
explicitly disabled MTP GEMM tuning. Each invocation excludes one warmup
and measures five requests; reverse-order repeats give ten measured
requests per provider/mode. Pooled TPS divides total generated tokens by
total elapsed time, and every measured sample is retained.

| Mode / provider | Pooled decode TPS | Pooled E2E TPS | Mean TTFT (s) | Mean request time (s) |
|---|---:|---:|---:|---:|
| Base / previous | 76.016 | 61.089 | 1.6590 | 8.3812 |
| Base / interleaved prefill | 76.087 | 61.780 | 1.5714 | 8.2874 |
| Width two / previous | 126.606 | 89.851 | 1.6621 | 5.6983 |
| Width two / interleaved prefill | 124.673 | 90.204 | 1.5773 | 5.6760 |

Base TTFT improves 87.54 ms and E2E improves 1.13%, reproduced in both
orders. Width-two TTFT improves 84.83 ms; pooled E2E improves only 0.39%
because decode/acceptance variation offsets some prefill savings.
The two width-two candidate invocations measure 90.86 and 89.56 E2E TPS,
versus 90.19 and 89.52 for the controls. Do not attribute the decode
differences to the prefill-only kernel or claim the single best sample
as sustained throughput. All forty measured requests pass exact token,
prompt-hash, overlay, loaded-library-path and zero-MTP-failure assertions.
The prefill change is retained for its repeated, directly measured TTFT
improvement. On this workload, 100 E2E TPS requires total request time
at most 5.12 seconds, not merely 100 decode TPS.

The retained runtime is staged separately as `sparse-prefill-ilp-final-python`;
CUDA provider SHA256 is
`85eefd48861f30392d9d91e7ee49245211a3d9903c72f1958f8e2260cad3edaf`.
The installed packages, previous isolated runtime, original graphs/config
and weights are unchanged. Full-model output equivalence is not claimed;
existing greedy nondeterminism remains observable.

#### NVFP4 verification row reuse and block geometry (rejected)

After publishing the 100-E2E-TPS branches, a bounded diagnostic captured
the first 192 eligible three-row NVFP4 layer invocations. A temporary
host routing copy ran only with CUDA graphs and dense GEMM tuning disabled;
its synchronized timings are not performance evidence. The diagnostic
hook was removed. Each invocation contains 30 routes (top ten per row),
averaging 23.531 unique experts and 3.195 shared experts between adjacent
query rows. Pairing expert-sorted rows could eliminate at most 19.6875%
of weight passes; reusing across all three rows could eliminate 21.5625%.
These are load-count upper bounds, not measured DRAM savings: L2 may
already serve repeated weights. This bounded, single-prompt sample does
not characterize every decode round.

An opt-in paired-row K-packed GEMV kept the original eight-lane K
partition, accumulation and reduction order for each output. Within each
expert's sorted range, even relative rows owned the following row if
present, shared packed-weight/scale decoding and wrote both outputs;
odd relative rows returned, and singleton/tail experts used one output.
FC1 source-row mapping, FC2 expanded inputs, FP16/BF16 and projection
biases were preserved. Seventy-two operator comparisons matched the
original provider bitwise, eagerly and through CUDA graph replay,
covering captured-like, complete and disjoint overlap and one/two/three
rows. Full overlap improved some screening cases by about 6-8%, but
disjoint three-row cases regressed by about 10-19%. Operator timing
reuses small synthetic expert matrices and is not full-model evidence.

The subsequent geometry-only experiment kept the original per-row
arithmetic and swept 8/16/32 output columns per block with 64/128/256
threads. All 216 operator comparisons matched the original provider
bitwise, including graph replay and all three routing patterns. Eight
columns showed small screening advantages; 32 did not justify further
integrated testing. An actual-model eight-column capture confirms
64-thread kernels and 38 registers/thread. Its average gate/up and down
durations are about 39.51 and 21.09 us, versus about 39.49 and 20.98 us
in the earlier production capture. Launch counts differ, so raw summed
times cannot be used as equal-work speedup evidence.

Both candidates were tested separately against the retained production
runtime, with the exact 8,192/512 policy, width two, both tensor flags,
decoder/head tuning and CUDA graphs. Each provider has two five-request
invocations, with provider order reversed and one excluded warmup per
invocation. Every measured sample is retained; instrumented profiles
are excluded. The two experiments have independent fresh controls:

| Experiment / provider | Pooled E2E TPS | Pooled decode TPS | Disposition |
|---|---:|---:|---|
| Paired-row fresh control | 100.172 | 125.933 | Retained |
| Paired-row candidate | 99.117 | 124.264 | Rejected, 1.05% lower E2E |
| Geometry fresh control | 101.176 | 127.598 | Retained |
| Eight-column candidate | 99.692 | 125.189 | Rejected, 1.47% lower E2E |

Actual paired-row profiling confirms both projection launches use the
two-row template, with 49/51 registers per thread. Gate/up and down
average about 40.81/21.91 us, slower than the earlier production means.
The profile and microbenchmarks do not independently identify which
fraction of that change comes from register pressure, branches or cache
behavior. Integrated output/acceptance variation also affects E2E.
Neither proposal demonstrates a gain, so both kernel experiments and
their environment switches are removed; the production provider is
rebuilt from the original source. The validated isolated 100.48-TPS
runtime is never replaced.

All forty measured comparison requests pass exact token counts, prompt
hash, flags, overlays, library paths, width-two proposal accounting and
zero MTP failures. Scripts, logs, candidate libraries and profiles remain
local session artifacts. The assertion-based summary is
`nvfp4-experiment-results.json`; the overlap report is
`nvfp4-routing-overlap-results.json`. Temporary numerical reference arrays
are removed after validation. No additional C++ test is retained for
these rejected source changes. The next priority is attribution and
optimization of large dense decode projections, rather than further
overlap-dependent GEMV work.

#### Four-K-lane NVFP4 K-packed tile (rejected)

A separate follow-up reduced the K-lane subgroup from eight to four and
expanded the K-packed CTA tile from 16 to 32 output columns, leaving the
128-thread block size unchanged. This changes the dot-product reduction
partition, so exact greedy-sequence equivalence is not claimed. The
targeted three-token MTP-window parity case passes against the
dequantized fallback, and both full Engine variants completed all
measured 512-token requests with zero MTP failures.

The comparison used the same candidate provider binary with the
four-lane environment flag off/on, width-two MTP, exact 8,192/512 policy,
CUDA graphs, both sparse-prefill tensor-core flags and decoder/head
tuning. Two batches with reversed variant order measured eight requests
per variant:

| Variant | Pooled E2E TPS | Decode TPS |
|---|---:|---:|
| Original eight K lanes | 104.508 | 132.793 |
| Four K lanes / 32-column tile | 103.539 | 131.208 |

The candidate is 0.93% lower in E2E and 1.19% lower in decode throughput.
Acceptance also differed (76.42% control versus 74.49% candidate), so the
full-request comparison is not a pure kernel measurement. Matched
profiled per-call gate/up duration rose from 39.58 to 40.78 us; down
duration fell from 20.91 to 20.36 us. One targeted MTP-window parity
case passed. The mixed result does not justify keeping this geometry. Its
flag and kernel changes were removed, and the CUDA provider was rebuilt
from the original source.

#### Dense-projection attribution and higher split-K parallelism

A temporary, graph-disabled MatMul diagnostic captured 16,000 synchronized
node events. It identified frequent three-row `input_mix_down_block_inject`
projections (N=324, K=10240), alongside large QKV, output and vocabulary
projections. These events identify shapes and dispatch choices, not
production wall-time attribution: synchronization changes execution and
graph-disabled selection differs from graph replay. The diagnostic is
removed from production source.

The retained local follow-up increases the shared small-N split limit
from 32 to 64 and the vector occupancy target from 128 to 256 CTAs.
Existing minimum-K-per-slice constraints, workspace sizing and the
GEMM tuner remain in place. The scalar fallback can also use 64 splits.
Standalone graph-tuner timings for N=324, K=10240 improve M2 GEMV from
9.728 to 9.280 us and M3 from 10.944 to 10.176 us; M1 still selects
cuBLAS. Plain Python graph microbenchmarks do not show a convincing wall
gain on their own, and verbose versus plain wall timing is not compared.

Twenty exact warmed requests use the unchanged final width-two policy,
with five requests per invocation in control/candidate/candidate/control
order. No measured samples are discarded:

| Configuration | Pooled E2E TPS | Pooled decode TPS | Mean TTFT | Mean request |
|---|---:|---:|---:|---:|
| Fresh production control, 10 requests | 99.692 | 125.186 | 1.053867 s | 5.135816 s |
| Higher-split candidate, 10 requests | 101.370 | 127.846 | 1.053759 s | 5.050782 s |

Candidate invocations individually reach 100.988 and 101.755 E2E TPS,
versus 98.888 and 100.510 for control. The pooled observed changes are
**+1.68% E2E and +2.12% decode**. Fresh, separate 64-token decode profiles
confirm grid (6,32) -> (6,64) for the targeted vector GEMV; M3 averages
8.70698 -> 7.72424 us/launch, **11.3% lower GPU duration**. Profile timing
is excluded from E2E measurements.

Verification rounds total 2,233 -> 2,195 and accepted drafts 2,872 ->
2,911, so acceptance/output variation contributes to the full-request
change. Split partitioning changes summation order. All 36 tuned
base-projection graph comparisons pass their existing tolerances; the
largest FP16 absolute difference is 0.0001220703125. All 11 targeted
C++ tests pass, covering FP16/BF16, scalar/vector variants, M1-64,
column/K boundaries, higher split counts, workspace capacity, repeated
CUDA-graph replay, forced/tuned dispatch and ineligible-shape fallback.
No general quality or full greedy-sequence equivalence is claimed.

The candidate provider SHA256 is
`c5e49b2f0adbb0e52fbda61664bd9c3c10b11a0ef17f4b3147a8ee5c238dac5b`.
The original 100.48-TPS isolated provider remains unchanged at
`103a10ceae4aaac2ae5dbdb0c6f40c35c012dcc3a8dfd1491b9a5fb564e035d2`.
The local production build contains the candidate; no deployment files
or model weights are replaced. Source and added tests are in separate
ORT commit [`0a03b230a7`](https://github.com/microsoft/onnxruntime/commit/0a03b230a7)
on `asonawane/qwen-38-flash-100-e2e`. Results documentation is committed
separately in GenAI; previously published commits are unchanged.

Artifacts are local session files: `benchmark_dense_split.sh`,
`analyze_dense_split.py`, `dense-split-results.json`, the four
`qwen_dense_split_{control,candidate}{,_repeat}.json` results and separate
control/candidate Nsight traces. The analyzer asserts workload, prompt,
policy, loaded library paths, draft accounting, zero failures, provider
hashes and actual split grids. Temporary numerical reference arrays are
removed after validation. Peak memory has not been remeasured for this
candidate; the previous sampled memory result applies to the original
configuration only.

#### Per-row NVFP4 loop scheduling and MTP FP8 scale hoisting

The next two experiments use the published split-K provider as control,
not the older 100.48-TPS provider. All requests retain the exact 8,192
uncached input / 512 greedy output policy, width two, tensor flags,
decoder/head tuning and graphs. Each invocation excludes one warmup;
all five measured requests are retained. Profiles are separate.

First, a two-way unroll directive on the raw K-packed NVFP4 reduction
loop tries to expose independent weight/activation loads without changing
per-row arithmetic. Seventy-two FP16/BF16, bias/no-bias and routing-overlap
cases match control bitwise, including CUDA graphs. Small-operator
screening improves, but twenty full requests in control/candidate then
candidate/control order regress **102.289 -> 100.828 E2E TPS (-1.43%)**
and **129.260 -> 126.965 decode TPS**. Verification rounds total
2,181 -> 2,241. This does not isolate a hardware regression from
acceptance variability, but it does not demonstrate the required E2E
benefit. The directive is removed; no NVFP4 source change is retained.

Second, FP8 MTP GEMV resolves expert/output-row weight and scale addresses
once before its K loop. For block size 128, it loads each lane's shared
block scale once for four successive K elements, retaining the same
FP16/BF16 weight rounding and FP32 FMA order. Other block sizes use a
generic path. The weight-row helper is shared with the existing GEMM
decoder, which retains its mathematical operation. Draft width,
scheduling, activation selection and token selection are unchanged.

An initial twenty-request comparison improves 100.722 -> 101.828 E2E TPS.
A second reversed-order batch retains every sample, bringing the complete
comparison to forty measured requests:

| Provider | Requests | Pooled E2E TPS | Decode TPS | Mean TTFT |
|---|---:|---:|---:|---:|
| Fresh published split-K control | 20 | 100.382 | 126.303 | 1.054653 s |
| MTP scale-hoisting candidate | 20 | 102.163 | 129.097 | 1.053310 s |

Observed gains are **1.77% E2E and 2.21% decode**. Candidate invocation
TPS values are 102.513, 101.152, 103.914 and 101.124. Two adjacent
candidate/control pairs are slightly negative, while both twenty-request
batches pool positively; this is not a per-request or per-pair guarantee.
Total verification rounds differ 4,443 -> 4,404, so the full-request gain
cannot be assigned solely to kernel speed.

Fresh 64-token profiles compare the same projection/route shapes:

| FP8 projection / routes | Control mean GPU us | Candidate mean GPU us |
|---|---:|---:|
| Gate/up / 10 | 114.436 | 64.655 |
| Gate/up / 20 | 197.169 | 111.935 |
| Gate/up / 30 | 285.578 | 154.991 |
| Down / 10 | 53.363 | 32.839 |
| Down / 20 | 98.784 | 55.715 |
| Down / 30 | 145.265 | 81.363 |

Registers decrease 40 -> 30 per thread. Launch counts differ; raw
cumulative profile totals are not equal-work speedup evidence.
Thirty-six full-QMoE FP16/BF16 cases with all three scale types and
optional bias match control bitwise through graph capture/replay.
Four targeted C++ tests pass. The new direct projection test exercises
432 combinations of FP16/BF16 activation, float/FP16/BF16 scales,
plain/split/block-fused weights, block sizes 64/128, K=127/128/257,
GEMV/GEMM and eager/graph replay, including row remapping and N tails.

Candidate provider SHA256:
`58232a5fa7f74ed561172638aab575f23a791812eb0d3cb4d487372dc42446c2`.
The original isolated split-K provider remains unchanged:
`c5e49b2f0adbb0e52fbda61664bd9c3c10b11a0ef17f4b3147a8ee5c238dac5b`.
The local production build contains only the retained MTP candidate.
Source, tests and results documentation are local and uncommitted/unpushed.
No new general model-quality or full-sequence-equivalence claim is made,
and peak memory has not been remeasured.

Local artifacts include `test_nvfp4_unroll.sh`, `test_mtp_scale_hoist.sh`,
`repeat_mtp_scale_hoist.sh`, `analyze_nvfp4_mtp_followup.py` and
`nvfp4-mtp-followup-results.json`, numerical comparison reports, raw
request JSONs and separate profiles. The analyzer asserts exact prompt,
policy, loaded provider paths, draft accounting and zero failures for
all sixty measured requests. Temporary reference arrays are removed
after validation.

#### MTP projection/activation fusion and eight-warp geometry (rejected)

Two further draft-path experiments compare against the retained isolated
scale-hoisting runtime. Each has an independent fresh control, twenty
measured exact 8,192/512 requests, control/candidate/candidate/control
ordering, one excluded warmup per five-request invocation and unchanged
width-two policy. All samples are retained.

The first fuses split FP8 gate/up GEMV with SiLU/multiply when there is
no FC1 bias, the projection uses GEMV and N is divisible by four.
Warp leaders stage projection values through shared FP16/BF16 memory,
preserving the original intermediate rounding, activation clamp and
lane FMA/reduction order. A block barrier permits paired gate/up output
without writing the projection to global memory. An actual-model
64-token profile confirms the fused specialization is selected and no
separate FP8 activation kernel is launched. Thirty-six full-QMoE
FP16/BF16 graph cases, with all scale types and optional down bias,
match the scale-hoisting runtime bitwise. Nevertheless the fresh
full-request pool is **102.615 -> 102.577 E2E TPS (-0.037%)** and
**129.825 -> 129.727 decode TPS**. This is no demonstrated improvement,
not a conclusive slowdown. Verification rounds differ 2,196 -> 2,184.
The fusion API, dispatch, kernel specialization and temporary candidate
test additions are removed.

The second keeps one warp per output column and identical arithmetic
but doubles FP8 GEMV blocks from four to eight warps (128 -> 256 threads,
four -> eight output columns). Thirty-six graph cases match control
bitwise. Actual-model profiles confirm 256-thread blocks, 30 registers
and halved column-grid sizes; GPU means are not better than the earlier
scale-hoisting profile. The fresh pool falls **101.904 -> 100.680 E2E TPS
(-1.20%)**, with decode **128.695 -> 126.717 TPS**. Verification rounds
differ 2,205 -> 2,240, so the E2E difference is not exclusively hardware
execution cost. This geometry change is also removed.

The retained scale-hoisting source and its four targeted C++ tests are
restored and rebuilt successfully. Its original measured isolated
package remains unchanged. The restored build SHA256 is
`7b7f89b69fb46ae491f311e9cb27eebb97929871d2919511fc1882a58a1268d3`;
this rebuilt binary must not be confused with the original measured
scale-hoisting provider SHA256
`58232a5fa7f74ed561172638aab575f23a791812eb0d3cb4d487372dc42446c2`.
No additional optimization or scheduler/token-selection changes are
retained from these two experiments.

Local evidence is preserved in `test_mtp_silu_fusion.sh`,
`test_mtp_eight_warps.sh`, the numerical/throughput JSONs, separate
candidate profiles, `analyze_mtp_candidates.py` and
`mtp-candidates-results.json`. The analyzer checks all forty measured
requests for exact policy, prompt, library paths, draft accounting and
zero failures, and asserts the actual candidate dispatch. Temporary
numerical reference arrays are removed after comparison. These results
and the previous scale-hoisting source remain local/uncommitted, not
pushed.

#### MTP medium-width dense and same-warp gate/up candidates (rejected)

Inspection of the actual `mtp.onnx` graph confirmed two 2560-by-2560
input projections, a 10240-by-320 final mixing projection and the
2560-by-248320 vocabulary projection. The benchmark uses Engine;
its greedy device selection consumes FP32 logits. No logits-cast
or selection-policy change was made.

The first candidate widened existing small-N split-K eligibility from
1024 to 4096 columns, leaving autotuning to choose the implementation.
Six FP16 projection cases (M=1/2/3 for K=10240,N=320 and K=N=2560)
matched the tuned reference exactly. Graph-replay tuner timings for
the newly eligible N=2560 cases were 10.080/10.400/11.104 us for
split-K GEMV versus 7.648/7.712/7.200 us for cuBLAS. cuBLAS won all
six shapes. The eligibility change was removed without a full-request
benchmark; it cannot establish an E2E gain.

The second candidate placed both gate and up accumulators in the same
warp, sharing activation loads and applying SiLU after the two warp
reductions. Unlike the previous shared-memory fusion, it required no
barrier or shared projection buffer. It preserved weight and projection
FP16/BF16 rounding, scale-hoisted FMA order, clamps and activation
arithmetic. Dispatch was restricted to split gate/up, no FC1 bias,
128-element scales and GEMV; all other cases retained their old path.

All 36 full-QMoE FP16/BF16, scale-type and bias combinations matched
the retained scale-hoisting runtime bitwise in eager and CUDA graph
replay. Twenty exact 8192-input/512-output requests ran sequentially
control/candidate/candidate/control, with five measured requests after
one excluded warmup in each invocation:

| Variant | Pooled E2E TPS | Decode TPS | Mean TTFT (s) | Draft rounds |
|---|---:|---:|---:|---:|
| Retained scale-hoisting control | 101.680 | 128.306 | 1.052704 | 2,221 |
| Same-warp paired SiLU | 101.848 | 128.568 | 1.052510 | 2,210 |

The observed +0.165% E2E result is inconclusive, particularly with
11 fewer draft rounds. A separate 64-output-token profile confirmed
that the new kernel was selected and removed the activation launch.
Ten-route gate/up took 106.381 us in that trace versus 64.655 us in
the earlier retained-runtime shape-matched trace; 20/30-route timings
were also worse. These are different trace samples, not additive
wall-time savings or a fresh equal-work paired profile. The new kernel
used 34 registers per thread versus 30 for the retained GEMV.
No additional full-request gain is retained.

Both candidates were completely removed. The retained scale-hoisting
provider and test target were rebuilt, and all four targeted FP8 C++
tests passed. Original measured runtime packages and model files remain
unchanged. New results remain local/uncommitted, not pushed.

Local evidence includes `mtp-medium-{control,candidate}-micro.json`
and their verbose tuner logs, `test_mtp_paired_silu.sh`,
`analyze_mtp_paired_silu.py`, `mtp-paired-silu-results.json`,
the exact-request JSONs, candidate CUDA profile and restore build/test
logs. Temporary reference arrays were removed after comparison.

#### Fresh retained-runtime bottleneck profile (2026-10-09)

Reprofiled the original measured `mtp-scale-hoist-python` runtime,
provider SHA256
`58232a5fa7f74ed561172638aab575f23a791812eb0d3cb4d487372dc42446c2`.
The model, width-two overlay, both tensor-prefill flags, greedy policy
and exact 8192-input/512-output workload are unchanged. Each capture
follows one excluded warmup and uses a fresh Engine. Separate Nsight
Systems captures cover prefill through the first token and decode from
after the first token through token 512, with CUDA graph-node tracing.
These are different requests; their kernel sums cannot be combined into
one measured request wall time.

A separate unprofiled three-request check achieves **101.099 E2E TPS**
and **127.421 decode TPS**, mean TTFT **1.053966 s** and mean request
time **5.064319 s**. All requests generate exactly 512 tokens with zero
MTP failures. This fresh timing check does not replace the earlier
40-request comparison establishing the retained 102.163-TPS result.
Do not use profiler-instrumented request TPS as inference throughput:
capture start/stop and graph-node tracing substantially inflate timings.

Prefill records 3388 kernel executions, summing to 1019.840 ms:

| Prefill category | GPU kernel sum (ms) | Share of kernel sum |
|---|---:|---:|
| Sparse attention | 237.202 | 23.26% |
| NVFP4 weight dequantization | 149.723 | 14.68% |
| Grouped MoE GEMM | 119.184 | 11.69% |
| Gated delta recurrent attention | 116.664 | 11.44% |
| Dense GEMM/GEMV and split-K setup | 91.301 | 8.95% |
| Sparse indexer | 83.707 | 8.21% |
| Hyperconnection mixing and normalization | 83.613 | 8.20% |
| Causal convolution | 12.453 | 1.22% |

Decode records 650517 kernel executions, summing to 3473.011 ms:

| Decode category | GPU kernel sum (ms) | Share of kernel sum |
|---|---:|---:|
| Dense GEMM/GEMV and split-K setup | 1086.507 | 31.28% |
| Raw NVFP4 MoE gate/up and down | 642.435 | 18.50% |
| Sparse attention | 405.495 | 11.68% |
| Pointwise elementwise | 247.183 | 7.12% |
| MoE routing, activation and finalization | 200.109 | 5.76% |
| Hyperconnection mixing and normalization | 192.410 | 5.54% |
| Sparse indexer | 189.266 | 5.45% |
| Tensor splits | 122.144 | 3.52% |
| Gated delta recurrent attention | 105.300 | 3.03% |
| FP8 MTP MoE | 64.135 | 1.85% |
| Speculative state rollback/replay | 55.712 | 1.60% |
| Causal convolution | 51.321 | 1.48% |
| Logits selection and dtype conversion | 21.800 | 0.63% |

Shares use disjoint kernel-name categories, not request wall time.
Dense includes vector/small-M/cuBLAS projections and split-K counter
clears; other nongrouped CUTLASS GEMMs remain a separate category.
Generic library names cannot establish exact ONNX node attribution.

Actionable details and priorities:

1. **Dense projections and surrounding small kernels:** largest decode
   family. Small-N three-row vector GEMV alone takes 164.959 ms; tinygemm2
   takes 154.950 ms. Pointwise, tensor splits and hyperconnection kernels
   add another 561.737 ms across 304706 executions. Investigate eligible
   producer/consumer fusion and exact shape/node attribution rather than
   repeating the rejected eligibility-only expansions.
2. **Raw NVFP4 MoE:** gate/up takes 420.108 ms and down 222.328 ms.
   This remains substantially larger than FP8 draft MoE; investigate
   different loading/reuse layouts, not the already rejected loop
   unroll or pairing geometries.
3. **Short-query sparse attention:** tiled kernel takes 349.817 ms
   and reduce kernel 55.678 ms. The main tile computation dominates;
   optimize useful candidate work and query/head reuse before focusing
   only on its reduction.
4. **Prefill dequantization and recurrent attention:** 96 NVFP4
   dequantization calls take 149.723 ms; chunked gated-delta kernels
   take 116.597 ms. A fused quantized prefill path or bounded reuse
   is worth investigating, but caching all expanded weights is not
   recommended without a memory budget.
5. **Indexer merge:** tile-top-K merge alone takes 119.014 ms,
   approximately 63% of the indexer family. Target merging traffic
   and launch count rather than already-live-work-bounded scoring.
6. **Speculative rollback:** generic replay takes 50.029 ms across
   92 calls (543.792 us/call); fast gated-delta replay adds 5.684 ms.
   The indexer replay branch iterates over full state capacity, not
   merely live key length. Investigate a bounded replay specialization
   while preserving buffer-bank, reset and inactive-tail semantics.

The vocabulary-shaped `nvjet_sm90_hsh_512x8...` launch (grid 2x66,
661 calls) totals 190.155 ms, about 5.48% of the decode kernel sum.
Earlier model diagnostics associate it with the vocabulary projection,
but the current generic kernel name alone does not prove exact node
identity. It is important but is not the majority of dense compute.
GenAI FP16-to-FP32 logits conversion totals only 1.535 ms across
647 calls; removing that conversion alone is not a leading opportunity.
The broader 21.800-ms selection/conversion category also includes
internal ORT casts and device top-1/top-K kernels.

The decode GPU activity span is 3947.051 ms. Merging kernel, memcpy
and memset intervals yields 88.81% timeline coverage and 441.771 ms
of uncovered gaps; gaps over 10 us account for 270.173 ms. This is
not SM occupancy, proof of CPU bottleneck, or a recoverable-speedup
estimate. CUDA stream-sync APIs accumulate 2530.220 ms, largely
overlapping GPU execution; graph-launch APIs accumulate 894.911 ms
under graph-node instrumentation. Neither should be added to kernel
time or interpreted as independent host overhead. Decode D2D copies
total 29.115 ms. Some on-demand graph setup/tuning remains in the
capture (nine graph instantiations and 6.747 ms of tuner cache flushes),
so this is not a graph-setup-free steady-state microbenchmark.

Evidence is preserved locally in `profile_qwen_final.sh`,
`analyze_qwen_final_profile.py`, `qwen_final_20261009_bottlenecks.json`,
the prefill/decode `.nsys-rep` and SQLite captures, exact-request JSONs
and separate unprofiled benchmark. The analyzer asserts provider
identity, prompt, policy, overlay, loaded libraries, output counts and
zero MTP failures. No source or model optimization was applied.

#### Adjacent shared-expert SiLU graph fusion (rejected)

The user approved source optimization and isolated model-graph
experiments, with the original model left untouched. Inspection of
dense projection consumers identified shared-expert gate projections
followed by `Mul(x, Sigmoid(x))`. The first candidate reuses the existing
`com.microsoft::ScaledSiLU` operator with alpha=1. It replaces 48 pairs
in the base graph and one pair in the MTP graph, preserving graph
inputs, outputs, initializer metadata and all other nodes. It does not
fuse the MatMul epilogue or remove the larger MatMul/Sigmoid/
HyperConnectionPreMix chains.

The candidate has independently copied external weight files rather
than shared writable weight links. All three external weight files
match the originals byte-for-byte. The source model/configuration and
the retained `mtp-scale-hoist-python` runtime are unchanged. The stock
ONNX checker validates the base graph; both original and candidate MTP
graphs fail identically on the original standard-domain
`SimplifiedLayerNormalization` node. ONNX Runtime successfully loads
and executes both graphs with MTP and CUDA graphs.

Sixteen numerical cases cover FP16/FP32, rows 1/2/3/8192, 640 features,
positive/negative extremes and eager/graph execution. All cases pass
the fixed rtol=0.002, atol=0.00001 thresholds, and repeated graph outputs
are unchanged. Maximum differences versus separate Sigmoid/Mul are
0.0001220703125 for FP16 and 0.0000007405178621411324 for FP32.
The existing fused operator uses slightly different sigmoid arithmetic,
so this is not universally bitwise-equivalent. Local numerical checks
are not broader model-quality validation.

Twenty exact 8192-input/512-output requests run in
control/candidate/candidate/control order, five measured requests
after one excluded warmup per invocation:

| Variant | E2E TPS | Decode TPS | Mean TTFT (s) | Draft rounds |
|---|---:|---:|---:|---:|
| Retained control | 101.857 | 128.594 | 1.052899 | 2,221 |
| Adjacent SiLU fusion | 100.874 | 127.045 | 1.053412 | 2,240 |

Candidate invocations reach 102.010 and 99.763 E2E TPS versus control
101.368 and 102.350. The pooled -0.965% result shows no demonstrated
gain. Candidate acceptance/round counts also change, so the whole
difference is not assigned to kernel cost. All runs finish with 512
tokens and zero MTP failures. Control outputs already vary among
requests; token hashes cannot establish model-level equivalence here.

A separate full-decode CUDA graph-node trace confirms the fused kernel
is selected: 11629 ScaledSiLU calls use an eight-block grid absent from
the control trace, consistent with three-row, 640-feature SiLU.
The candidate trace uses 236 draft rounds versus 216 in the earlier
control profile, so raw total Sigmoid/ScaledSiLU durations are not an
equal-work speedup comparison.

The candidate is rejected and is not selected in retained benchmark
settings. No runtime source edits, installed-package changes or
original model edits are made. Further work should target the larger
projection-to-hyperconnection gate chains with preserved sigmoid
rounding, rather than promoting this numerically different small fusion.

Local evidence is preserved in `test_dense_adjacent_silu.py`,
`benchmark_dense_adjacent_silu.sh`, `analyze_dense_adjacent_silu.py`,
`dense-adjacent-silu-results.json`, the 16-case validation JSON, exact
request JSONs and separate decode profile. The rejected graph copies
remain isolated under `qwen-dense-adjacent-silu-model`; the retained
default model remains `qwen_38_flash_nvfp4_engine`. The analyzer checks
unchanged initializer metadata and unaffected nodes, exact benchmark
policy, library paths, provider hash, output counts, failure counts
and candidate kernel dispatch. No new changes are pushed.

#### Sigmoid and hyperconnection premix fusion (not retained)

A temporary opt-in `gate_sigmoid=1` attribute on HyperConnectionPreMix
accepted FP16 gate logits, reproduced the CUDA Sigmoid branch arithmetic,
rounded the sigmoid to FP16, then used the existing FP32 mixing order.
Default raw-gate behavior was unchanged. CPU support and explicit
WebGPU rejection accompanied the experimental CUDA mode; all these
source/schema changes were subsequently removed.

An isolated model removed 97 base-model and three MTP sigmoid nodes.
External weights were independently copied after the new runtime's
path validation rejected links escaping the candidate directory.
The initial interrupted benchmark is not part of the comparison.
Original model files, configuration and measured runtime remain
unchanged. This fuses the consumer, not the MatMul epilogue.

Each variant passed 80 bitwise eager/graph checks against separate
Sigmoid/PreMix: FP16 gates, FP16/FP32 streams, rows 1/2/3/8192,
hidden sizes 2560 and 7, and branch/singleton/feature/flattened layouts.
Numerical validation does not establish broader model quality.

| Variant/comparison | Control E2E TPS | Candidate E2E TPS | Interpretation |
|---|---:|---:|---|
| All-row fusion, twenty measured requests | 102.125 | 102.105 | Flat; candidate TTFT 1.070 s versus 1.054 s |
| Decode-only fusion, first twenty requests | 101.058 | 102.538 | Observed +1.46%; rounds 2230 versus 2202 |
| Decode-only independent reversed-order repeat | 101.998 | 102.023 | Observed +0.024%; rounds 2198 versus 2218 |
| Decode-only pooled forty requests | 101.526 | 102.280 | Observed +0.74%; rounds 4428 versus 4420 |

All requests use the exact 8192-input/512-output greedy width-two policy,
one excluded warmup per invocation, original tensor-attention flags and
loaded-library checks. Every measured request completes with zero MTP
failures. Acceptance variability prevents assigning pooled changes
entirely to fusion.

The decode-only variant used fusion for up to three rows and the
original sigmoid implementation plus scratch gate storage for larger
inputs. This removed the all-row TTFT penalty. Matched same-input CUDA
graph microprofiles show separate Sigmoid+PreMix kernel sums
3.141/3.235/3.373 us versus fused 2.784/2.802/2.966 us for rows 1/2/3,
approximately 11-13% lower. These are kernel sums, not request-wall
savings. The all-row full-decode trace confirms the fused specialization
is selected; it does not establish a repeatable E2E benefit.

The added operator mode and model requirement are not retained because
the request gain does not clearly reproduce. Source rollback restores
all eight modified schema/kernel files while preserving the earlier
FP8 scale-hoisting work. Core library, Python extension and CUDA provider
are rebuilt. All 13 existing HyperConnectionOpsTest tests pass in
onnxruntime_provider_test. An initial test invocation used the wrong
runner and executed zero tests; it is not counted as validation.

Local evidence includes `validate_hc_sigmoid.py`,
`benchmark_hc_sigmoid.sh`, `benchmark_hc_sigmoid_decode.sh`,
`profile_hc_sigmoid_micro.py`, `analyze_hc_sigmoid.py`,
`hc-sigmoid-results.json`, exact-request JSONs, isolated graph/runtime
copies, profiles and restore logs. The rejected graph requires its
experimental runtime and must not be used with restored libraries.
No new optimization or model configuration is selected or pushed.

#### Dense INT4 projection screening (synthetic, no model changes)

The next experiment tests the retained runtime's existing MatMulNBits
path rather than adding a kernel or quantizing the original model.
Actual ONNX projection dimensions are reproduced using deterministic
synthetic normally distributed FP16 weights/activations:
K,N=(2560,7168), (2560,6144), (6144,2560), (2560,248320), with M=1/2/3.
Quantization uses the packaged quantize_matmul_4bits helper, symmetric
zero point 8, FP16 block scales, bits=4, blocks 64/128 and accuracy_level=0.
FP16 MatMul retains production GEMM autotuning.

Both timing screens use the original measured scale-hoisting provider
SHA256 58232a5fa7f74ed561172638aab575f23a791812eb0d3cb4d487372dc42446c2.
Each timed case follows four warmup graph runs and captures 30 replays
in hot and cache-flush modes. The second screen measures INT4 before
FP16 and reverses block order; numerical metrics match the first screen
exactly. CUDA/NVTX traces verify all attributed executions are graph
nodes. Median sums of kernel duration per replay exclude setup,
autotuning, host calls, synchronization and the flush itself. Host-loop
timings are retained only diagnostically and are not used for speedups.

Cache-flush mode writes 256 MiB with cudaMemsetAsync and synchronizes
the device before each replay to avoid cross-stream ordering ambiguity.
It is intended to evict weight cache contents, not a measurement of
cache hit rate or a faithful reproduction of production scheduling.
Vocabulary weights exceed cache capacity even in the hot case.

| Projection K -> N | Block-128 M=1 speedup | M=2 speedup | M=3 speedup |
|---|---:|---:|---:|
| 2560 -> 7168 | 2.05-2.08x | 1.20-1.21x | 1.21-1.23x |
| 2560 -> 6144 | 1.99-2.01x | 1.14-1.16x | 1.69-1.73x |
| 6144 -> 2560 | 1.61-1.65x | 0.62x | 1.15-1.16x |
| 2560 -> 248320 | 1.97x | 1.69x | 1.37x |

Both runs agree on the main outcomes. The vocabulary projection takes
292-294 us in FP16 versus 148-149 us for INT4 M=1, 173-174 us for M=2,
and 214-215 us for M=3. Block 128 is generally slightly faster than 64,
at the cost of slightly larger quantization error. The attention-output
projection with M=2 regresses from about 14.8 us to 23.8-24.0 us; do not
convert all dense projections indiscriminately. Hot-cache results and
full block-64 results are retained in the machine-readable reports.

Profiled INT4 dispatch uses MatMulFloat4BitsKernelM1 and
MatMulFloat4BatchedKernel, not an assumed native INT4 tensor-core
implementation. Reported packed-weight-plus-scale storage improves
3.88x at block 128 and 3.76x at block 64. This excludes allocator
overhead, workspaces, temporary dequantization and duplicated packing;
it is not a measured model GPU-memory reduction.

All 48 quantized comparisons (24 per screen) pass unchanged
rtol=0.02, atol=0.002 checks against separately dequantized FP16
weights executed with GPU MatMul. Maximum absolute difference from
that reference is 0.00091552734375. Quantized and FP16 outputs remain
bitwise stable across repeated graph replays. Comparing instead to the
original unquantized FP16 outputs shows approximately 9.19-10.40%
relative L2 error for these random cases. That error comes primarily
from weight quantization and is not evidence of acceptable language
model quality or MTP acceptance.

Conclusion: lower-bit dense execution is a viable operator-level
direction, especially for the vocabulary head, but no model-quality,
prefill, acceptance or E2E result is established. A selective isolated
weight-calibration/model-quality experiment is the next gate.
No original model files, runtime libraries or runtime source are edited.

Local evidence: `benchmark_int4_projections.py`,
`int4_projection_nvtx.cc` (a small NVTX-only instrumentation wrapper),
`analyze_int4_projections.py`, both screen/repeat JSONs and Nsight
reports/SQLite files, and `int4-projection-comparison.json`. The
instrumentation library is built using existing g++ and the installed
Nsight headers; no new package dependencies or model artifacts are added.
The analyzer checks runtime identity, 36 timed cases per screen, 72
NVTX ranges per screen, graph dispatch and 30 replays per cache mode.

#### Sparse indexer live-key radix selection (retained opt-in)

Instead of merging a capacity-sized padded tree, the eligible short-QSA
path can reuse `QsaPartialTopKKernel` with packed scored keys as input.
The existing scoring kernel and its 32-key sorted tiles are unchanged.
Radix selection scans only the causally visible key count and sorts
at most 512 selected keys. When the threshold bucket is fully selected,
the selector terminates finer passes exactly: lowering the bucket's
lower bound by one makes the strict-greater gather include the entire
bucket without introducing any additional score.

No score is recomputed. Equal-score keys within each sorted tile are
still encountered in ascending original-index order, and tile order is
ascending, preserving stable ties at the threshold. Selection rewrites
the score row only after gathering all required keys. The existing
emitter, state update, overflow handling, output shapes and workspace
allocation remain unchanged. The original float-score selector keeps
its existing behavior through a compile-time specialization.

The new environment flag `ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=1` is
off by default and cached on the first eligible launch for each type.
It applies only to the existing hierarchical-QSA eligibility:
1-64 packed rows, at least 2048 state blocks, four 128-channel heads,
compression ratio four and a bounded top-K greater than 32 and at
most 512 blocks. Other paths and the default hierarchy are unchanged.
Weights and model files are never edited.

Forty exact 8192/512 Engine requests use the production width-two
overlay and tensor-core flags with the retained scale-hoisting control:

| Comparison | Control E2E TPS | Candidate E2E TPS | Change |
|---|---:|---:|---:|
| Control/candidate/candidate/control, twenty measured requests | 100.546 | 101.753 | +1.200% |
| Candidate/control/control/candidate, independent twenty requests | 100.844 | 101.540 | +0.690% |
| Pooled twenty per variant | 100.695 | 101.646 | +0.945% |

Pooled decode is 126.875 -> 128.494 TPS (+1.276%); mean TTFT is
1.057019 -> 1.060150 s. Rounds are 4470 -> 4442 and evaluated-draft
acceptance is 5744/7805 (73.59%) -> 5769/7806 (73.90%).
Every request emits 512 tokens with zero MTP failures. Retained control
outputs already vary, so the entire observed E2E margin is not assigned
to faster selection.

Matched synthetic graph profiles use thirty warmed replays per range:

| Live blocks | Rows | Control selection us | Radix selection us |
|---:|---:|---:|---:|
| 2048 | 1 | 35.456 | 22.592 |
| 2048 | 2 | 37.424 | 25.344 |
| 2048 | 3 | 42.144 | 27.184 |
| 4096 | 1 | 38.672 | 29.184 |
| 4096 | 2 | 41.536 | 32.896 |
| 4096 | 3 | 45.072 | 32.591 |

Eleven merge kernels become one selector. Timings are median summed
GPU selection durations, excluding host dispatch, copies and setup;
they are not equal-work full-model causal attribution. A version without
the exact early exit only reduced 2048-block selection to
33.983/33.903/35.391 us and was not Engine-benchmarked.

Measured opt-in validation passes 336 FP16/FP32 and 168 BF16 bitwise
graph-boundary cases, plus 336 disabled-mode cases. Each enabled case
checks three graph replays; the earlier selector also passes 56
FP16/FP32 eager cases including 65532 live blocks and 64 packed rows.
The final source widens packed row-stride arithmetic and expands
C++ boundary coverage to 2047/2049/2051 live blocks and one/three rows.
The rebuilt final package again passes all 504 enabled graph cases and
four C++ tests with the flag on, plus four with it off.
VS Code test discovery found no C++ tests, so the actual provider runner
is used; zero-discovery is not counted as validation.

Measured candidate provider SHA256 is
`bb229374ad840ae8e595c7d76e3d597fbb6dbd92ca0a3bed975f45eb8c53db8b`.
Final validation provider SHA256 is
`8452f0c87e2318f31e01a738c6a036902d8a3b68c1f250296f408a6285a99aa2`.
Three additional exact Engine requests on that final package pass with
103.523 E2E / 131.191 decode TPS and zero failures. They are unpaired,
not pooled into the gain estimate and not claimed as a new record.
The first orchestration shell failed after writing all twenty request
records due to a script edit while execution was active; the analyzer
validates every completed record, and the independent repeat/profile
orchestration completes successfully.

Disposition: retain the source locally as an opt-in optimization; no
default flag, original model/runtime replacement or push. Long-context
performance is not established, despite numerical long-context checks.
Evidence is in `indexer-radix-optin-results.json`, separate
`indexer-radix-optin-python` and `indexer-radix-final-python` packages,
Engine request JSONs, matched microprofiles and expanded test logs.
`analyze_indexer_local_merge.py --kind radix-optin --repeat --profile
--final-validation --disposition retained-opt-in` validates protocol,
runtime identities, graph comparisons, final-build execution and results.

#### Sparse indexer local merge fusion (rejected)

The final profile attributed 119.014 ms of summed decode GPU time to
hierarchical indexer merges. An isolated provider fused the first five
32-to-1024-key merge levels into one CTA with shared-memory ping-pong
storage; upper levels and emission remained unchanged. The numerical
scoring, packed sort keys and stable ties were preserved.

All 336 FP16/FP32 cases matched control bitwise, including random/tied
scores, capacities 2051/4097, live-count boundaries around 32/512/1024/2048,
one/two/three rows, empty/overflow outputs and three graph replays.
Twenty exact 8192/512 Engine requests in control/candidate/candidate/control
order measured 101.784 control versus 101.322 candidate E2E TPS (-0.454%).
There were no MTP failures; rounds were 2216 versus 2243, so request
variation also contributes to this comparison.

Matched 30-replay profiles reduced merge launches from eleven to seven.
At 2048 live blocks, median summed merge time was
35.392/37.504/42.048 us control versus 39.792/40.271/41.088 us candidate
for one/two/three rows. At 4096 blocks, it was
38.624/41.664/45.232 versus 42.624/43.520/44.031 us.
One- and two-row regressions make this unsuitable for retention.
The fused kernel/dispatch was removed from active source. Evidence and
the isolated provider remain in `indexer-local-merge-results.json`,
`indexer-local-merge-rejected.cu`, exact Engine request JSONs and matched
microprofiles. No model changes or push were made.

#### Actual-weight INT4 MTP vocabulary head (Engine, not promoted)

The user selected draft-head-only quantization, preserving target
weights, and explicitly confirmed Engine execution. The isolated
`qwen-mtp-int4-head-model` replaces only the MTP `/lm_head/MatMul`
with MatMulNBits: K=2560, N=248320, bits=4, block_size=128,
accuracy_level=0, FP16 scales and symmetric zero point 8. The packaged
quantizer processes the original FP16 weight in bounded column chunks.
This is uncalibrated round-to-nearest quantization, not GPTQ/AWQ.
Packed weights plus scales occupy 327782400 bytes versus 1271398400
bytes for the original head.

The MTP configuration removes only `lm_head.MatMul.weight` from shared
initializers; it otherwise preserves the original configuration, including
ep.cuda.fpa_intb_gemm=1. Target graph bytes, other MTP nodes, original
initializer metadata and graph inputs/outputs are checked unchanged.
External weight files and the target graph are independently copied;
the original model is never edited. The FP16 head remains in the target
external file, so no model-file or peak-memory saving is claimed.

An attempted legacy Generator hidden-state diagnostic failed with
the compact recurrent-state contract before producing any data. It is
excluded from validation. All measured requests use the existing
`benchmark_qwen_mtp.py` Engine path, exactly 8192 uncached input tokens,
512 greedy output tokens, width two, production overlay/tensor flags
and retained scale-hoisting runtime.

| Batch | Control E2E TPS | INT4 E2E TPS | Change |
|---|---:|---:|---:|
| Control/candidate/candidate/control, twenty measured requests | 101.414 | 102.472 | +1.043% |
| Candidate/control/control/candidate, independent twenty requests | 103.036 | 102.884 | -0.147% |
| Pooled twenty requests per variant | 102.218 | 102.677 | +0.449% |

Candidate invocation pools are 101.890, 103.061, 102.450 and 103.322
E2E TPS. Pooled decode throughput is 129.183 control versus 129.937
candidate TPS; mean TTFT is 1.053234 versus 1.053775 s. Evaluated-draft
acceptance is 5799/7755 (74.78%) control versus 5775/7763 (74.39%)
candidate, with 4410 versus 4435 rounds. All measured requests finish
with 512 tokens and zero MTP failures.

A separate full-decode profile confirms the quantized draft path:
M1 MatMulFloat4BitsKernel averages 145.911 us across 277 calls;
M2/M3 MatMulFloat4BatchedKernel averages 169.957/210.815 us across
40/125 calls. The FP16 vocabulary-shaped kernel remains for the target,
averaging 286.949 us across 225 calls. These timings verify dispatch,
not equal-work causal E2E attribution.

The candidate is not promoted: the first gain does not clearly reproduce,
and the pooled margin is small. Draft acceptance is measured only for the
benchmark prompt; no real-hidden-state logit/top-1 agreement dataset or
broader language-quality evaluation is available. Unchanged target
weights do not establish identical emitted tokens, particularly with
existing control output variability. Calibration or broader acceptance
evaluation would be required before treating this as production-ready.

Evidence is retained in `create_mtp_int4_head.py`,
`benchmark_mtp_int4_head.sh`, `analyze_mtp_int4_head.py`,
`mtp-int4-head-results.json`, exact Engine request JSONs, decode profile
and the isolated model/quantization manifest. The analyzer asserts
graph/config scope, runtime hash and library paths, policy, token counts,
draft accounting, zero failures and actual INT4/FP16 mixed dispatch.
Original model/runtime and default benchmark model remain unchanged.
No source optimization or new configuration is pushed.

#### Vector K/V staging, opt-in tensor attention and 100 E2E TPS

The user subsequently approved MTP optimization and confirmed that the
target is 100 **end-to-end** tokens/s including uncached 8,192-token
prefill, not merely 100 decode tokens/s. The user also approved testing
opt-in tensor attention with fixed numerical tolerances rather than
requiring bitwise-identical attention outputs.

The twelve-head prefill path now stages K/V in 16-byte vectors, eight
FP16/BF16 channels per thread. Invalid candidates are zero-filled; source
pointers lacking 16-byte alignment retain scalar loads. Head size 256
and the aligned shared row stride prevent vector overreads and misaligned
shared stores. Short decode/verification and six-/four-head fallbacks
retain their previous staging. With four-row dot-product interleaving,
this default path is bitwise identical in captured full-shape tests and
reduces FP16 attention from approximately 63.2 to 41.3 ms per layer.

The opt-in tensor path pads shared rows to reduce bank conflicts and
shares query/probability storage after QK completes. Four warps handle
all twelve query heads. QK uses FP16/BF16 WMMA with FP32 accumulation.
FP16 PV uses the existing documented `m16n8k16` fragment helpers from
`gated_delta_net_mma.cuh`, FP32 accumulator rescaling in registers and
direct output stores. Softmax/maxima/denominators remain FP32, but
unnormalized PV probabilities are rounded to FP16. BF16 uses tensor QK
only and retains scalar FP32 PV. No model weights or graph nodes change.

These process-local opt-in flags must be set before creating the session:

```bash
export ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
```

They default to false. Dispatch requires the existing twelve-head long
prefill geometry, matching FP16/BF16 caches, single split, SM80+ and
compiled PTX version at least 80. Actual compiled kernel attributes are
queried to check shared-memory fit instead of assuming a compiler's
scratch allocation. Ineligible shapes/devices retain the scalar path.
Enabling PV alone without QK has no effect.

Initial scalar-staging tensor experiments were slower and rejected:
384-thread WMMA PV took about 106 ms FP16; four-warp WMMA PV about 74 ms;
padding/register-only MMA alone about 64 ms. Vector staging was essential.
The final vector/tensor FP16 path takes about 20.1 ms per full-8K layer.
The actual model prefill profile confirms twelve tensor QK/PV launches
totaling 238.13 ms, versus 754.66 ms for scalar-staged interleaved attention.
Instrumented request wall times are excluded from throughput.

Validation with the final attention provider passes 192 standalone
numerical-reference cases (96 default and 96 opt-in), plus sixteen
full-8K CUDA graph replay comparisons: FP16/BF16, first/last captured
layer selections, default/opt-in, aligned/misaligned current K/V.
The default vector path is bitwise equal; opt-in maximum absolute
differences are 0.0009765625 FP16 and 0.001953125 BF16, within the
unchanged reference tolerances. This is operator-level validation and a
single-prompt performance workload, not a general model-quality or
lossless-output evaluation. Tensor attention can change target logits;
keep it opt-in until deployment-specific accuracy validation.

MTP draft widths three and four were tested on the retained interleaved
provider before the tensor experiment. Five measured requests each gave
89.38 and 85.64 E2E TPS, with 1.64 and 1.79 accepted drafts per round,
respectively. Wider chains reduce target passes but add too much draft
work here; width two remains the selected policy. All runs retain exact
token counts and zero MTP failures.

The existing GEMM tuner is additionally enabled explicitly for the MTP
session in `qwen_width2_mtp_gemm_tune_overlay.json`. Previously only the
base decoder was tuned. Six M-one/two/three cases cover the two MTP
projection shapes absent from the prior 36-case base validation:
`(K=10240,N=320)` and `(K=2560,N=2560)`. Both retain cuBLAS according to
verbose tuner timings and pass numerical comparison. Other eligible
small MTP projections may use tinygemm2. Tuning stays opt-in, and unseen
shapes can still trigger real tuning outside CUDA capture. The first
five-request tuned-head measurement was 100.0135 E2E TPS, versus 99.5805
for the reverse-order untuned-head comparison; this small difference
alone is not conclusive evidence of a head-kernel speedup.

The final MTP-specific source optimization removes repeated `ldexpf`
work from E4M3FN weight conversion: normal FP8 values are assembled
directly as their exact FP32 exponent/mantissa bits; subnormals, signed
zeros and the original canonical NaNs are preserved. The same helper
also serves related NVFP4 scale/dequantization paths. All 256 codes pass
an independent exhaustive SM90 GPU bit comparison, including both NaN
codes and signed zero. The test compiles the actual source function via
NVRTC to native cubin after the local driver rejected the PTX module.
Thirty-six actual-shape FP8 MTP QMoE cases
(FP16/BF16 activations, FP32/FP16/BF16 scales, one/two/three rows,
optional supported down-projection bias) and 24 NVFP4 QMoE cases match
the prior provider bitwise. Unsupported separate-FC3 gate/up bias cases
were rejected before measurement; the validator was corrected to use
supported down-projection bias. The related C++ exhaustive-code
regression is added but not compiled/run; editor discovery finds no tests.

Final paired exact-workload comparisons use width two, base+MTP GEMM
tuning, both tensor flags, greedy batch one, original graphs/weights,
8,192 input / 512 output tokens, CUDA graphs and the established cache
budget. Each invocation excludes one warmup and measures five requests.
Provider order is reversed on repetition, and all measured samples remain.

| Runtime | First E2E TPS | Repeat E2E TPS | Pooled E2E TPS | Pooled decode TPS | Mean TTFT (s) | Mean request (s) |
|---|---:|---:|---:|---:|---:|---:|
| Vector/tensor + head tuning, previous FP8 decoder | 99.660 | 99.216 | 99.437 | 124.910 | 1.0580 | 5.1490 |
| Same configuration + exact FP8 bit decoder | 100.167 | 100.798 | **100.481** | 126.419 | 1.0533 | **5.0955** |

The target is passed in both five-request invocations and the ten-request
pool: 5,120 output tokens divided by 50.9547 seconds. Candidate requests
range from 98.22 to 104.92 E2E TPS; this is an average-throughput result,
not a per-request 100-TPS guarantee, and the margin above target is small.
The two providers have nearly identical aggregate acceptance
(2,889 versus 2,892 accepted drafts); timing/greedy variation remains.
Do not assign all of the 1.05% E2E comparison delta to FP8 conversion alone.
Exact prompt hash, tokens, flags, overlays, loaded libraries, actual draft
width and zero MTP failures are assertion-checked. A final 64-token decode
capture also validates the full 512-token request; its profiled wall TPS
is not included in the throughput pool.

In that decode capture, cumulative kernel times are 192.53 ms for
recognized dense-projection/cuBLAS kernels, 119.00 ms for fused NVFP4
MoE projections, 73.70 ms for sparse attention, 34.68 ms for the sparse
indexer, 21.15 ms for GatedDeltaNet and 17.17 ms for fused FP8 MTP MoE
projections; other kernels account for 196.48 ms. These are profiled
GPU-duration sums, not additive components of unprofiled request wall
time. Fused projection timings include dequantization and multiplication
and do not independently establish the fraction spent on dequantization.
The persisted report includes the fifteen largest individual kernels.

The final isolated runtime is `mtp-fp8-decoder-python`, provider SHA256
`103a10ceae4aaac2ae5dbdb0c6f40c35c012dcc3a8dfd1491b9a5fb564e035d2`.
Installed packages, previous isolated runtimes and original model files
remain unchanged. The session artifact `run_qwen_100_e2e.sh` reproduces
the exact configuration with ten measured requests. Assertion-checked
results are in `qwen-100-e2e-results.json`; scripts, logs and profiles
persist. Temporary standalone reference arrays are removed after validation.

The paired-channel implementation resolves each candidate once for two output
channels, preserving each channel's candidate accumulation order. The
specialization alone did not improve performance. Before paired accumulation,
the optimized full-8K prefill profile attributed approximately 79% of summed
GPU kernel time to sparse attention and 3.4% to NVFP4 dequantization.

The grouped tiled prefill path stages 64 selected K/V rows in shared memory and
reuses them across four query heads belonging to the same KV head. Key storage
is reused for values after all dot products finish. This path requires head size
256, FP16 or BF16 queries and matching non-quantized caches, selected-only
main-cache attention, a GQA ratio divisible by four, and sufficient per-block
shared memory. Other configurations retain the existing implementation. Candidate
resolution is shared with the fallback, preserving causal filtering, invalid
block/slot handling, and direct current-token reads.

Compared with paired accumulation, end-to-end throughput improved by about 16%
and TTFT fell by about 40%. Nsight confirmed all 12 sparse-attention prefill
launches used the grouped tiled kernel. Its summed time was approximately
1.47 seconds (60% of summed GPU kernel time), compared with 3.69 seconds in the
earlier profile before paired accumulation. This profile comparison is not an
isolated tiled-versus-paired comparison. Profile-instrumented wall-clock timings
are not used in the throughput table.

The grouped kernel now also supports candidate splits and single-token decode.
Dispatch eligibility is shared between scratch allocation and kernel launch.
The split-count heuristic accounts for four heads per block and 64-row tiles,
with the existing 32-split cap. On this H200, decode uses six head groups and 32
splits rather than 24 individual heads and 11 splits; the existing reduction
kernel merges the per-head FP32 partials and applies head sinks.

Graph-node-aware Nsight captures confirmed the grouped decode path. Across 804
sparse-attention launches, combined attention/reduction GPU time fell from
approximately 178 milliseconds to 117 milliseconds (34%). Reduction time
increased with the larger split count, but was outweighed by the attention-kernel
improvement. CUDA graph nodes must be explicitly traced to include replayed
kernels; the default trace included only eager launches and was not used for
this comparison.

The packed QSA indexer's hierarchical decode path now scores 32 blocks per tile
instead of eight. Each scoring warp reuses its query vector across four blocks,
and a warp bitonic sort replaces serial insertion sorting. Fully padded tiles
write padding keys without loading queries or scoring. The existing stable
score/index packing and merge-rank implementation are unchanged. Scratch sizing,
key strides, and merge starting widths use the new tile size; ragged prefill
continues to use its existing eight-block scoring tiles.

At this model's 65,536-block state capacity, Nsight confirmed score grids shrank
from 8,192 to 2,048 CTAs and merge stages fell from 13 to 11 per indexer call.
Across 804 calls, combined score/merge time fell from approximately 95.06 to
59.37 milliseconds (38%). Eight standalone A/B cases produced bitwise-identical
indices, counts, key state, buffer state, and state lengths, covering two
requests, capacities 2,051 and 65,536, zero live blocks, 31/32/33-block tile
boundaries, partial tiles, and stable score ties. FP16/BF16 stable-tie regression
coverage was also added and compiled, but not run as a C++ suite.

Hierarchical merging now checks each sorted list's first key for the padding
sentinel. When either list is empty, it copies the other list and pads remaining
positions instead of binary-searching each output rank. When both lists contain
valid keys, the existing rank merge is unchanged. Eight standalone A/B cases
again matched all outputs bitwise. Stable-tie regression inputs also cover
511/512/513 and 1,023/1,024/1,025 live blocks around merge boundaries.

Fusing the first five merge stages into a shared-memory kernel was tested and
removed: it passed the same A/B checks but reached only 73.89 decode TPS and
54.43 end-to-end TPS, below the simpler padding-aware implementation. The final
source and staged provider retain only the padding-aware merge change.

Three NVFP4 row-major GEMV experiments were evaluated separately on top of the
indexer change:

| GEMV experiment | Decode TPS | End-to-end TPS | Decision |
|---|---:|---:|---|
| Shared activation staging | 73.38 | 54.14 | Removed; negligible gain |
| Eight-column, 64-thread tiles for narrower projections | 73.48 | 54.19 | Removed; no convincing gain |
| Paired output columns per thread | 71.92 | 53.35 | Removed; throughput regression |

All three passed 12 bitwise-output A/B checks spanning FP16/BF16, bias on/off,
one/two input rows, and hidden widths 1,280, 2,560, and 4,352. No GEMV experiment
is retained in the final source or staged runtime. The retained throughput gain
in this iteration comes from the sparse indexer, not NVFP4 GEMV.

Benchmarks run in tmux using a copied ORT Python package with the locally built
CUDA provider; loaded library paths and exact token counts are checked.
Installed conda packages and the model package are unchanged. These are
experimental throughput results, not a lossless-generation claim: repeated
greedy outputs also varied in unchanged baseline runs. Earlier sparse-attention
and NVFP4 value tests passed. The tiled path passed 16 FP16 and 16 BF16 standalone
numerical-reference checks covering two requests, two KV heads, cached/current
tokens, tile tails, empty/all-invalid selections, duplicate selections, causal
and non-causal modes, softcap, head sinks, and optional slot mappings. FP16 checks
use absolute/relative tolerances of 0.002; BF16 uses absolute tolerance 0.002 and
relative tolerance 0.008. The grouped split extension passed those checks again
plus 16 FP16 and 16 BF16 single-token decode checks (64 cases total). Maximum
single-token absolute error was approximately 0.000061 for FP16 and 0.000479 for
BF16. Changing the softmax tile size or split boundaries can change rounding.
Persistent C++ prefill and single-token regression coverage was added and
compiled, but the C++ test suite was not rerun for this iteration. A broader
QMoE test run has one unresolved
mixed-width fallback validation failure.

The 100 end-to-end TPS target requires at most 5.12 seconds per request.
The latest decode interval alone is approximately 6.90 seconds, so reducing
prefill alone cannot meet this target.

#### Bounded sparse-indexer state replay (opt-in)

The GenAI CUDA replay kernel ordinarily scans the full fixed capacities of
the sparse-indexer key and auxiliary buffers. With
`ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1`, kind-4 replay copies only the
active key prefix and new auxiliary-buffer prefix. This relies on the
shared indexer state being append-only: separately committed lengths keep
inactive tails invisible until a later transition overwrites them.
The flag is disabled by default; other state kinds keep their existing
replay path.

Validation used separate control and candidate Engine packages with
byte-identical ORT and GenAI libraries. Two reversed-order batches each
measured ten exact 8,192-input / 512-output requests on one H200 with
width-two MTP, the production overlay and sparse-prefill tensor-core flags.
The pooled control reached 101.117 E2E / 127.354 decode TPS; the candidate
reached 102.803 / 130.119 TPS (+1.67% E2E). Mean TTFT was 1.051 versus
1.053 seconds. All requests generated 512 tokens without MTP failures.
Acceptance changed from 73.58% to 74.49% (4,462 to 4,421 rounds), so the
whole request-level gain is not assigned to replay.

A separate matched decode profile measured 100
`ReplayStateUpdatesKernel` calls at 54.56 ms total for control and 3.68 ms
for the candidate (545.6 versus 36.8 us/call, 93.3% lower). The four
`CudaFixedStatePoolTest.*` cases pass, including active-key and
active-buffer updates and preservation of inactive destination tails.
Paired output IDs are not identical, while independent repeats within
each variant also diverge early; no full-sequence or model-quality
equivalence claim is made. The optimization remains opt-in.

### Historical INT4 export

The inspected export uses:

- INT4 `MatMulNBits` weights.
- An INT4 LM head (`lm_head.MatMul.weight_Q4`), not INT8.
- FP16 model inputs, outputs, K/V cache, hidden states, and indexer state.
- A separate `mtp.onnx` model.
- A 48 GB `engram.onnx.data` external initializer.
- No persisted `search.chunk_size`; benchmarks set `chunk_size=1024` at runtime.

## Issues

### 1. MTP export and speculative state correctness

Greedy one-draft MTP matches greedy baseline output on the short validation path. It is not
universally token-identical: at several coding-context lengths, a sequence-2 verification forward
diverges from sequential target decoding after otherwise matching prefixes. The replay fallback
exhibits the same issue, so this remaining difference is caused by batched verification numerics
rather than accepted-prefix state commit.

#### Root cause

There were two independent correctness problems:

- `select_projection_outputs()` shallow-copied a PyTorch module. The copy shared its `_parameters` dictionary with the source, so selecting Q rows mutated the original projection and the subsequent gate selection also used Q rows.
- Rejection rollback did not restore all attention, recurrent, convolution, PLE, sparse-indexer, and external Engram state.

The projection helper now copies `_parameters` and `_buffers` before selecting rows. Runtime rollback supports every target state surface.

#### Direct accepted-prefix commit

Qwen's `GatedDeltaNet` does not expose a full recurrent-state window. Instead, the target captures a compact first-token transition capsule. On rejection, the runtime:

1. promotes the first-token convolution and PLE convolution state from a two-slot window,
2. replays the compact GatedDeltaNet transition from the pre-verify snapshot,
3. commits the first verified token to external Engram history,
4. crops K/V and sparse-indexer state to the accepted target length,
5. uses verify-row-zero logits and hidden state directly.

This removes the former full target replay. Exact greedy output still requires a verification
kernel whose row-zero numerics match a sequence-1 target forward at every context length.

### 2. MTP configuration and runtime contracts were incomplete

The Qwen3.8 target exposes a 10,240-wide hyper-connection state, while the normal decoder hidden size is 2,560. MTP handoff buffers were allocated using the decoder hidden size instead of the actual ONNX tensor width.

The MTP graph also requires:

- `past.%d.indexer_key`, not `past_key_values.%d.indexer_key`,
- `past_sequence_length` for the fixed shared indexer cache.

#### Required fix

- Derive MTP handoff width from target output and MTP input metadata.
- Validate matching static widths and data types.
- Export and parse the correct indexer name template.
- Export and project `past_sequence_length`.
- Support CPU- and CUDA-backed hidden-state outputs without invalid device-to-device copies.

### 3. MTP output projection remains workload-sensitive

Before direct prefix commit, every rejected draft reran the accepted token through all 48 target layers. After the fix, a warmed one-draft natural-language test reports:

```text
Baseline: 40.0 tokens/s
MTP:      84.8 tokens/s
Acceptance: 70.3%
Target forwards: 37 for 64 emitted tokens
```

The remaining cost is the MTP model's vocabulary projection. On a synthetic 1K coding prompt, 93.9% acceptance is lossless but throughput improves only from 38.4 to 46.2 tok/s (1.20x), because each round still executes the large draft LM head. Further gains require a faster quantized output-projection kernel, safe overlap of draft and target work, or verified n-gram proposals that skip draft-head execution.

#### Depth 2-3 direct commit

The compact commit path now scales with `state_window`: a four-slot convolution/PLE window and
three captured GatedDeltaNet transitions support up to three draft tokens. On the synthetic 1K
coding prompt with corrected INT4 MTP:

| Draft depth | Generated tokens | Baseline | MTP | Speedup | Acceptance | Exact match |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 64 | 38.0 tok/s | 46.7 tok/s | 1.23x | 88.9% | yes |
| 3 | 64 | 37.2 tok/s | 48.7 tok/s | 1.31x | 92.0% | yes |
| 3 | 400 | 50.9 tok/s | 69.5 tok/s | 1.37x | 78.3% | no |

The 400-token result demonstrates sustained performance potential, but it is not production-valid
until wide target verification is numerically consistent with sequential greedy decoding. MTP-head
CUDA graph capture was also tested and did not help reliably; depth 2 changed draft behavior and
depth 3 regressed sustained throughput.

The full 64-token coding-context sweep used fixed 1K prefill chunks and depth-3 INT4 direct commit:

| Context | Baseline decode | MTP decode | Relative | Acceptance | Exact match |
|---:|---:|---:|---:|---:|:---:|
| 1K | 38.30 tok/s | 49.48 tok/s | 1.29x | 92.0% | yes |
| 2K | 37.57 tok/s | 20.46 tok/s | 0.54x | 58.6% | no |
| 4K | 37.92 tok/s | 21.71 tok/s | 0.57x | 51.7% | no |
| 8K | 38.79 tok/s | 22.53 tok/s | 0.58x | 63.6% | yes |
| 16K | 38.60 tok/s | 37.45 tok/s | 0.97x | 70.4% | no |
| 32K | 38.26 tok/s | 17.25 tok/s | 0.45x | 47.5% | no |
| 64K | 36.94 tok/s | 18.94 tok/s | 0.51x | 58.2% | no |
| 128K | 36.54 tok/s | 14.98 tok/s | 0.41x | 9.7% | no |
| 162K | 36.73 tok/s | 23.88 tok/s | 0.65x | 45.2% | no |
| 216K | 35.73 tok/s | 16.34 tok/s | 0.46x | 11.3% | no |

Depth 3 therefore cannot be enabled globally. It needs acceptance-aware admission and exact
wide-verification handling; otherwise low-acceptance contexts spend more time in serial draft-head
forwards than they save in target forwards.

#### Sparse-indexer rollback root cause

Debugging at 2K isolated two independent correctness constraints:

1. Shared QSA indexer state was only logically rewound. A wide speculative verification overwrites
   completed block representatives and the four raw-key scratch rows in place, so later decoding
   consumed rejected-token state. A bounded snapshot now preserves the next representative, the
   scratch rows, and speculative append rows for every sparse-attention layer. Forced sequential
   reconciliation became token-identical after this fix.
2. Direct accepted-prefix commit is still unsafe for QSA. GatedDeltaNet, convolution, PLE, KV, and
   Engram have a window or compact transition representation, but the sparse indexer does not retain
   the state after each verified token. `HasCroppableRecurrentState()` therefore returns false when
   an indexer cache is present.

The conservative exact 2K depth-3 path reaches 21.9 tok/s versus 37.4 tok/s baseline because it
sequentially reconciles every speculative round. A performant exact implementation requires compact
per-token QSA indexer transitions (or bounded rollback planes like llama.cpp), followed by a
numerically consistent target verification kernel. Margin-based replay is insufficient because
small state differences can accumulate before they change the current top-1 token.

#### ORT QSA transitions and warmed throughput

ORT now optionally emits the compact QSA cache-row value written by each verified token. GenAI
restores the pre-verify representative/scratch rows and applies only accepted transitions, allowing
direct prefix commit without target replay. Transition capture is fused into the existing raw-tail
and block-representative kernels, so it adds no separate CUDA kernel per sparse layer. The MTP head
also snapshots its own indexer before speculative chaining.

The first M=2/M=4 decode invokes CUDA MoE tactic profiling. Fair steady-state measurements therefore
warm both baseline M=1 and MTP verification shapes before timing, matching the warm-up-discarded
llama.cpp methodology.

Warmed depth-1 coding results:

| Context | Baseline | MTP | Relative | Acceptance | Exact match |
|---:|---:|---:|---:|---:|:---:|
| 1K | 54.99 tok/s | 88.52 tok/s | 1.61x | 93.9% | yes |
| 2K | 53.53 tok/s | 67.14 tok/s | 1.25x | 53.7% | no |
| 4K | 54.85 tok/s | 69.57 tok/s | 1.27x | 56.1% | yes |
| 8K | 55.55 tok/s | 76.72 tok/s | 1.38x | 65.8% | yes |
| 16K | 55.08 tok/s | 66.13 tok/s | 1.20x | 46.5% | no |
| 32K | 54.45 tok/s | 64.61 tok/s | 1.19x | 57.5% | yes |
| 64K | 54.40 tok/s | 71.61 tok/s | 1.32x | 65.8% | no |
| 128K | 53.41 tok/s | 34.61 tok/s | 0.65x | 34.0% | no |
| 162K | 52.68 tok/s | 61.09 tok/s | 1.16x | 80.0% | no |
| 216K | 50.19 tok/s | 28.36 tok/s | 0.57x | 12.5% | no |

Depth 3 is preferable at 128K, where warm throughput is 88.42 tok/s versus 52.22 baseline (1.69x)
at 80% acceptance. No depth can guarantee 1.2x at 216K with the current trained head: depth-1
acceptance is 12.5% and depth-3 acceptance is 6.5%, placing the ideal zero-overhead emitted-token
bound below 1.2x. That context needs a higher-quality/retrained draft head or a different proposer;
runtime kernel optimization cannot overcome rejected predictions.

### 4. The Engram lookup prevents target CUDA graph capture

The current decoder contains:

```text
input_ids
  -> NGramHashMapping
  -> GatherBlockQuantized (CPU)
  -> PLE projections and gate (CUDA)
```

The 48 GB Engram table is intentionally kept on CPU. The gather node is annotated with `layer_ann=cpu_embedding` and assigned to CPU through:

```text
session.layer_assignment_settings=cpu(=cpu_embedding)
```

ONNX Runtime CUDA graph capture requires the session compute graph to be fully assigned to CUDA. The CPU lookup and its CPU-to-CUDA boundary therefore prevent target graph capture, even though the table itself is constant.

#### Required fix

Move the n-gram mapping and Engram gather into a separate CPU model:

```text
engram.onnx (CPU)
  input_ids + committed token history
  -> NGramHashMapping
  -> GatherBlockQuantized
  -> engram_embeddings

text.onnx (CUDA)
  engram_embeddings
  -> PLE projections, gate, convolution
  -> decoder
```

The resulting FP16 payload is approximately:

```text
2,560 values * 2 bytes = 5 KiB/token
```

This keeps the 48 GB table out of HBM while allowing `text.onnx` to become CUDA-only.

### 5. Long-context prefill consumes excessive workspace

The non-paged export completed through 22K tokens only when MTP's explicit 1K chunked prefill path was used. Ordinary single-shot prefill reached approximately 139 GiB at 22K and failed at larger contexts.

The sequence cache is not the dominant allocation. The main contributors are:

- model/session residency,
- CUDA weight transforms,
- activation workspace,
- CUDA arena retention,
- large unchunked prefill tensors.

#### Required fix

- Make chunked prefill a common target-model capability rather than an MTP-only optimization.
- Reuse bounded per-chunk buffers.
- Avoid retaining maximum transient allocations in the CUDA arena where possible.
- Revisit paged attention after Qwen3.8 sparse-index selection is supported by the runtime.

## Proposed Engram split

### New `engram.onnx`

Inputs:

```text
input_ids          INT64 [batch, sequence]
past_ple_tokens    INT64 [batch, ngram_size - 1]
```

Outputs:

```text
engram_embeddings  FP16 [batch, sequence, 2560]
present_ple_tokens INT64 [batch, ngram_size - 1]
```

Contents:

- `NGramHashMapping`,
- `GatherBlockQuantized`,
- n-gram constants,
- `engram.onnx.data`,
- Engram quantization scale.

### Revised `text.onnx`

Remove:

- `NGramHashMapping`,
- `GatherBlockQuantized`,
- the Engram initializer,
- the CPU layer annotation,
- PLE token-history bindings.

Add:

```text
engram_embeddings FP16 [batch, sequence, 2560]
```

Keep the following on CUDA:

- PLE key projection,
- PLE value projection,
- Engram gate,
- PLE convolution,
- hyper-connection injection.

### Runtime `EngramState`

Add an `EngramState` component that:

1. Runs the CPU Engram model before each decoder invocation.
2. Tracks the last `ngram_size - 1` committed tokens.
3. Copies the Engram result to a stable CUDA input buffer.
4. Supports prompt chunking.
5. Supports speculative verify blocks.
6. Commits token history only after draft acceptance is known.

For Qwen3.8, `ngram_size=3`, so only the last two committed token IDs are required. Deriving history from the committed sequence is preferable to maintaining another rewindable tensor.

## Recommended implementation order

1. Implement multimodal decoder snapshot, rewind, and accepted-prefix commit.
2. Make MTP greedy output token-identical to baseline.
3. Add regression tests for accepted and rejected drafts.
4. Add the `engram.onnx` exporter.
5. Add `model.engram` configuration parsing.
6. Add runtime `EngramState`.
7. Replace the decoder's internal gather with `engram_embeddings`.
8. Verify that every `text.onnx` node is assigned to CUDA.
9. Enable CUDA graph capture for decode and verify shapes.
10. Re-run accuracy, acceptance, throughput, memory, and long-context tests.

## Completion criteria

The work is complete when:

- greedy MTP output matches greedy baseline output exactly,
- reset and rejection restore every state component,
- `text.onnx` has no CPU nodes,
- target CUDA graph capture succeeds,
- the Engram table remains CPU-resident or disk-mapped,
- long-prompt chunked prefill stays within the H200 memory budget,
- MTP provides a repeatable throughput improvement on representative prompts.
