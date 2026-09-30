# Qwen3.8 Flash Next: MTP and Engram Issues

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
