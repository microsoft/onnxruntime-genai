# Engine Batching Design

Status: "Phase 1" is [PR #2343](https://github.com/microsoft/onnxruntime-genai/pull/2343);
"Phase 2" is [PR #2345](https://github.com/microsoft/onnxruntime-genai/pull/2345), stacked on it.
"Phase 3" is [PR #2361](https://github.com/microsoft/onnxruntime-genai/pull/2361), stacked on Phase 2.
All three are described here as built.

This document explains how batching works in the two generation paths in onnxruntime-genai, why
they differ, and what was changed to close most of the remaining throughput gap between them.

---

## 1. Two batching designs

onnxruntime-genai has two independent ways to run a batch of sequences.

### Design A — `Generator` (static batch)

The classic path, used by models whose decoder takes 2-D `input_ids` and uses
`GroupQueryAttention` or `MultiHeadAttention`.

One `Generator` owns exactly **one** `Search` object that covers the whole batch
(`src/generators.h`, `std::unique_ptr<Search> search_`). Everything is sized once at construction:

```cpp
// src/cuda/search_cuda.cpp — GreedySearch_Cuda constructor
next_tokens_buffer_ = params.p_device->Allocate<int32_t>(params.search.batch_size);
...
samplingdata_ = std::make_unique<cuda::SamplingData>(random_seed, params.search.batch_size,
                                                     params.config.model.vocab_size, GetStream(), ...);
```

Token selection is one call over the whole batch:

```cpp
// src/cuda/search_cuda.cpp — GreedySearch_Cuda::SampleTopKTopP
cuda::GetSample(samplingdata_.get(), GetStream(), next_tokens_.data(), scores.data(),
                int(scores.size() / params_->search.batch_size),
                params_->search.batch_size, k, p, temperature);
cuda::Launch_CheckForEOSAndPad(next_tokens_.data(), static_cast<int>(next_tokens_.size()), ...);
cuda::Launch_AppendNextTokensToSequences(next_tokens_buffer_.Span(), sequences_.GetSequences().Span(),
                                         params_->BatchBeamSize(), sequences_.GetSequenceLength(), ...);
```

Three kernel launches and one synchronization for the entire batch, regardless of its size. That
is why the `Generator` path is fast.

The price is rigidity:

- **All search parameters are shared.** `Config::Search` (`src/config.h`) holds a single
  `max_length`, `min_length`, `top_k`, `top_p`, `temperature`, `repetition_penalty`,
  `no_repeat_ngram_size`, `num_beams`, `random_seed` for the batch. There is no per-row variant of
  any of them.
- **One sequence-length cursor for the batch.** `Sequences` (`src/sequences.h`) stores a single
  `current_length_`, and `GetSequence(i)` is `sequences_.subspan(i * max_length_, current_length_)`.
  Every row is at the same length by construction, and the append kernel writes every row at the
  same `past_length` offset.
- **Sequences that finish early are padded, not evicted.** `CheckForEOSAndPad` marks a row in
  `eos_seen_` and overwrites its token with `pad_token_id`; `*done_cpu` is set only when *all* rows
  have finished. The batch keeps burning full-width compute until the longest sequence ends.
- **The batch is fixed.** `eos_seen_`, `sequence_lengths_`, `sequences_` and the sampling
  workspace are all allocated for `BatchBeamSize()` at construction. `Search` has `AppendTokens`
  and `RewindTo`, but nothing to add or remove a row from a batch in flight.

That is fine for offline batch scoring, where you hand over N prompts at once and wait for all N.
It is a poor fit for a server.

### Design B — `Engine` (continuous batch)

The path used by `PagedAttention` models. Requests arrive and depart independently; each step the
scheduler picks whichever requests are runnable and forms a batch out of them.

Each `Request` owns its own `GeneratorParams` and its own single-sequence `Search`:

```cpp
// src/engine/request.cpp
Request::Request(std::shared_ptr<GeneratorParams> params)
    : params_{params}, search_{CreateSearch(*params.get())} {
  search_->DeferCompletion(true);
}
```

The decoder IO is chosen from the cache manager, which is chosen from config:

```cpp
// src/engine/cache_manager.cpp
if (model->config_->engine.dynamic_batching)
  return std::make_unique<PagedCacheManager>(model);   // SupportsDynamicBatching() == true
return std::make_unique<StaticCacheManager>(model);    // SupportsDynamicBatching() == false

// src/engine/decoders/simple_decoder.cpp
cache_manager_->SupportsDynamicBatching()
    ? std::make_unique<VarlenDecoderIO>(...)      // 1-D input_ids[total_num_tokens], PagedAttention
    : std::make_unique<StaticBatchDecoderIO>(...) // 2-D input_ids[batch, seq], GQA/MHA
```

So the `Engine` already supports both a varlen and a static-batch decoder. What it does *not* have
is a batched **search**: `ScheduledRequests::GenerateNextTokens()` walks the requests one at a
time, because each one has its own `Search`.

### Comparison

| | Design A — `Generator` | Design B — `Engine` |
|---|---|---|
| `Search` objects for N sequences | 1 | N |
| Batch membership | fixed at construction | changes every step |
| Sequence length | one shared cursor | per request |
| Prompt lengths | padded to equal length | ragged, no padding (varlen) |
| `max_length`, EOS set, `top_k`/`top_p`/`temperature` | shared by the batch | per request |
| Early finishers | padded until all finish | evicted, slot reused immediately |
| Sampler launches per step | 3 | 3N |
| Device synchronizations per step | 1 | N |
| KV cache | contiguous, `[batch, max_length]` | paged blocks, allocated on demand |
| `input_ids` | `int64[batch, seq_len]` | `int64[total_num_tokens]` + `cu_seqlens` |

---

## 2. Why the Engine cannot simply adopt the Generator design

The natural question is: the `Generator` path is faster, so why not use its batching design for
`PagedAttention`?

The answer is that the `Generator` design is fast **because** of the assumptions it makes, and
those assumptions are exactly the ones continuous batching exists to break. Adopting it wholesale
would mean giving up:

1. **Independent arrival and departure.** A server admits a request at step *k* and retires it at
   step *k+n*. `Search` allocates `eos_seen_`, `sequences_`, `sequence_lengths_` and the sampling
   workspace for a fixed `BatchBeamSize()` and has no resize path. Joining an in-flight batch is
   not expressible.
2. **Per-request generation parameters.** Two clients calling the same server may pass different
   `max_length`, `temperature`, `top_k`, or stop tokens. `Config::Search` has exactly one of each.
3. **Ragged sequences.** `Sequences` has a single `current_length_` and the append kernel writes
   every row at the same offset. Requests in a continuous batch are at different lengths by
   definition — that is the entire point. `VarlenDecoderIO` is built around this: it flattens
   variable-length token runs into one 1-D tensor and describes them with cumulative sequence
   lengths.
4. **Prefill/decode mixing.** A single step can contain one request prefilling 800 tokens and
   thirty requests decoding one token each. The shared-cursor model cannot represent that.
5. **Early eviction.** Padding finished rows until the longest finishes wastes a large fraction of
   a server's throughput. Evicting them and admitting new work is the main reason continuous
   batching wins.

Also, mechanically, `PagedAttention` models cannot run under `Generator` at all: they need 1-D
`input_ids` plus `cu_seqlens`, and `DefaultInputIDs` (`src/models/input_ids.cpp`) always produces
`{BatchBeamSize(), sequence_length}`. This is why `determinism_test.py`, which uses `og.Generator`,
fails on the paged model with `Invalid rank for input: input_ids Got: 2 Expected: 1`.

**But the question is still the right one**, because it points at the real answer: the Engine does
not need a different batching design *for the sampler*. It needs continuous batching for the
*model* and Generator-style batching for the *tail*. Those are separable, and separating them is
what this document proposes.

---

## 3. Where the cost actually is

Profiling gpt-oss-20b (INT8 KV) on H200 at batch 32 showed the gap between the two paths was **not**
in the attention operator. Per decode step:

| Per step | `Engine` (before) | `Generator` |
|---|---|---|
| `cudaStreamSynchronize` | 130 | 4 |
| Tail `cudaLaunchKernel` | 160 | 7 |
| `cudaMemcpyAsync` | 100 | 1 |
| GPU time in sampling | 451 µs | 47 µs |

The decoder graph body was essentially identical. The entire regression was the per-request tail:
32 sequential sampler invocations, each one blocking the host on the device before the next could
launch.

A tempting first fix — deleting the synchronizations — was tried and **produced no speedup**
(4056 vs 4109 tok/s). The synchronization primitive was never the cost. The cost was the
serialization it forced: launch request *i*, block, launch request *i+1*. The device sat idle
between tiny kernels while the host walked the list.

---

## 4. Phase 1 ([PR #2343](https://github.com/microsoft/onnxruntime-genai/pull/2343)): defer completion

PR #2343 split *launching* token selection from *completing* it. Each request still runs its own
sampler, but the device-dependent part is deferred, so all requests launch before anyone waits.
Since all requests share one CUDA stream, one wait covers all of them.

Two no-op-by-default virtuals were added to `Search` (`src/search.h`):

```cpp
virtual void DeferCompletion(bool /*defer*/) {}
virtual void CompleteGeneration() {}
```

`ScheduledRequests::GenerateNextTokens()` became two loops — launch all, then complete all — and
`Request` gained a host-side mirror of its sequence (`tokens_host_`) so the decoder input builders
no longer copy tokens back from the device each step. `VarlenDecoderIO::ProcessLogits()` also does
a single batched `Cast` when the logits rows are already contiguous, which they are on any
all-decode step.

Result at batch 32: 4109 → 4517 tok/s. Syncs 130 → 32, tail launches 160 → 129, tail time
2.384 → 1.423 ms.

Review of PR #2343 added three hardening changes that Phase 2 builds on:

- `Request`'s constructor now rejects `search.batch_size != 1` and `search.num_beams != 1`. A wider
  search would have mirrored the wrong row's tokens (several places read row 0 while
  `CompleteGeneration()` took the tail of the next-token span), and beam search never overrides the
  deferred-completion contract, so its next tokens were never copied back. Phase 2's gate can
  therefore assume single-row greedy searches instead of re-checking per step.
- `UnprocessedTokensCpu()`'s documented lifetime was corrected: the span points into `tokens_host_`
  and is valid only until the next call that appends to the sequence.
- The `cudaStreamSynchronize` in `Search_Cuda::IsDone()` is now `CUDA_CHECK`ed. Clearing
  `done_pending_` after a failed sync would make the next `IsDone()` skip the wait and read a stale
  `done_cpu_`.

What Phase 1 did **not** do is reduce the amount of sampling work. There are still 32 separate
single-row `GetSample` invocations costing ~451 µs of GPU time to do what one batched call would do
in ~50 µs.

---

## 5. Phase 2 ([PR #2345](https://github.com/microsoft/onnxruntime-genai/pull/2345)): batched sampling fast path

### 5.1 Key insight

The reason the Engine needs per-request `Search` objects is **lifecycle**: ragged lengths, per-request
params, join/leave. None of that applies to the sampler itself. On a pure decode step, sampling is a
pure function of a `[batch, vocab]` logits tensor and a set of sampling parameters. If every request
in the step happens to share those parameters — which is the overwhelmingly common server case — the
sampler can run **once** for the whole batch, exactly as in Design A.

So: keep per-request `Search` objects for state, but let the *sampling* be batched when it is safe.
This is "follow the GQA batching design" applied to the one place where it actually pays, without
inheriting any of the constraints that make it unusable for a server.

### 5.2 Eligibility gate

The fast path is taken only when all of the following hold for the current step. Otherwise the
existing two-phase per-request loop runs unchanged.

- At least two scheduled requests, none of them already `Completed`. A completed request is skipped
  by the per-request loops, which would leave a hole in the logits rows that a single batched
  sampler call cannot express.
- Every scheduled request resolves to the same `(k, p, temperature)` triple. `Request` funnels every
  sampling branch into `SampleTopKTopP`, so comparing the resolved triple rather than the raw
  options also treats equivalent spellings (`top_k == 1`, `temperature == 0`, `do_sample == false`)
  as the same.
- If the resolved triple is not argmax, no request pinned `random_seed`. Argmax ignores the random
  state, so batching cannot change it; a batch-wide generator would otherwise break the
  reproducibility a pinned seed promises.
- The logits rows are `vocab_size` long, back to back in request order, and inside a single
  allocation. `DeviceSpan::SameBufferAs` establishes the last part, which is what makes it safe to
  widen row 0 into the `[batch, vocab]` view the sampler reads.
- Every request's `Search` accepts a shared next-token slot. Only `GreedySearch_Cuda` does, so this
  doubles as the device and single-row check. `num_beams == 1` and `batch_size == 1` are already
  enforced by `Request`'s constructor.

Note the layout check naturally excludes mixed prefill/decode steps, which is correct: those pick
their rows out of a larger tensor and are not the steady-state case worth optimizing.

Logits processors do **not** have to be no-ops. Each request still runs `ApplyMinLength`,
`ApplyRepetitionPenalty` and `ApplyNoRepeatNgram` over its own row of the shared tensor before the
batched sampler runs, so `min_length` and `repetition_penalty` stay per-request.

Per-request `max_length` and per-request EOS token sets likewise do not need to match, because those
are handled after sampling, in the per-request tail (see 5.4).

### 5.3 Layering

`cuda::SamplingData` and `cuda::GetSample` live in `src/cuda/` and cannot be referenced from
device-agnostic code under `src/engine/`. The batched entry point therefore goes behind
`DeviceInterface` (`src/smartptrs.h`), alongside the existing `Cast`, `UpdatePositionIds` and
`LaunchAddLogitsMask` hooks, with a default implementation that reports "unsupported" so non-CUDA
devices fall back automatically:

```cpp
// src/smartptrs.h — DeviceInterface
virtual bool SampleTopKTopP(DeviceSpan<float> /*scores*/, DeviceSpan<int32_t> /*next_tokens*/,
                            int /*vocab_size*/, int /*batch_size*/,
                            int /*k*/, float /*p*/, float /*temperature*/) { return false; }
```

The CUDA override owns a lazily created `SamplingData` cached on the device interface, grown when a
batch exceeds the cached capacity and reused across steps.

### 5.4 Control flow

`ScheduledRequests::GenerateNextTokens()` gains a fast path:

1. Evaluate the gate. If it fails, run the existing two loops and return.
2. Bind each request's `Search` to its own one-element slice of the engine-owned shared next-token
   buffer.
3. For each request, run the pre-sampling half of token selection: sequence bookkeeping, handing it
   its logits row, and the logits processors.
4. Call `SampleTopKTopP` once over the contiguous `[batch, vocab]` logits.
5. For each request, launch the per-sequence tail: EOS check against *that request's* EOS set and
   append to *that request's* sequence. Nothing here reads a device result, so the whole batch
   queues without waiting.
6. `CopyDeviceToCpu()` the shared buffer once — the single synchronization for the step. It has to
   come after step 5 so that it observes the tokens padded for sequences that just hit EOS.
7. For each request, complete: advance the host mirror, `max_length` check, status update.

Steps 5 and 7 keep the two remaining per-request launches (`CheckForEOSAndPad`,
`AppendNextTokensToSequences`). They are cheap and, crucially, they are where the per-request
semantics live — which is precisely why per-request `max_length` and EOS sets do not need to be
uniform for the fast path to be valid.

Because `DeviceSpan::subspan` shares the underlying buffer and `CpuSpan()` returns the view at the
subspan's offset, the one copy in step 6 populates every request's view with no extra copies. A
request that stays bound but later takes the slow path is still correct: it samples into its own
slot and copies the shared buffer back itself, which is why a partial bind can simply fall back.

### 5.5 Changes

| File | Change |
|---|---|
| `src/smartptrs.h` | Add `DeviceInterface::SampleTopKTopP(...)` returning `false` by default, and `DeviceSpan::SameBufferAs` so the engine can tell whether pointer arithmetic between two spans is meaningful. |
| `src/cuda/interface.cpp` | Override `SampleTopKTopP`: lazily create/reuse a batch-sized `cuda::SamplingData`, call `cuda::GetSample` once, return `true`. Re-create the workspace only when the requested batch exceeds the cached capacity. |
| `src/search.h` | Add `virtual bool BindNextTokensSlot(DeviceSpan<int32_t> /*slot*/) { return false; }` and `virtual void OnNextTokensSampled() {}`, both no-ops so CPU/beam/`Generator` paths are unaffected. |
| `src/cuda/search_cuda.h` / `.cpp` | `GreedySearch_Cuda` overrides both. `BindNextTokensSlot` rejects unless the search is single-row and the slot is one element, then repoints `next_tokens_buffer_` and `next_tokens_`. The post-sampling tail moves into `LaunchNextTokensTail()`, shared by `SampleTopKTopP` and `OnNextTokensSampled`. `CompleteGeneration` skips its own `CopyDeviceToCpu()` when the caller owns the copy. |
| `src/engine/engine.h` / `.cpp` | Own the shared next-token `DeviceSpan<int32_t>`, grown as batches get larger, and hand it to `ScheduledRequests` each step. |
| `src/engine/scheduled_requests.h` / `.cpp` | Implement the gate and the fast path in `GenerateNextTokens()`; keep the existing two-loop path as the fallback. |
| `src/engine/request.h` / `.cpp` | Split the pre-sampling half of `GenerateNextTokens()` into `PrepareGeneration()`, and forward `SearchOptions()`, `BindNextTokensSlot()` and `OnNextTokensSampled()`. |
| `src/engine/decoders/*_decoder_io.cpp` | Wrap the fp32 logits tensor once instead of per row. `Tensor::GetDeviceSpan()` wraps the tensor memory afresh on each call, so calling it inside the loop produced rows that were adjacent in device memory but belonged to unrelated `DeviceBuffer` objects, which the gate has to reject. This also removes N redundant wraps per step. |

Nothing outside `src/engine/` and `src/cuda/` changes behaviour: the new `Search` virtuals default
to no-ops, and `DeviceInterface::SampleTopKTopP` defaults to "unsupported".

### 5.6 Secondary benefit: memory

Today every `Request` allocates its own `SamplingData` sized for one row but a full vocabulary —
several vocab-length fp32 buffers plus curand state, per request. A shared batch-sized workspace
replaces N of those with one, which matters at high concurrency with large vocabularies.

### 5.7 Result

Measured on H200, gpt-oss with INT8 per-channel KV, prompt 128 / 256 new tokens:

| Throughput | Before PR #2343 | Phase 1 | Phase 2 | GQA `Generator` |
|---|---|---|---|---|
| batch 1 | 376.3 | 376.6 | 377.4 | 374.4 |
| batch 8 | 1683.8 | 1749.7 | 1810.2 | 1792.7 |
| batch 32 | 4109.5 | 4517.1 | 4999.4 | 5303.5 |

Batch 1 is unchanged, as expected: a batch of one does not take the fast path. Batch 8 now edges
past the `Generator`. Batch 32 closes the gap from −22.5% to −5.7%.

The remaining gap is the two per-request tail launches per step (`CheckForEOSAndPad`,
`AppendNextTokensToSequences`) plus the paged attention kernel itself; see section 6.

---

## 6. Phase 3 ([PR #2361](https://github.com/microsoft/onnxruntime-genai/pull/2361)): scheduler-owned batched sampler

Phase 2 proves the performance value of batching, but its CUDA workspace is cached on the process-
global device interface and its fast path requires uniform parameters, contiguous logits, and no
pinned random seeds. Phase 3 makes batching the scheduler's normal sampling architecture:

- Each `Scheduler` owns one `BatchedSampler` and its reusable CUDA workspace. Separate Engines do
  not share mutable sampling state.
- Each `Request` owns a sampler state initialized from its configured seed. The state remains with
  the request when the scheduler reorders rows or requests join and leave, preserving its random
  stream independently of batch position.
- The sampler groups rows by resolved `(k, p, temperature)`. It runs one existing `GetSample` call
  per distinct group, so launch complexity is proportional to the number of parameter groups, not
  the number of requests.
- Noncontiguous logits rows, including mixed prefill/decode steps, are gathered into one reusable
  packed tensor and sampled by group. Results are scattered back to request order before the
  per-request EOS and sequence tails run.
- The scheduler and sampler reserve host planning arrays and CUDA workspace from the configured
  maximum batch size. Steady-state decoding does not allocate memory on the sampling path.

The CUDA top-k local algorithm cache is keyed by both `k` and bucket batch size. This matters for
heterogeneous batches: an algorithm selected for a single-row bucket is not necessarily valid for
a multi-row bucket with the same `k`.

The control flow remains device-neutral. `DeviceInterface::CreateBatchedSampler()` returns null on
providers without a batched implementation, and `ScheduledRequests` retains the Phase 1 fallback.
Per-request `Search` objects still own sequences, EOS state, and logits processors; Phase 3 does not
turn ragged request lifecycle into one fixed-width `Search`.

The remaining opportunity is to batch the EOS/pad and sequence-append tails using per-row pointer
descriptors. Those two launches still run once per request, but sampling itself no longer falls back
because of heterogeneous parameters, pinned seeds, batch size one, or ragged logits layout.

---

## 7. Rejected alternatives

**Give the Engine one `Search` sized to max_batch, with per-row state.** This is the "just use the
Generator design" option in its strongest form. It was rejected because it requires reworking
`Sequences` to carry a per-row length cursor (today: one `current_length_`, and `GetSequence(i)`
depends on it), reworking every kernel that writes at a shared `past_length` offset, moving
per-request `GeneratorParams` into per-row arrays, and adding row allocation/eviction to `Search`.
It also collides with `Request`'s public lifecycle — `Assign`, `Remove`, `AddTokens` can all be
called outside the engine. It is a plausible long-term direction, but it is a rewrite of the search
layer, and Phase 2 gets most of the benefit without touching any of it.

**Batch `CheckForEOSAndPad` and `AppendNextTokensToSequences` too.** Worth roughly 0.16 ms. Needs
per-row done flags, per-row EOS token sets and pointer-array kernels because each request's
`sequences_` buffer is a separate allocation at a different length. Higher risk than the sampler
batching, for a fraction of the win. Deferred.

**Remove the synchronizations without restructuring.** Measured: no speedup. See section 3.

---

## 8. Testing

The gate makes this a behaviour-preserving optimization, so the bar is bit-exactness against the
pre-change build plus coverage of the fallback path.

### Choosing a valid oracle

PagedAttention output is **not reproducible across processes**. With greedy search and a fixed
prompt, two runs of the same binary on an idle GPU produce different tokens *and* a different token
count. This reproduces identically on `main`, so it is a pre-existing property of the operator, not
of the engine. It makes paged output useless as a bit-exactness oracle. (Repeating a batch *within*
one process is stable, which is a much weaker signal.)

The valid oracle is the **GQA model driven through the `Engine`**, which is deterministic across
processes. It exercises `Request`, the `tokens_host_` mirror,
`ScheduledRequests::GenerateNextTokens`, `GreedySearch_Cuda`'s deferred completion and the batched
sampler — everything on the changed path except `VarlenDecoderIO`, since it goes through
`StaticBatchDecoderIO` instead. That gap is covered by the paged regression suite below on
behaviour rather than exact tokens.

Note that this also localizes the nondeterminism: identical engine and search code is deterministic
under GQA attention and nondeterministic under PagedAttention, so the variability lives in the
PagedAttention operator. That is worth investigating separately.

Note also that a batched run and a solo run of the same prompt legitimately produce different
tokens, because the batch size changes the GEMM shapes and therefore the logits. Comparisons must
be like-for-like across builds, never batch-vs-solo.

### What was run

- **Bit-exactness, GQA through the `Engine`, before vs after.** Seven scenarios, 628 tokens,
  identical: uniform greedy (fast path), `repetition_penalty = 1.2`, `min_length = 20`, staggered
  per-request `max_length`, a batch mixing greedy and two different sampling configurations,
  a batch with pinned `random_seed`, and four solo runs. The instrumented build confirmed which of
  these take the fast path and which fall back, so none of the comparisons is vacuous.
- **Paged regression suite.** `check_e2e`, `needle_test --filler 400` (8093-token prompt, needle
  retrieved), and `long_decode_test` — the last is the only one that decodes past 2048 KV and
  crosses block-table column boundaries.
- **Unit tests.** `unit_tests --gtest_filter='*CAPITests*:*ContinuousDecoding*:*Sampling*Cuda*:*Rewind*'`,
  36 passed.
- **`Generator` regression.** Unchanged, since all new virtuals default to no-ops there.
