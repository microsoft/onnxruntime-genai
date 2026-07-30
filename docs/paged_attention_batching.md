# Engine Batching Design

Status: design proposal. The "Phase 1" work described here is merged
([PR #2343](https://github.com/microsoft/onnxruntime-genai/pull/2343)); "Phase 2" is not yet
implemented.

This document explains how batching works in the two generation paths in onnxruntime-genai, why
they differ, and what would have to change to close the remaining throughput gap between them.

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

## 4. Phase 1 (merged): defer completion

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

What Phase 1 did **not** do is reduce the amount of sampling work. There are still 32 separate
single-row `GetSample` invocations costing ~451 µs of GPU time to do what one batched call would do
in ~50 µs.

---

## 5. Phase 2 (proposed): batched sampling fast path

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

- Device is CUDA (`SamplingData` and `GetSample` are CUDA-only).
- Every scheduled request has identical `do_sample`, `top_k`, `top_p`, `temperature`.
- Every request has `num_beams == 1`.
- Logits processors are no-ops for every request:
  - `sequences_.GetSequenceLength() >= min_length` (see `Search_Cuda::ApplyMinLength`, which
    early-returns in that case),
  - `repetition_penalty == 1.0f` (`Search_Cuda::ApplyRepetitionPenalty` early-returns),
  - `no_repeat_ngram_size == 0` (CUDA throws otherwise anyway).
- The decoder produced a contiguous `[batch, vocab]` fp32 tensor. `VarlenDecoderIO::ProcessLogits`
  already builds exactly this in `logits_fp32_` on its batched-cast path; that path is taken when
  `valid_token_indices[i] == i` for all *i*, i.e. every request contributed exactly one token.

Note the gate naturally excludes mixed prefill/decode steps, which is correct: those have
non-contiguous logits rows and are not the steady-state case worth optimizing.

Per-request `max_length` and per-request EOS token sets deliberately do **not** need to match,
because those are handled after sampling, in the per-request completion (see 5.4).

### 5.3 Layering

`cuda::SamplingData` and `cuda::GetSample` live in `src/cuda/` and cannot be referenced from
device-agnostic code under `src/engine/`. The batched entry point therefore goes behind
`DeviceInterface` (`src/smartptrs.h`), alongside the existing `Cast`, `UpdatePositionIds` and
`LaunchAddLogitsMask` hooks, with a default implementation that reports "unsupported" so non-CUDA
devices fall back automatically:

```cpp
// src/smartptrs.h — DeviceInterface
virtual bool SampleBatch(DeviceSpan<float> /*logits*/, DeviceSpan<int32_t> /*next_tokens*/,
                         int /*batch_size*/, int /*vocab_size*/,
                         int /*k*/, float /*p*/, float /*temperature*/) { return false; }
```

The CUDA override owns a lazily created, batch-sized `SamplingData` cached on the device interface
(or on the `Engine`), sized to the configured maximum batch and reused across steps.

### 5.4 Control flow

`ScheduledRequests::GenerateNextTokens()` gains a fast path:

1. Evaluate the gate. If it fails, run the existing two loops and return.
2. Acquire the shared next-token buffer, sized `max_batch`, owned by the `Engine`.
3. Bind each request's `Search` to its own one-element slice of that buffer.
4. Call `SampleBatch` once over the contiguous logits.
5. `CopyDeviceToCpu()` the shared buffer once — this is the single synchronization for the step.
6. For each request, run the per-request tail: EOS check against *that request's* EOS set, append
   to *that request's* sequence, `max_length` check, status update.

Step 6 keeps the two remaining per-request launches (`CheckForEOSAndPad`,
`AppendNextTokensToSequences`). They are cheap and, crucially, they are where the per-request
semantics live — which is precisely why per-request `max_length` and EOS sets do not need to be
uniform for the fast path to be valid.

### 5.5 Required changes

| File | Change |
|---|---|
| `src/smartptrs.h` | Add `DeviceInterface::SampleBatch(...)` returning `false` by default. |
| `src/cuda/interface.cpp` | Override `SampleBatch`: lazily create/reuse a batch-sized `cuda::SamplingData`, call `cuda::GetSample` once, return `true`. Re-create the workspace only when the requested batch exceeds the cached capacity. |
| `src/search.h` | Add `virtual bool BindSharedNextTokens(DeviceSpan<int32_t> /*slot*/) { return false; }` and `virtual void OnSharedNextTokensReady() {}`, both no-ops so CPU/beam/`Generator` paths are unaffected. |
| `src/cuda/search_cuda.h` | `GreedySearch_Cuda` overrides both. Add `bool shared_next_tokens_{false};`. |
| `src/cuda/search_cuda.cpp` | `BindSharedNextTokens`: reject unless `BatchBeamSize() == 1`; repoint `next_tokens_buffer_` and `next_tokens_` at the supplied slice; set `shared_next_tokens_ = true`. `OnSharedNextTokensReady`: run the existing `CheckForEOSAndPad` + `AppendNextTokensToSequences` tail and set `completion_pending_` / `done_pending_`. `CompleteGeneration`: skip its own `next_tokens_buffer_.CopyDeviceToCpu()` when `shared_next_tokens_` is set, since the engine already copied the shared buffer. |
| `src/engine/engine.h` / `engine.cpp` | Own the shared next-token `DeviceSpan<int32_t>` sized to the configured max batch, and hand it to `ScheduledRequests`. |
| `src/engine/scheduled_requests.h` / `.cpp` | Implement the gate and the fast path in `GenerateNextTokens()`; keep the existing two-loop path as the fallback. |
| `src/engine/request.h` / `request.cpp` | Expose whatever the gate needs to inspect (sampling params, `num_beams`, `min_length` satisfaction) and split `CompleteGeneration()` so the shared-buffer variant skips the redundant copy. |

Nothing outside `src/engine/` and `src/cuda/` changes behaviour: the new `Search` virtuals default
to no-ops, and `SampleBatch` defaults to "unsupported".

### 5.6 Secondary benefit: memory

Today every `Request` allocates its own `SamplingData` sized for one row but a full vocabulary —
several vocab-length fp32 buffers plus curand state, per request. A shared batch-sized workspace
replaces N of those with one, which matters at high concurrency with large vocabularies.

### 5.7 Projected result

| Per step at batch 32 | Before PR #2343 | After PR #2343 | Projected Phase 2 |
|---|---|---|---|
| `cudaStreamSynchronize` | 130 | 32 | 1 |
| Tail `cudaLaunchKernel` | 160 | 129 | ~68 |
| GPU time in sampling | 451 µs | 451 µs | ~50 µs |
| Tail wall time | 2.384 ms | 1.423 ms | ~0.45–0.55 ms |
| Step | 8.603 ms | 7.620 ms | ~6.7 ms |
| Throughput | 4109 tok/s | 4517 tok/s | ~5100 tok/s |

For reference the `Generator` path reaches 5303 tok/s at batch 32, so this would bring the Engine
to roughly −3%, versus −22% before any of this work.

---

## 6. Rejected alternatives

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

## 7. Testing

The gate makes this a behaviour-preserving optimization, so the bar is bit-exactness plus coverage
of the fallback path.

- **Bit-exactness against the pre-change build.** Drive the Engine with a fixed prompt set, both as
  a batch and one at a time, and compare generated token IDs. This is how PR #2343 was validated
  (582 tokens across 16 sequences, identical).
- **Fallback coverage.** Requests with differing `top_k`/`top_p`/`temperature` in one batch;
  a request with `repetition_penalty != 1`; a request still under `min_length`; a mixed
  prefill/decode step. Each must take the slow path and produce the same tokens as today.
- **Per-request semantics under the fast path.** Requests with different `max_length` and different
  EOS sets in the same batch must still stop independently.
- **Paged-specific regression suite.** `check_e2e`, `needle_test --filler 400`, and
  `long_decode_test` — the last is the only one that decodes past 2048 KV and crosses block-table
  column boundaries.
- **Unit tests.** `unit_tests --gtest_filter='*CAPITests*:*ContinuousDecoding*:*Sampling*Cuda*:*Rewind*'`.
- **`Generator` regression.** Must be unchanged, since all new virtuals default to no-ops there.

### Known issue, not caused by this work

Paged decode is not reproducible **across processes** when the KV block pool is in a different
allocation state: a request can generate a different number of tokens from run to run. This
reproduces identically on `main`. Batch-8 and single-request outputs *are* stable within a process
and across builds, which is what the bit-exactness comparison relies on. Mixed staggered batches
should not be used as a determinism signal.
