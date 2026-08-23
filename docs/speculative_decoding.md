# Speculative Decoding in ONNX Runtime GenAI

This document describes the two speculative-decoding runtimes in onnxruntime-genai:

1. **Draft-model speculative decoding** — a separate, smaller ONNX model proposes tokens that the
   main ("target") model verifies. Driven by `DecodingStrategy` and configured through
   `model.draft` in `genai_config.json`.
2. **MTP (Multi-Token Prediction) self-speculative decoding** — a single extra decoder layer
   exported alongside the main model drafts from the main model's own last hidden state. Driven by
   `MtpGenerator` (`OgaMtpGenerator` / `og.MtpGenerator`).

Both share the same sampling, penalty, verification and statistics code so that their outputs match
plain autoregressive decoding.

---

## 1. Why speculative decoding

Autoregressive decode is memory-bandwidth bound: each token requires a full pass over the model
weights to produce one token. Speculative decoding amortizes that pass by:

1. **Draft** — cheaply propose `K` candidate tokens `d_0 .. d_{K-1}`.
2. **Verify** — run the target model once over `[t, d_0 .. d_{K-1}]`, which yields `K+1` logit rows
   for the price of roughly one decode step.
3. **Accept** — keep the longest prefix the target agrees with, plus one free "bonus" token from
   the last accepted row.

If `a` drafts are accepted, the round emits `a + 1` tokens from one target forward. Correctness is
preserved by the acceptance rule, not by the draft's quality — the draft only affects speed.

---

## 2. Shared building blocks

| Component | File | Role |
|---|---|---|
| `SampledCategorical` | [src/sampling_distribution.h](../src/sampling_distribution.h) | Sparse truncated distribution: kept ids + renormalized probs |
| `ComputeSampledCategorical` | [src/sampling_distribution.h](../src/sampling_distribution.h) | Builds the top-k / top-p / temperature truncated distribution. `top_k > 1` enables top-k; `0 < top_p < 1` enables the nucleus cutoff |
| `FindNucleus` | [src/sampling_distribution.h](../src/sampling_distribution.h) | Adaptive partial-sort nucleus search with a log-space tail bound (avoids the O(V) softmax partition) |
| `LogitsPenaltyProcessor` | [src/sampling_distribution.h](../src/sampling_distribution.h) | Applies min-length, repetition penalty and no-repeat-ngram to one logits row, identical to `Search_Cpu` |
| `ComputeAcceptProb` | [src/speculative_sampling.h](../src/speculative_sampling.h) | `min(1, p_target / p_draft)` |
| `GetSparseTokenProbability` | [src/speculative_sampling.h](../src/speculative_sampling.h) | O(K) probability lookup in a sparse distribution |
| `SampleSparseToken` | [src/speculative_sampling.h](../src/speculative_sampling.h) | Draw from a sparse categorical |
| `SampleCorrectionToken` | [src/speculative_sampling.h](../src/speculative_sampling.h) | Draw from the normalized residual `max(0, p - q)` over the target's support |
| `ComputeTargetTokenSelection` | [src/speculative_sampling.h](../src/speculative_sampling.h) | Penalties + greedy argmax or truncated sampling for one target row |
| `SpeculativeStats` | [src/speculative_stats.h](../src/speculative_stats.h) | Counters, timings and derived rates reported by both runtimes |

### Acceptance rules

**Greedy** (`do_sample=false`, or `temperature == 0`): accept `d_i` iff `d_i == argmax(p_i)`.

**Sampling** (`do_sample=true`, `temperature > 0`): the Leviathan/Chen rejection test.

$$
\text{accept } d_i \text{ with probability } \min\!\left(1, \frac{p_i(d_i)}{q_i(d_i)}\right)
$$

On the first rejection at position `a`, redraw from the normalized residual

$$
\tilde{p}_a(x) \;=\; \frac{\max\!\left(0,\; p_a(x) - q_a(x)\right)}{\sum_y \max\!\left(0,\; p_a(y) - q_a(y)\right)}
$$

and if every draft is accepted, draw a bonus token from `p_K`. This makes the emitted sequence a
draw from the same distribution as plain top-k/top-p sampling, so near-tie numeric differences
between the wide verify forward and a 1-token decode become valid draws rather than mismatches.

Both `p_i` and `q_i` are the **truncated** distributions (same `top_k` / `top_p` / `temperature`),
which keeps the residual sparse — it lives on the target's support only, so the reject path never
materializes two full-vocab vectors.

---

## 3. Draft-model speculative decoding

### Strategy selection

`MakeDecodingStrategy(Generator&)` in [src/decoding_strategy.cpp](../src/decoding_strategy.cpp)
picks the strategy once per `Generator`:

```
IsTransducer(model.type)  -> TransducerDecodingStrategy
config.model.draft        -> BaseSpeculativeStrategy
otherwise                 -> StandardDecodingStrategy
```

`DecodingStrategy` (see [src/decoding_strategy.h](../src/decoding_strategy.h)) exposes
`Step`, `Reset`, `GetStats`, `PrepareForAppend`, `TryGetExternalLogits` and `PrepareForSetLogits`.
`Generator::GenerateNextToken` calls `Step` exactly once per user-visible token, so speculative
decoding is transparent to callers.

### Configuration

```jsonc
{
  "model": {
    "decoder": { "filename": "model.onnx", "...": "..." },
    "draft":   { "filename": "draft.onnx", "...": "..." }
  },
  "speculative": { "max_draft_tokens": 4 }
}
```

`model.draft` is an `std::optional<Decoder>` in [src/config.h](../src/config.h), so it carries the
same `filename`, `session_options`, `run_options`, `inputs.*` and `outputs.*` fields as the main
decoder. The top-level `speculative.max_draft_tokens` sets `K` (default 4, validated to a small
positive range at parse time).

### Round structure

`SpeculativeDecodingStrategy` ([src/speculative_decoding_strategy.h](../src/speculative_decoding_strategy.h))
owns the round state machine and defers to two virtuals implemented by `BaseSpeculativeStrategy`:

- `Proposal Propose(Generator&, int K, int seed_length)` — run the draft model `K` times, applying
  the same penalties as the target, and return the `K` tokens plus (in sampling mode) the sparse
  `q_i` per position.
- `void Advance(Generator&, const Proposal&, int n_direct, int32_t final_token, int seed_length)` —
  bring the draft model's KV cache in line with what was actually committed.

`Step` drives:

```
Step()
 ├─ round idle?  -> RunRound()   : Propose -> target verify -> accept -> buffer tokens
 └─ DrainOne()                   : emit exactly one buffered token
```

`RoundState::Phase` moves `kIdle -> kDraining -> kFinalizing -> kIdle`, with a `kReconcilePending`
phase for the case where the caller interrupts a partially drained round (e.g. `AppendTokens`
mid-stream or an external `SetLogits`).

Two notable optimizations:

- **Re-anchor fold** — `pending_anchor_token_` carries the round's own committed token into the next
  round's verify batch (width `K+1`), saving one target forward per round when the target emits one
  logits row per input token.
- **Guidance rounds** — `RunGuidanceRound` / `FinalizeGuidanceRound` verify drafts against a
  grammar-masked target and carry grammar-forced tokens across rounds in `ff_carry_`.

State lives in `SpeculativeDecodingState` ([src/models/speculative_decoding.h](../src/models/speculative_decoding.h)),
a dual-state container exposing `target_state()` and `draft_state()` plus a reusable
`draft_pending_logits_` buffer so no allocation happens per round.

---

## 4. MTP self-speculative decoding

### Model shape

MTP (Qwen3.6 / DeepSeek style) needs no second model. The main model is exported with its final
hidden state as an extra output, and a **one-layer** head (`mtp.onnx`) consumes that hidden state
plus the just-emitted token to predict the *next-next* token:

```
h_i          = main_model(t_0 .. t_i)          # last hidden state, [1, S, H]
h'_i         = mtp.fc( concat(mtp.norm_h(h_i), mtp.norm_e(embed(t_{i+1}))) )
h''_i        = mtp.layer(h'_i)                 # one attention + MLP/MoE block
logits       = lm_head( mtp.norm(h''_i) )      # predicts t_{i+2}
```

Because the head reuses the main model's embedding and `lm_head`, a draft step costs roughly
`1/num_layers` of a main forward.

`genai_config.json` for the main model declares the hidden-state output (and, for the model
builder's benefit, an `mtp` block):

```jsonc
{
  "model": {
    "decoder": {
      "filename": "model.onnx",
      "outputs": {
        "logits": "logits",
        "hidden_states": "hidden_states"      // required: feeds the MTP head
      }
    },
    "mtp": {
      "filename": "mtp.onnx",
      "num_hidden_layers": 1,
      "num_key_value_heads": 16,
      "head_size": 256,
      "main_hidden_states": "hidden_states",
      "outputs": { "hidden_states": "hidden_states_out" }
    }
  }
}
```

The head is loaded as a standalone `og.Model` from its own folder, whose `genai_config.json`
declares `model.decoder.inputs.hidden_states` (and, for `N > 1` chaining, an extra
`hidden_states_out` output produced by exporting with `mtp_emit_hidden=true`).

See [examples/python/qwen-3.6-mtp.md](../examples/python/qwen-3.6-mtp.md) for the full export
recipe and [examples/python/qwen-3.6-mtp.py](../examples/python/qwen-3.6-mtp.py) for a runnable
example.

### Gemma4 assistant heads

`OgaCreateMtpGenerator` also accepts a Gemma4 *assistant* head (`model.type` =
`"gemma4_assistant"`). It differs from the Qwen-style head in two ways: it consumes the target's
token embedding concatenated with the carried hidden state, and it reads the target's present KV
for a few layers instead of owning a cache. Both are declared in the target's `mtp` block:

```jsonc
"mtp": {
  "filename": "assistant.onnx",
  "main_hidden_states": "final_hidden_state",   // target output: final hidden state
  "main_inputs_embeds": "inputs_embeds",        // target output: token embeddings
  "shared_kv_layers": [22, 23],                 // target layers whose present KV the head reads
  "inputs": {
    "hidden_states": "inputs_embeds",           // head input: embedding ++ hidden, [1, 1, 2H]
    "attention_mask": "attention_mask",
    "shared_key_names": ["shared_kv.sliding_attention.key", "shared_kv.full_attention.key"],
    "shared_value_names": ["shared_kv.sliding_attention.value", "shared_kv.full_attention.value"]
  },
  "outputs": {
    "logits": "logits",
    "hidden_states": "projected_state"          // head output fed back into the next draft
  }
}
```

`shared_key_names[i]` / `shared_value_names[i]` are bound to the target's present key/value for
`shared_kv_layers[i]`, composed from `model.decoder.outputs.present_key_names` and
`present_value_names`. Buffer widths come from `model.decoder.hidden_size`.

`main_inputs_embeds` must name an output the runtime *binds*, not merely one the ONNX graph
declares — for a multi-modal package that is the embedding stage's
`model.embedding.outputs.embeddings`.

A target whose logits output has a bounded sequence dimension (it emits one row per input token
only while the input fits that bound, and last-token logits for longer prefill forwards) must
declare `model.decoder.max_logits_sequence_length`.

`builder.py` has no Gemma 4 support, so both graphs come from an external exporter
([mobius](https://github.com/onnxruntime/mobius)) plus a post-processing pass that adds the two
extra target outputs and writes the `mtp` block;
[examples/python/gemma-4-mtp-build.py](../examples/python/gemma-4-mtp-build.py) drives the whole
pipeline.

See [examples/python/gemma-4-mtp.md](../examples/python/gemma-4-mtp.md) for the full contract and
build recipe, and [examples/python/gemma-4-mtp.py](../examples/python/gemma-4-mtp.py) for a
runnable example.

### Runtime

`MtpGenerator` ([src/mtp_generator.h](../src/mtp_generator.h),
[src/mtp_generator.cpp](../src/mtp_generator.cpp)) composes two ordinary `Generator`s — the main
decoder and the head — on the **shared compute stream**, and keeps the hidden-state handoff
device-to-device. Constraints enforced at construction: `batch_size == 1`, `num_beams == 1`,
`num_return_sequences == 1`, no guidance, and matching device type / vocab size / hidden size /
hidden-state dtype between the two models.

`GenerateNextToken()` emits exactly one token per call; an internal round may commit several, which
are buffered in `pending_tokens_` and drained one per call. The exposed sequence is
`emitted_sequence_`.

#### Loop-carried state

| Field | Meaning |
|---|---|
| `length_` | Committed main-model cache length `L` |
| `next_token_` | Token predicted for position `L`, committed at the top of the next round |
| `hidden_slice_` | `[1,1,H]` device buffer holding the main hidden paired with `next_token_` |
| `head_out_hidden_` | `[1,1,H]` capture of the head's own post-norm output, fed back when chaining `N > 1` drafts |
| `head_len_` | Number of committed generated tokens in the head's KV cache |

#### Round: `N == 1` (greedy)

```
draft   d = head(hidden_slice_, t)                     # 1 head forward
verify  main([t, d])                                   # 1 main forward, 2 logit rows
accept  d == argmax(row 0)?
  yes -> commit d, bonus = argmax(row 1), length_ += 2
         DraftTwo(hidden, d, bonus)                    # fused KV-advance + next draft
  no  -> RewindToLength(L), re-run [t], length_ += 1
```

The accept path fuses the post-accept head KV-advance and the *next* round's draft into one
2-token head forward (`DraftTwo`), and stashes the result in `pending_draft_`, so a fully accepted
stream costs one main forward and one head forward per two tokens.

#### Round: `N > 1` (greedy)

The single head module is chained `N` times, feeding its own post-norm hidden forward
(`head_out_hidden_`), the way vLLM's `AutoRegressiveSpeculator` does. Then `[t, d_0 .. d_{N-1}]` is
verified in one `N+1`-wide main forward, `a` = length of the greedy-matching prefix, and one of four
finalize branches runs:

| Branch | Condition | Cost |
|---|---|---|
| All accepted | `a == N` | No extra forward; bonus is verify row `N` |
| Lossless crop + `M=1` replay | `a >= 1` and windowed recurrent state | Crop to `L+a`, replay the last committed token (1-wide, decode-consistent bonus) |
| Snapshot rewind + replay | otherwise | `RewindToLength(L)` + replay `[t, d_0 .. d_{a-1}]` |

Re-materializing the `a` accepted drafts in the head's KV and drafting the next round's first token
both consume *main-model* hidden states over consecutive positions, so they are **deferred and
fused** into one `(a+1)`-token head forward issued at the top of the next round
(`pending_refeed_count_`, `merged_tokens_`, `refeed_multi_`). The head KV is therefore left in its
speculative state between rounds and rewound lazily — nothing reads it in between.

#### Round: sampling (`do_sample=true`, any `N`)

Same structure, but each draft is sampled from its truncated `q_k` (recorded in
`draft_idx_`/`draft_prob_`), each verify row yields a truncated `p_k`
(`target_idx_`/`target_prob_`), and acceptance uses the rejection test of §2. Because rejection
sampling corrects the *output distribution* regardless of small numeric drift in the wide verify
forward, the reject path can crop instead of replaying whenever the recurrent state is windowed.

#### Numerical caveat

A wide (`M = N+1`) verify forward is numerically close to, but not bit-identical to, `M = 1` decode
(different GEMM tiling; XQA vs. Flash attention). Only the **last** row of a forward matches a
1-token decode. Greedy MTP is therefore "lossless modulo near-ties": the finalize branches are
ordered so that the bonus token comes from a decode-consistent row wherever that is affordable.
Under sampling this is a non-issue by construction.

### Device offloads

MTP adds five `DeviceInterface` virtuals ([src/smartptrs.h](../src/smartptrs.h), CUDA
implementations in [src/cuda/interface.cpp](../src/cuda/interface.cpp)). All are appended at the end
of the struct for vtable/ABI stability and return `false` on devices without an implementation, so
every call site has a host fallback.

| Virtual | Purpose |
|---|---|
| `ArgMax(logits, type, num_rows, vocab_size, out_tokens)` | On-device greedy argmax; only token ids leave the GPU |
| `ArgMaxDevice(..., DeviceSpan<int32_t> out_tokens)` | Same, but the result stays on device to feed the next chained head forward |
| `TopKScores(..., k, out_tokens, out_scores)` | Per-row top-`k` ids + raw fp32 scores, so speculative sampling never copies full-vocab logits |
| `CopyStateSlots(descs_device, count, src_slot, dst_slot)` | One kernel replaces the 60+ `cudaMemcpyAsync` calls of the per-tensor window-slot promote loop |

Because the add-on is loaded dynamically, `kDeviceInterfaceVersion` in
[src/smartptrs.h](../src/smartptrs.h) is exported as `GetInterfaceVersion` and checked in
`OrtGlobals::LoadCudaInterface` *before* `GetInterface` is resolved, so a mismatched
`onnxruntime-genai-cuda` package fails loudly instead of calling through a shifted vtable.

### Hybrid (linear-attention) models

Models like Qwen3.6 interleave GQA layers with GatedDeltaNet layers. The attention KV cache can be
cropped to any length, but the recurrent state cannot — it is a single tensor that has already
absorbed every token of the forward. `RecurrentState`
([src/models/recurrent_state.h](../src/models/recurrent_state.h)) offers two rollback mechanisms:

- **Snapshot / restore** — `Snapshot(position)` copies the live conv + recurrent buffers before a
  speculative forward; `RewindTo` restores them in place (in place, so buffer addresses stay stable
  for CUDA-graph replay). Always available, but costs `2 * num_layers` device copies per step.
- **Windowed state** — when the model is exported with `state_window = W`, the state
  tensors carry a window axis of `W` per-token states, right-aligned: slot `j` holds the state after
  token `seq_len - W + j`, and slot `W-1` is the one the ops read. `CropToPosition(position)` then
  commits a partial accept by promoting the slot for `position` into slot `W-1` — one batched kernel
  instead of a replay forward. `IsWindowed()` reports availability; `SetForwardLength()` records the
  current forward's length so positions map to slots.

`Generator` exposes this as `SnapshotState()`, `CanCropRecurrentState()` and
`CropToAccepted(new_length, recurrent_position)`
([src/generators.h](../src/generators.h)). `MtpGenerator` snapshots before each wide greedy verify
so a partial rejection can always replay the committed prefix with decode-consistent numerics.

`RecurrentState::GraphCaptureVariant()` feeds the graph-capture annotation id so a double-buffered
recurrent state never replays a graph bound to the other buffer.

### CUDA graph capture across two shapes

Ordinary decode captures one graph for `sequence_length == 1`. MTP also runs the `N+1`-wide verify
shape, so `GeneratorParams::max_graph_capture_length` is set to `num_speculative_tokens + 1` and:

- `input_ids`, `logits`, `position_ids` and the hidden-state feeders use static buffers for every
  length in `[1, max_graph_capture_length]`, pre-sized to the largest captured length through
  `Tensor::CreateTensor(shape, make_static, static_capacity_bytes)`, so the base address is stable
  across shapes while ORT still sees only the bytes of the current shape.
- `State::GraphIdForLength(length, variant)` hands out a distinct random annotation id per
  `(length, recurrent variant)` pair, so ORT captures and replays an independent graph per shape.

`HiddenStatesInputs` / `HiddenStatesOutputs`
([src/models/hidden_states.h](../src/models/hidden_states.h)) keep one dedicated
static buffer per captured length and a single shared dynamic buffer for everything else (prompt
prefill), so the prompt does not leave a per-length buffer behind. `HiddenStatesInputs::Update`
validates the source element type and byte count, then prefers a stream-ordered device-to-device
copy, falling back to a host-staged copy only when ORT placed the source on the CPU.

---

## 5. API surface

### C

```c
OgaResult* OgaCreateMtpGenerator(const OgaModel* main_model, const OgaModel* mtp_model,
                                 const OgaGeneratorParams* params, OgaMtpGenerator** out);
OgaResult* OgaMtpGenerator_AppendTokens(OgaMtpGenerator*, const int32_t* input_ids, size_t count);
OgaResult* OgaMtpGenerator_GenerateNextToken(OgaMtpGenerator*);
bool       OgaMtpGenerator_IsDone(const OgaMtpGenerator*);
size_t     OgaMtpGenerator_GetSequenceCount(const OgaMtpGenerator*);
const int32_t* OgaMtpGenerator_GetSequenceData(const OgaMtpGenerator*);
size_t     OgaMtpGenerator_GetForwardCount(const OgaMtpGenerator*);
size_t     OgaMtpGenerator_GetAcceptCount(const OgaMtpGenerator*);
size_t     OgaMtpGenerator_GetTrialCount(const OgaMtpGenerator*);
OgaResult* OgaMtpGenerator_GetSpeculativeStats(const OgaMtpGenerator*, OgaSpeculativeStats** out);
void       OgaDestroyMtpGenerator(OgaMtpGenerator*);
```

Two `OgaGenerator` entry points support building an MTP-style loop by hand:

```c
OgaResult* OgaGenerator_SnapshotState(OgaGenerator*);
OgaResult* OgaGenerator_SetHiddenStates(OgaGenerator*, OgaTensor* hidden_states);
```

`OgaGenerator_SnapshotState` is a no-op for models without recurrent state;
`OgaGenerator_SetHiddenStates` is a no-op for models without a `hidden_states` input.

### C++

`OgaMtpGenerator` in [src/ort_genai.h](../src/ort_genai.h) is the usual RAII wrapper
(`Create`, `AppendTokens`, `GenerateNextToken`, `IsDone`, `GetSequenceCount`, `GetSequenceData`,
`GetForwardCount`, `GetAcceptCount`, `GetTrialCount`, `GetSpeculativeStats`). `OgaGenerator` gains
`SnapshotState()` and `SetHiddenStates()`.

### Python

```python
import onnxruntime_genai as og

main_model = og.Model(main_model_path)
mtp_model  = og.Model(mtp_model_path)

params = og.GeneratorParams(main_model)
params.set_search_options(max_length=1024, do_sample=False)

gen = og.MtpGenerator(main_model, mtp_model, params)
gen.append_tokens(prompt_tokens)          # numpy int32, 1-D
while not gen.is_done():
    gen.generate_next_token()

tokens = gen.get_sequence()               # numpy int32
stats  = gen.get_stats()                  # dict
```

`Generator` also gains `snapshot_state()` and `set_hidden_states(ndarray)` for building a
speculative loop in Python (see the reference implementation in
[examples/python/qwen-3.6-mtp.py](../examples/python/qwen-3.6-mtp.py)).

---

## 6. Statistics

Both runtimes report the same `SpeculativeStats` schema
([src/speculative_stats.h](../src/speculative_stats.h)), surfaced as a dict in Python and through
`OgaSpeculativeStatsGetCount` / `OgaSpeculativeStatsGetNumber` / `OgaSpeculativeStatsGetBool` in C.

| Group | Fields |
|---|---|
| Rounds | `rounds`, `completed_rounds`, `interrupted_rounds`, `active_rounds` |
| Draft tokens | `draft_tokens_proposed`, `draft_tokens_evaluated`, `draft_tokens_accepted` |
| Emitted tokens | `correction_tokens`, `bonus_tokens`, `tokens_queued`, `tokens_emitted`, `tokens_discarded`, `tokens_buffered` |
| Forwards | `draft_forward_passes`, `target_forward_passes` |
| Timings | `total_draft_ms`, `total_target_ms`, `total_reconciliation_ms`, `avg_draft_ms_per_token`, `avg_target_ms_per_round` |
| Derived | `acceptance_rate`, `avg_draft_tokens_per_round`, `mean_emitted_tokens_per_round`, `expected_tokens_per_round`, `target_baseline_ms_per_token`, `target_overhead_ratio`, `estimated_speedup`, `observed_speedup`, `formula_supported` |

`MtpGenerator` populates the counters plus `acceptance_rate`, `avg_draft_tokens_per_round` and
`mean_emitted_tokens_per_round`; timing fields are left zero. `MtpGenerator.get_stats()` in Python
additionally aliases `forwards`, `accepts` and `trials` for convenience.

The single most useful number is `acceptance_rate`. With `N` drafts per round and a per-token
acceptance probability `α`, the expected tokens per round is

$$
E[\text{tokens}] = \frac{1 - \alpha^{N+1}}{1 - \alpha}
$$

so speedup saturates quickly in `N` unless `α` is high — increasing `max_draft_tokens`
past the point where `α^N` decays only adds wasted head forwards.

---

## 7. Tuning knobs (MTP)

Both are read once at `MtpGenerator` construction, from the same generator parameters the
draft-model speculative path uses.

| Parameter | Set with | Default | Effect |
|---|---|---|---|
| `speculative.max_draft_tokens` | `SetSpeculativeNumber("max_draft_tokens", N)` | `4` | `N`, the number of chained draft tokens per round. For `N > 1`, `model.mtp.outputs.hidden_states` must name the feedback output exported with `mtp_emit_hidden=true`. `N` is capped at `state_window - 1` on a windowed-state model, whose verify forward is `N + 1` wide. |
| `search.chunk_size` | `SetSearchNumber("chunk_size", n)` | `256` on windowed-state models, `0` otherwise | Max tokens per prompt forward. Bounds the ORT activation arena (measured 54 GB chunked vs. 94 GB unchunked on a 2.8k-token prompt). `0` = single forward |

On a windowed-state model (`state_window > 1`) with at least one accepted draft, the
greedy path crops to the last accepted state and replays only that token with an `M=1` forward.
When no draft is accepted, it restores the snapshot and replays the committed token.

---

## 8. Choosing between the two

| | Draft model | MTP |
|---|---|---|
| Extra artifact | A second, smaller ONNX model | One extra decoder layer (`mtp.onnx`) |
| Draft cost | Full forward of the draft model | ~`1/num_layers` of a main forward |
| Batch / beams | General | `batch_size == 1`, `num_beams == 1` |
| Guided generation | Supported (`RunGuidanceRound`) | Not supported |
| Integration | Transparent: `og.Generator` + `model.draft` in config | Explicit: `og.MtpGenerator(main_model, mtp_model, params)` |
| Acceptance | Depends on how well the draft model tracks the target | Typically high — the head sees the target's own hidden state |

---

## 9. Testing

| Test | Coverage |
|---|---|
| [test/sampling_distribution_tests.cpp](../test/sampling_distribution_tests.cpp) | `ComputeSampledCategorical`, `FindNucleus`, the three logits penalties |
| [test/speculative_sampling_tests.cpp](../test/speculative_sampling_tests.cpp) | `ComputeAcceptProb`, sparse lookup/sampling, correction sampling, densification |
| [test/python/models/test_speculative_decoding.py](../test/python/models/test_speculative_decoding.py) | End-to-end draft-model speculative decoding, config validation, stats schema |

When changing either runtime, the load-bearing invariant to test is **distributional equivalence to
plain decoding**, not token-for-token parity: a one-ULP difference can flip a top-k selection and
change the whole continuation. Compare teacher-forced logits/NLL over a fixed token sequence, and
run an identity control (the same model twice) first to confirm the noise floor is zero.
