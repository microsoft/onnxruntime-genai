# Speculative decoding

Speculative decoding reduces autoregressive decode latency by proposing several tokens and
verifying them with the target model in one multi-token forward pass. ONNX Runtime GenAI
provides two proposer strategies:

| Method | Proposer | Additional model | Best fit |
| --- | --- | --- | --- |
| **Base speculative decoding** | A smaller draft model | Required | A compatible draft model is much cheaper than the target and predicts it well |
| **N-gram decoding** | Repeated token patterns in committed history | Not required | Repetitive or structured text such as code, templates, and input-grounded generation |

Both methods use the same `Model`, `GeneratorParams`, `Generator`, and token-generation loop.
They also share proposal-width controls, optional cooldown behavior, lifecycle handling, and
statistics. The target model decides every committed token; speculative work does not relax the
target model's decoding distribution.

The continuous-batching `Engine` also has a lower-level API for **caller-supplied greedy draft
proposals**. It is not a third automatic proposer strategy: the application produces the draft
tokens and attaches them to an individual `Request` for its next Engine step. See
[Engine caller-supplied proposals](#engine-caller-supplied-proposals) and
[Paged Attention Engine](paged_attention_engine.md#caller-supplied-speculative-drafts).

## Choose a method

Use **base speculative decoding** when:

- A smaller model uses the same tokenizer and vocabulary as the target.
- The draft is inexpensive relative to the target on the selected execution provider (EP).
- The extra model, session, and KV-cache memory fit on the device.
- The workload does not repeat enough for history lookup to cover much of the output.

Use **n-gram decoding** when:

- Loading a second model is undesirable.
- Prompts or generated outputs contain repeated local patterns.
- Code, structured output, summarization, translation, or retrieval context can provide reusable
  continuations.
- Occasional lookup misses and standard single-token fallback are acceptable.

Neither method is guaranteed to improve performance. Proposal cost, verification cost,
acceptance, n-gram lookup coverage, model size, EP behavior, and hardware all affect throughput.
Measure end-to-end and decode-only performance on the intended workload.

## Python usage

The generation loop remains unchanged. Select the strategy through the model configuration and
speculative options.

```python
import onnxruntime_genai as og

model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
input_ids = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(
    max_length=len(input_ids) + 256,
    do_sample=False,
)
params.set_speculative_options(**speculative_options)

generator = og.Generator(model, params)
generator.append_tokens(input_ids)

while not generator.is_done():
    generator.generate_next_token()

output_ids = generator.get_sequence(0)
stats = generator.get_speculative_stats()
```

For fixed-width base speculative decoding:

```python
speculative_options = {
    "max_draft_tokens": 4,
}
```

For n-gram decoding with chained lookup and adaptive proposal width:

```python
speculative_options = {
    "ngram_size": 4,
    "max_draft_tokens": 8,
    "ngram_chained_lookup": True,
    "min_adaptive_k": 2,
    "cooldown": False,
}
```

`GeneratorParams.get_speculative_options()` returns the effective configuration stored in
`GeneratorParams`. It does not return the controller's current adaptive width; use
`generator.get_speculative_stats()["effective_k"]` for that value.

## Engine caller-supplied proposals

Dynamic Engine clients can attach a proposed continuation to one request:

```python
# After Run has produced this request's first token event, prefill is complete.
event_buffer = engine.create_event_buffer(8)
max_drafts = engine.max_draft_tokens_per_proposal()
if max_drafts:
    request.set_draft_tokens(np.asarray(draft_ids[:max_drafts], dtype=np.int32))

while engine.has_pending_requests():
  events = engine.run(event_buffer)
  for event in events:
    if event.flags & og.EngineEventFlags.TOKEN:
      consume(event.token)
    if event.flags & og.EngineEventFlags.TURN_FINISHED:
      finish(event.finish_reason)
```

Process every event in the reusable buffer before calling `run` again. If one model execution
produces more events than the buffer holds, subsequent `run` calls drain the retained events before
the Engine performs another model execution.

The corresponding C APIs are `OgaEngineMaxDraftTokensPerProposal` and
`OgaRequestSetDraftTokens`; the C++ wrappers are `OgaEngine::MaxDraftTokensPerProposal` and
`OgaRequest::SetDraftTokens`.

This API has a different ownership boundary from the automatic `Generator` strategies:

- The application owns proposal generation; the Engine only verifies the supplied IDs.
- The request must have completed prefill and be ready to decode; only dynamic batching is
  supported.
- Verification is greedy and accepts the longest prefix matching the target model's argmax rows.
- Guidance and logits penalties are not supported for a request carrying a proposal.
- The proposal belongs to the current turn and applies to its next committed decode operation.
  Budgeting or cache pressure may verify fewer drafts and still consumes the proposal. A
  rolled-back operation preserves it for retry, while canceling the turn discards it.
- One verification run can publish several ordered token events: accepted drafts followed by one
  target replacement or bonus token.

`max_draft_tokens_per_proposal()` returns zero when the model emits only one logits row per request
or its cache/state cannot commit a partial speculative prefix. A positive value is the per-request
capability limit; the scheduler may use a smaller width for a particular operation.

## Model configuration

### Base speculative decoding

Adding a `model.draft` decoder to `genai_config.json` enables base speculative decoding. Start
from the target model's configuration and add the complete decoder description from the draft
model. The draft filename is resolved relative to the composite configuration directory.

The relevant shape is:

```json
{
  "model": {
    "decoder": {
      "filename": "../target/model.onnx"
    },
    "draft": {
      "filename": "../draft/model.onnx"
    }
  },
  "speculative": {
    "max_draft_tokens": 4
  }
}
```

This fragment omits required model architecture, input/output, token, and session fields. Copy
the complete target `model.decoder` block and complete draft decoder block rather than using the
fragment as a standalone configuration. For an end-to-end example, see
[`test/python/models/test_speculative_decoding.py`](../test/python/models/test_speculative_decoding.py).

The target and draft must:

- Have the same static positive logits vocabulary dimension and compatible token IDs.
- Use the same tokenizer files and special-token mapping.
- Use identical EP lists, provider options, and device-filtering options.
- Be plain decoder-only text models with separately rewindable state.
- Return the logits and KV-cache inputs/outputs described by their decoder blocks.

Cross-EP execution, such as a CPU draft with a CUDA target, is not supported.

### N-gram decoding

N-gram decoding uses an ordinary decoder-only model. Enable it in `GeneratorParams` as shown
above or in `genai_config.json`:

```json
{
  "speculative": {
    "ngram_size": 4,
    "max_draft_tokens": 8,
    "ngram_chained_lookup": true,
    "min_adaptive_k": 2,
    "cooldown": false
  }
}
```

N-gram decoding cannot be combined with a configured draft model. The lookup indexes only tokens
that are already committed to the public sequence. A miss runs one standard decoding step.

## Speculative options

The `speculative` object in `genai_config.json` and the runtime speculative-option APIs use the
same names.

| Option | Type | Default | Valid values | Behavior |
| --- | --- | --- | --- | --- |
| `max_draft_tokens` | Number | `4` | `1` to `16` | Fixed proposal width when `min_adaptive_k` is `0` |
| `ngram_size` | Number | `0` | `0`, or `2` to `16` | `0` disables n-gram decoding; otherwise matches the last `N-1` tokens |
| `ngram_chained_lookup` | Boolean | `false` | `true`, `false` | Repeats exact lookup over a local synthetic context to fill more of the proposal budget |
| `min_adaptive_k` | Number | `0` | `0` to `16` | `0` disables adaptation; a positive value is the starting width and floor, with a hard ceiling of `16` |
| `cooldown` | Boolean | `false` | `true`, `false` | At minimum K, three completed zero-accept rounds trigger one standard decoding step |

When `min_adaptive_k` is positive, `max_draft_tokens` is not the adaptive controller's ceiling.
The controller starts at `min_adaptive_k`, probes adjacent widths, and may grow to 16. When
`min_adaptive_k` is zero, `max_draft_tokens` remains the fixed width for the generator lifetime.

`ngram_chained_lookup` requires `ngram_size` to be enabled. Boolean values in JSON must be JSON
Booleans, not numeric `0` or `1` values.

## Adaptive proposal width

Adaptive K is shared by both methods. It learns smoothed committed-token throughput and acceptance
from completed rounds that filled their proposal budget and have valid timing. Lookup misses,
truncated proposals, interrupted rounds, and discarded buffered output do not train the policy.

The controller:

- Gives each new observation 25% weight in an exponentially weighted moving estimate.
- Probes `K-1` when acceptance is below 50% and K is above the configured floor.
- Probes `K+1` when acceptance is at least 75% and K is below 16.
- Keeps a probe after two valid observations when throughput is at least 97% of the prior width;
  upward probes must also retain at least 75% acceptance.
- Rejects a probe immediately when throughput falls below 80% of the prior width.
- Waits one valid round after a successful probe or six after a rejected probe before probing
  again.

These values are general heuristics rather than settings fitted to one model or EP. Fixed K is
useful for controlled benchmarking; adaptive K is useful when one fixed width must serve varied
prompts.

## Supported scope

Both methods support the regular single-sequence generator workflow, including greedy decoding,
non-beam sampling, guidance when ONNX Runtime GenAI is built with guidance support, append,
rewind/resume, and external logits APIs.

| Capability or requirement | Base speculative decoding | N-gram decoding |
| --- | --- | --- |
| Batch size | `1` | `1` |
| Beam search | Not supported | Not supported |
| Return sequences | One | One |
| Model type | Plain decoder-only text target and draft | Plain decoder-only text target |
| Target logits | Full multi-token logits preferred; last-token-only output uses a sequential fallback | Full multi-token logits required |
| KV state | Target and draft must be rewindable | Target must be framework-managed and rewindable |
| EP configuration | Target and draft must match exactly | One target model; EP must support variable-width multi-token decode and rewind |
| Pipeline, vision, or audio models | Not supported | Not supported |
| Physically sliding KV cache | Not supported | Not supported |
| Hybrid SSM/attention state | Not supported | Not supported |
| Model-managed state | May be used only when its rewind behavior satisfies the base path | Not supported |
| Pruned last-token-only logits | Functional through sequential target verification, without the intended batched-verification benefit | Not supported |
| Automatic proposing in Engine and continuous batching | Not supported | Not supported |

Caller-supplied greedy proposals are available through the separate Engine request API described
above; this does not enable either automatic proposer in the table.

Base speculative decoding also rejects `no_repeat_ngram_size != 0`. This search option is distinct
from the `ngram_size` option that enables n-gram decoding.

Do not infer support from an EP name alone. Some EP configurations use model-managed state, fixed
input shapes, pruned logits, or cache layouts that do not satisfy these contracts. Functional
support and measured speedup should be recorded for the exact model export, precision, EP version,
provider options, and hardware.

## Tuning

Start with conservative settings and measure against the same target-only baseline.

### Base speculative decoding

1. Start with the smallest compatible draft model and fixed `max_draft_tokens=2` or `4`.
2. Compare decode throughput, end-to-end latency, acceptance, and memory with the target alone.
3. Increase K only when the accepted-token benefit exceeds the added draft and verification cost.
4. Try `min_adaptive_k=2` when prompt behavior varies enough that one fixed K is unreliable.
5. Enable cooldown only if repeated zero-accept rounds create measurable overhead.

A larger draft can increase acceptance while reducing speedup because every proposed token costs
more. Optimize committed tokens per unit time rather than acceptance alone.

### N-gram decoding

1. Start with `ngram_size=3` or `4` and `max_draft_tokens=4`.
2. Track lookup hit rate, accepted proposals, speculative output coverage, and target throughput.
3. Enable `ngram_chained_lookup` when matching continuations are often shorter than K.
4. Try `min_adaptive_k=2` after confirming that lookup coverage is meaningful.
5. Compare code/structured workloads separately from open-ended conversation and writing.

Lower n-gram orders usually match more often but can propose less precise continuations. Higher
orders usually match less often but can have better conditional acceptance. Lookup coverage and
acceptance must be considered together.

## Statistics

`Generator.get_speculative_stats()` returns a snapshot. Counters are cumulative for the generator;
`tokens_buffered`, `active_rounds`, `effective_k`, `cooldown_remaining`, and
`adaptive_k_throughput` also describe current state.

### Core work and delivery

| Field | Meaning |
| --- | --- |
| `rounds` | Speculative rounds started |
| `completed_rounds` | Rounds whose buffered output was fully delivered and finalized |
| `interrupted_rounds` | Rounds invalidated by an external operation before normal completion |
| `active_rounds` | `1` while a round has buffered output, otherwise `0` |
| `draft_tokens_proposed` | Tokens produced by the active proposer; for n-gram decoding these are lookup proposals |
| `draft_tokens_evaluated` | Proposed token positions logically resolved before the committed output boundary: the accepted prefix plus the first rejected proposal when rejection ends verification. Engine stops counting at a stop string or token limit, so later proposal positions are not counted. |
| `draft_tokens_accepted` | Proposed tokens accepted directly |
| `correction_tokens` | Target-selected tokens committed after a rejection |
| `bonus_tokens` | Target-selected trailing tokens committed after full proposal acceptance |
| `tokens_queued` | Output tokens placed in the internal delivery queue |
| `tokens_emitted` | Queued tokens made externally visible |
| `tokens_discarded` | Buffered tokens invalidated before delivery |
| `tokens_buffered` | Tokens currently waiting for delivery |
| `draft_forward_passes` | Draft-model executions; always zero for n-gram decoding |
| `target_forward_passes` | All target verification, re-anchor, and lifecycle reconciliation executions |

`acceptance_rate` is `draft_tokens_accepted / draft_tokens_evaluated`. It is zero until at least
one proposed token has been evaluated. An Engine round truncated by a stop string or token limit can
have an acceptance rate of `1.0` even when later proposed tokens were not evaluated.

### Round and target execution breakdown

| Field | Meaning |
| --- | --- |
| `full_accept_rounds` | Completed verification rounds that accepted every evaluated proposal token |
| `partial_accept_rounds` | Rounds that accepted a non-empty prefix before rejection |
| `zero_accept_rounds` | Rounds that rejected the first evaluated proposal token |
| `target_verify_forward_passes` | Batched target verification executions |
| `target_reanchor_forward_passes` | Target executions used to establish the next-round anchor |
| `target_reconciliation_forward_passes` | Target executions caused by append, rewind, or other lifecycle reconciliation |
| `total_target_verify_ms` | Time in target verification |
| `total_target_reanchor_ms` | Time in target re-anchoring |
| `total_reconciliation_ms` | Time in lifecycle reconciliation |

### Adaptive K and cooldown

| Field | Meaning |
| --- | --- |
| `effective_k` | Current proposal width |
| `adaptive_k_increases`, `adaptive_k_decreases` | Width changes, including probe moves and reverts |
| `adaptive_k_observations` | Valid rounds incorporated into adaptive estimates |
| `adaptive_k_probes` | Adjacent-width probes started |
| `adaptive_k_throughput` | Current smoothed committed tokens per millisecond used by the controller |
| `cooldown_entries` | Times the zero-accept threshold triggered cooldown |
| `cooldown_steps` | Standard steps completed for cooldown |
| `cooldown_remaining` | Standard cooldown steps still pending |
| `standard_fallback_steps` | Standard steps, including n-gram lookup misses and cooldown |

### N-gram lookup

| Field | Meaning |
| --- | --- |
| `ngram_lookup_hits`, `ngram_lookup_misses` | Exact history lookups with or without a proposal |
| `ngram_lookup_tokens_proposed` | Tokens proposed by n-gram lookup |
| `ngram_chained_tokens_proposed` | Proposed tokens obtained after the first synthetic-context lookup |
| `ngram_grammar_candidate_rejections` | Lookup candidates rejected by active guidance |
| `ngram_history_syncs` | Times committed sequence state was synchronized into lookup history |
| `ngram_history_tokens_synced` | Committed tokens processed during synchronization |
| `total_ngram_history_sync_ms` | Time spent synchronizing history |
| `total_ngram_lookup_ms` | Time spent performing lookup |

### Formula-based estimates

The speedup fields are analytical estimates, not wall-clock benchmark results. `formula_supported`
is true only when at least one round exists and every round supports the formula. Guidance rounds
disable these estimates.

| Field | Meaning |
| --- | --- |
| `total_draft_ms` | Total proposer time, including proposal construction and proposer advancement during finalization |
| `total_target_ms` | Target verification plus target re-anchor time; lifecycle reconciliation is reported separately |
| `avg_draft_ms_per_token` | `total_draft_ms / draft_tokens_proposed` |
| `avg_draft_tokens_per_round` | `draft_tokens_proposed / rounds` |
| `mean_emitted_tokens_per_round` | `tokens_emitted / rounds` |
| `avg_target_ms_per_round` | `total_target_ms / rounds` |
| `target_baseline_ms_per_token` | Mean measured single-token target re-anchor time when such runs exist |
| `target_overhead_ratio` | Relative target cost per round compared with `target_baseline_ms_per_token` |
| `estimated_speedup` | Formula estimate using expected tokens per round from aggregate acceptance |
| `observed_speedup` | The same cost formula using actually emitted tokens per round |

Let `a` be aggregate acceptance and K the proposal width. The expected output tokens for a round
at that width are:

```text
expected_tokens(K) = 1 + a + a^2 + ... + a^K
```

For adaptive K, `expected_tokens_per_round` weights this value by the number of rounds observed at
each K. The implementation then computes:

```text
target_baseline_ms_per_token = total_target_reanchor_ms / target_reanchor_forward_passes
target_overhead_ratio = avg_target_ms_per_round / target_baseline_ms_per_token - 1

denominator = 1
            + avg_draft_tokens_per_round
              * avg_draft_ms_per_token / target_baseline_ms_per_token
            + target_overhead_ratio

estimated_speedup = expected_tokens_per_round / denominator
observed_speedup = mean_emitted_tokens_per_round / denominator
```

Despite its name, `observed_speedup` is still a formula using observed emitted tokens. Use an
external timer and a target-only run for authoritative performance comparisons.

## Benchmarking checklist

- Use the same target model, prompt, output-token budget, search settings, and seed policy.
- Warm up each model and EP configuration before measurement.
- Separate model loading/prefill from decode timing, or report both decode-only and end-to-end.
- Run the target-only baseline without simultaneously retaining an extra target session.
- Report per-prompt results and aggregate speedups with a geometric mean.
- For greedy decoding, compare generated token IDs with the target-only output.
- For sampling, evaluate distributional or task-quality parity across multiple seeds; exact token
  equality is not expected.
- Record model and draft revisions, precision, ORT and GenAI revisions, EP version and options,
  hardware, K/N settings, acceptance, lookup coverage, throughput, latency, and peak memory.

## Current exclusions and future work

The current implementation does not provide speculative decoding through the Engine or continuous
batching APIs. Batch sizes greater than one, beams, multiple returned sequences, multimodal and
pipeline models, physically sliding caches, and hybrid model state require additional design work.
Cross-EP target/draft execution and broader fixed-shape or model-managed-state support also require
explicit state-reconciliation and validation work.

For implementation history, see:

- [Base speculative decoding v0, PR #2233](https://github.com/microsoft/onnxruntime-genai/pull/2233)
- [Base speculative decoding v1, PR #2287](https://github.com/microsoft/onnxruntime-genai/pull/2287)
- [N-gram decoding v0, PR #2314](https://github.com/microsoft/onnxruntime-genai/pull/2314)
- [N-gram decoding v1, PR #2315](https://github.com/microsoft/onnxruntime-genai/pull/2315)