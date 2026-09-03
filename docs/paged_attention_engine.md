# Paged Attention Engine

This document describes how the paged attention engine and continuous batching work in the current codebase. It is intended to be the living architecture guide for this part of ONNX Runtime GenAI.

Update this document whenever a change affects request admission, scheduling, paged KV-cache ownership, model input packing, sampling, transaction behavior, request completion, or failure handling. Historical design and performance work can remain in separate documents, but this document should always describe the behavior that exists on the current branch.

## Scope

The `Engine` can use either static batching or dynamic batching. This document focuses on the dynamic path used by models configured with `engine.dynamic_batching`, because that path provides continuous batching and uses the paged KV cache.

The dynamic path manages paged KV decoder state together with per-request search and sampler state. When the decoder manifest also declares `fixed_conv` or `fixed_recurrent` state groups, `PagedCacheManager` owns a `FixedStatePool` and reserves, stages, and commits its per-request slots inside the same transaction as the paged blocks. `HybridDecoderIO` binds those fixed tensors alongside the existing packed variable-length contract. Execution is selected from manifest capabilities rather than model names, and every Engine-owned mutable state participates in the same transaction boundary.

> **Experimental low-level API:** An Engine creates model-bound Requests. `BeginTurn()` queues both
> initial input and later continuation input and snapshots an opaque `TurnOptions` object. Request
> creation can also snapshot a prompt-plus-generated session-token limit. Successful Turns receive
> nonzero, monotonically increasing request-local IDs; zero means no Turn. `Run()` performs
> synchronous progress and writes zero or more typed `EngineEvent` records into caller-provided
> storage; capacity one preserves one-event pacing. Token and completion payloads are selected by
> event flags.
> `CancelTurn(turn_id)` stops only the named Turn, and `Close()` releases Engine resources.
> `OgaCreateEngine` retains shared ownership of the underlying Model, so the caller may release its
> `OgaModel` handle after successful Engine creation. The Model remains alive until Engine teardown
> and the release of any other retaining objects.
>
> **Single-owner requirement:** One host-owned thread must perform all Engine and Request operations,
> including request creation, `BeginTurn()`, `Run()`, `CancelTurn()`,
> and `Close()`. Other host threads marshal commands and copied inputs to that owner. The
> Engine enforces its owner thread and has no worker thread. Final
> Request-handle release may occur on another thread because it only publishes abandonment work.
> External handle zero/one transitions serialize the C-API self-owner and abandonment state, so a
> concurrent final release and handle reacquisition cannot lose the Request lifetime. The Engine
> strongly retains the Request until cleanup runs at the next serialized Engine boundary.

The main implementation is under `src/engine/`:

| Responsibility | Main files |
| --- | --- |
| Top-level orchestration and transaction handling | `engine.h`, `engine.cpp` |
| Request lifecycle and per-request search state | `request.h`, `request.cpp` |
| Batch planning and request admission | `scheduler.h`, `scheduler.cpp`, `decode_first_scheduler_policy.*` |
| KV-cache capacity planning and ownership | `cache_manager.h`, `cache_manager.cpp`, `paged_key_value_cache.h`, `paged_key_value_cache.cpp` |
| Speculative cache reservation | `paged_cache_reservation.h`, `paged_cache_reservation.cpp` |
| Fixed recurrent and convolutional state slots | `fixed_state_pool.h`, `fixed_state_pool.cpp` |
| Packed model inputs and logits selection | `decoders/varlen_decoder_io.h`, `decoders/varlen_decoder_io.cpp` |
| Model execution boundary | `model_executor.h`, `model_executor.cpp`, `decoders/simple_decoder.cpp` |
| Batched token sampling | `scheduled_requests.h`, `scheduled_requests.cpp` |
| Planned work and transaction outcomes | `step_plan.h` |
| State consistency checks | `engine_invariants.h`, `engine_invariants.cpp` |

## The main idea

Continuous batching allows requests to enter and leave the active batch independently. The engine does not create one fixed batch and run it until every sequence is finished. Instead, every engine step builds a new batch from the requests that can make progress at that moment.

Each request keeps its own sequence, search options, random state, completion state, and sequence-length counters. The engine combines only the work needed for the current model invocation.

The paged KV cache makes this practical. A request does not need one large contiguous cache allocation sized for its maximum sequence length. It owns a block table that points to smaller physical KV-cache blocks. Blocks are added as the sequence grows and returned to the pool when the request leaves the engine.

At a high level, one dynamic engine step is:

```text
drain pending Engine events up to caller capacity, if any exist
    |
plan a runnable batch
    |
reserve all paged and fixed decoder state needed by that plan
    |
preflight event publication bookkeeping
    |
checkpoint request and sampler state
    |
pack request tokens into one variable-length model input
    |
run the ONNX decoder once
    |
sample and stage one result per request
    |
commit search state, cache state, and request bookkeeping
    |
write committed events up to caller capacity and retain overflow
```

The step is transactional before publication. Planning, reservation, execution, and fixed-state
preparation do not change committed state; recoverable failure restores request/search state and
releases all reservations. Once publication begins, containment is **fail-stop**, not rollback:
request/search, paged, fixed, and request-bookkeeping publication is ordered and prevalidated, but
an unexpected internal failure may occur after an earlier component publishes. The Engine then
becomes permanently unhealthy rather than exposing the partially published state to further steps.

## How the dynamic path is selected

`Engine::CreateDependencies()` creates three collaborators from the model:

1. A `CacheManager`.
2. A `Scheduler` that uses that cache manager.
3. A `ModelExecutor` that also uses that cache manager.

When `model->config_->engine.dynamic_batching` is present:

- `CacheManager::Create()` returns `PagedCacheManager`.
- `PagedCacheManager::SupportsDynamicBatching()` returns `true`.
- `Scheduler::Create()` returns `DynamicBatchScheduler`.
- `SimpleDecoder` uses `VarlenDecoderIO`.
- `Engine::Run(output)` calls `RunDynamic()` when no retained events exist.

Dynamic batching limits scheduled rows with `max_batch_size` and limits the total
query tokens in one model run with `max_scheduled_tokens`. The token limit defaults
to 2048. Both limits are positive and independent.

Without dynamic batching, the engine uses the older static batching path. Static batching allocates and advances a batch as a unit. It does not use the transaction flow described below.

## Decoder state manifest

`model.decoder.state_groups` can describe decoder-owned state without identifying a model family. The supported group kinds are `paged_kv`, `fixed_conv`, and `fixed_recurrent`. Each group contains logical layer IDs. Tensor name templates live in the decoder's existing `inputs` and `outputs` objects: paged KV uses the key/value templates, while the two fixed kinds use their corresponding convolution or recurrent templates. Every referenced template contains one `%d` placeholder for the logical layer ID. A layer may own more than one group, and layers omitted from a group do not own that kind of state.

The group kind defines the state transition:

- Paged KV grows by appending token slots and commits by advancing logical occupancy.
- Fixed convolution and recurrent state replace one request-indexed state value and require a staged output to be published at commit.

Configuration loading rejects unknown kinds, malformed decoder templates, duplicate IDs or logical layers, and conflicting resolved names. Overlay application validates a complete copy and publishes it only on success. When `state_groups` is absent, the typed configuration retains the legacy dense paged-KV behavior over `0..num_hidden_layers-1` using the existing decoder key/value templates; it does not insert a synthetic group into the parsed configuration.

When an explicit manifest is present, model loading resolves each group's decoder templates, expands every name, and verifies that its decoder input and output exist with compatible dtype and shape. Paged tensors must also have compatible rank-four geometry throughout their group.

The dynamic Engine requires exactly one `paged_kv` group. It allocates cache tensors only for that group's logical layer IDs, expands their exact binding names without renumbering, derives the cache dtype from the first validated key input, and sizes an automatic block pool using the number of participating full-attention layers after reserving storage for participating sliding-window layers. Every configured sliding-window layer must belong to the paged group. Multiple paged groups are rejected because the Engine currently owns one shared paged pool. The synthesized legacy group preserves dense sequential behavior when no explicit manifest exists.

`fixed_conv` and `fixed_recurrent` groups may accompany the required `paged_kv` group. `ValidateDynamicEngineCompatibility` accepts them (it still rejects any configuration without exactly one `paged_kv` group), and `PagedCacheManager` constructs a `FixedStatePool` with `max_batch_size` slots when at least one fixed group is present. The composite manager reserves and commits fixed slots together with the paged blocks (see "Fixed request-state pools" below), while `HybridDecoderIO` binds either direct bank views or gathered/staged fallback tensors alongside the packed variable-length inputs. Expanded fixed bindings must resolve to real session inputs and outputs; the pool rejects absent or incompatible bindings.

Fixed state currently requires `engine.dynamic_batching`; static batching is rejected because
`StaticBatchDecoderIO` does not own request-indexed fixed-state bindings.

## Request lifecycle

A `Request` is one sequence. Engine requests currently require:

- `search.batch_size == 1`
- `search.num_beams == 1`
- At least one input token in every `BeginTurn()`

The engine creates throughput by batching several independent requests, not by placing several sequence rows inside one request.

The important request states are:

```text
Unassigned (Created) -- BeginTurn(tokens, turn_options) --> Assigned (Queued) -- schedule --> Active
                              ^                                           |
                              |                                           | turn stops
                              +----- BeginTurn(tokens, turn_options) -- TurnComplete

Unassigned (Created) -+
Assigned (Queued) ----+
Active ---------------+-- Close() --> Closed
TurnComplete ---------+
```

`CancelTurn(turn_id)` moves the matching assigned or active Turn to `TurnComplete`, preserving
committed state and publishing one `TurnFinished/Cancelled` event. It is an owner-thread operation
between `Run()` calls, not a cross-thread interrupt. Stale, unknown, and completed IDs succeed with
`false`; cancellation before the first Turn or after `Close()` is an error.

### `Unassigned`

The request is already bound to its creating Engine but has no scheduler or cache membership.
`BeginTurn()` copies the initial input, snapshots the turn options, and establishes submission
order. Request-level generation parameters were snapshotted at creation and cannot change between
turns. The caller does not need to keep either the input or options storage alive after the call.

First-turn admission is transactional. Device input allocation, Search append, scheduler storage,
and per-request sampler acquisition must all succeed before the new Turn is committed. The
scheduler reserves its request container before it creates sampler state, and its final insertion
and sampler-state install are nonthrowing. A failure restores the Search checkpoint, host token
mirror, counters, status, and turn metadata, leaving the Request `Unassigned` and retryable without
duplicating the prompt.

### `Assigned`

This is the queued state. A successful first or later `BeginTurn()` moves a Request here while its
copied input waits for execution. `BeginTurn()` is rejected while already queued or active. Input must leave room for at least one generated token below the request's internal
`max_total_tokens`.

Public `max_session_tokens` (stored internally as `max_total_tokens`) is the cumulative total
sequence limit for the entire Request: the initial
prompt, generated output, and every continuation input all count against the same limit. It defaults
to `max_length`, cannot exceed `max_length`, and is not reset by `BeginTurn()`.

Opaque nullable `TurnOptions` are separate. Null, or setting `max_generated_tokens` to zero, means
that no per-Turn limit applies beyond the cumulative Request limit. The current Turn counts only
generated tokens appended as output. Initial
input, continuation input, and the previously generated token replayed during continuation prefill
do not count. A later turn may begin whenever its input still leaves room for at least one generated
token under the cumulative limit, and it may choose a different per-Turn limit.

`OgaRequestOptions` is an opaque, reusable handle. Null options and zero
`max_session_tokens` use the Request's snapshotted `OgaGeneratorParams.search.max_length`. That
search value normally defaults from the model context length, but a caller may set it lower before
Request creation. `OgaTurnOptions` is opaque and reusable; `BeginTurn()` snapshots supported values.

### `Active`

The current turn is executable and owned by the Engine. It normally has one unprocessed token at
the beginning of a decode step: the token sampled by the previous step.

### `TurnComplete`

The current Turn reached an end condition. A `TurnFinished` event reports `Eos`, `StopString`,
`MaxGeneratedTokens`, `MaxSessionTokens`, `Cancelled`, or `Failed`, plus per-Turn usage. For the
same generated token, precedence among the reasons Request-level classification can choose between
is `StopString`, then `MaxGeneratedTokens`, then `MaxSessionTokens`: a token that would otherwise
end the turn on the turn or context limit still reports `StopString` when it also completes a
match. `Eos` sits outside that ordering: Search decides whether a sampled token is EOS and never
appends it to a single-sequence Request's history before Request-level stop-string classification
ever runs, so an EOS token's bytes never reach the matcher. This is not a practical gap -- GenAI
tokenizers default to `skip_special_tokens=true` and register EOS as a special added token, so a
real EOS token always decodes to zero bytes and could never independently complete a stop match. A
generated EOS token is not appended to the logical sequence or emitted as a token event; a
stop-matching token is appended, emitted, and counted like any other generated token, even though
bytes after the match inside that token's decoded text are never trimmed from it.
The next continuation fragment is therefore responsible for any turn-boundary tokens required by
the model's chat template.

`BeginTurn(tokens, turn_options)` appends the next input fragment and moves a resident Request back
to `Assigned`. The completed Turn's terminal event must first be drained; calling `BeginTurn()`
while any event for the Request is pending fails without mutation. Turn-scoped
generated count and limit reset only after all validation, input allocation, Search append, and
scheduler preparation succeeds.

Request Rewind has not yet been implemented. Cancellation does not rewind the Search sequence or
committed cache, so accepted continuation input and generated output remain part of the logical
request. A canceled resident request can begin a later turn after its terminal notification is
drained. A first turn canceled before admission has no cache state, but it can likewise begin a
later turn: the retained initial input and new continuation input are prefetched together. If
retained model state has otherwise ceased to be resident, continuation still fails. Until Request
Rewind is implemented, callers that need to discard canceled input or release capacity must
`Close()` and create a new Request.

Every completed Turn publishes exactly one terminal event. A final visible token and completion may
be combined in one event. An unserviceable Request receives `TurnFinished | Failed`; fatal Engine
failure emits `TurnFinished | Failed` for every affected Turn and makes later progress calls
rethrow the stored failure. A fatal failure with no affected Turn returns a request-null `Failed`
event.

A turn-complete request remains Engine-owned. Once admitted, it normally remains resident; a turn
canceled before initial admission has not allocated cache state. `Close()` ends logical ownership
immediately and releases dynamic cache resources; static batch storage may remain until batch
recycling. If every external handle is released instead, the request is marked abandoned and
reclaimed at the next serialized Engine boundary, including `HasPendingRequests()`. An atomic
Engine-level pending flag makes the common no-abandonment boundary allocation-free.

Planning skips turn-complete residents and does not release their cache. Retained requests still
consume paged-cache blocks and a batch slot, so applications must call `Close()` when they no
longer need continuation. Dynamic resources are reclaimed immediately; static resources remain
until the shared batch is recycled.

### `Close()`

`Close()` is legal from every state, is sequentially idempotent, and moves the request to terminal
`Closed`. The Engine immediately excludes it from scheduling and removes any undrained pending
events for that request. On the dynamic path, close also immediately releases committed paged-cache
ownership and, for a composite model, the request's committed fixed-state slot. Search, guidance,
sampler, parameter, and token-mirror state are then released before the Engine drops its strong
Request reference.

A resident static row cannot release all runtime objects independently because the static decoder
and shared cache allocation still read its Search sequence, parameters, host token mirror, and
length metadata while another row executes. Static close releases sampling-only state immediately
but retains that row-essential state, without logical execution or event delivery, until the
complete static batch is recycled on the Engine owner thread. The Engine then releases the retained
runtime state and drops its strong Request reference, leaving any surviving public handle as a
lightweight closed tombstone. A nonresident static Request completes physical close immediately.
Final-handle abandonment performs the same logical removal and event purge at the next owner-thread
boundary and follows the same dynamic-immediate or static-retained physical cleanup.

Events already copied by the host remain host-owned. Destroying the Engine terminalizes every bound
Request through a teardown-specific no-throw detach path that does not rely on the Request's expired
weak Engine reference. Scheduler/cache ownership and Request runtime state are released first;
surviving external handles are lightweight closed tombstones and still must be destroyed.

### `Closed`

`Closed` is distinct from `Unassigned` because close may already have destroyed residency and
scheduler ownership. Returning to `Unassigned` would imply that the same logical sequence could be
submitted as a new request. A closed static-batch row may remain physically allocated until the
shared batch is recycled, but it is no longer sampled or returned.

## The request length counters

Request progress uses these values:

| Value | Meaning |
| --- | --- |
| `CurrentSequenceLength()` | Number of tokens currently held by the request's search sequence |
| `processed_sequence_length_` | Number of sequence tokens already represented in the committed KV cache |
| `turn_prompt_tokens_` | Input tokens accepted by the current Turn |
| `turn_generated_tokens_` | Generated output tokens committed during the current turn; reset only by a successful `BeginTurn()` |
| `current_turn_id_` | Request-local `uint64_t` ID assigned after successful Turn admission; starts at 1 and increases monotonically; 0 means no Turn |
| `finish_reason_` | Strongly typed reason for current-turn completion; `None` while generation is in progress |

Turn IDs use the nonzero `uint64_t` range. Once `UINT64_MAX` has been assigned, a later
`BeginTurn()` fails before mutation rather than wrapping to zero.

The turn also snapshots an optional `max_generated_tokens`. This is Request bookkeeping, not
Search's cumulative sequence limit. Transaction staging computes whether the next generated count
completes the turn without changing committed Request state; `CommitStep()` publishes the count
together with the token, request status, and processed length.

`tokens_host_` mirrors the complete logical sequence, including prompt, generated output, and
continuation input. Public generated-token delivery is event-based; committed event records capture
the token, Request, and Turn ID together.

The unprocessed tokens are:

```text
[processed_sequence_length_, CurrentSequenceLength())
```

The dynamic plan sends a non-empty prefix of that interval. For a newly
assigned request, `processed_sequence_length_` is zero, so the whole prompt is
initially unprocessed and the first model run is prefill.

Partial prefill commits advance `processed_sequence_length_` by the planned
count without sampling. When the final prefill contribution commits, the prompt
is marked processed and the newly sampled token is appended to the search
sequence. That sampled token is not in the KV cache yet, so it becomes the one
unprocessed token for the next decode step.

The same pattern repeats during decoding:

```text
before the model run:
    cache contains tokens [0, N)
    search contains tokens [0, N + 1)
    token N is the current model input

after a successful commit:
    cache contains tokens [0, N + 1)
    search contains tokens [0, N + 2), unless generation completed without appending
    token N + 1 is the next model input
```

This separation between search length and processed length is what lets each model run consume the token produced by the previous run.

## `Engine::Run()` and event draining

The low-level engine API advances through repeated calls to synchronous bulk `Run()`. The canonical
C operation receives an Engine-bound `OgaEngineEventBuffer` whose capacity was allocated once.
After Run, count and indexed access expose borrowed opaque `OgaEngineEvent` views. The C++ wrapper
provides RAII Buffer ownership and borrowed event/usage getters. Python exposes
`Engine.create_event_buffer(capacity)` and a sequence-like `Engine.run(buffer)` result.

The Buffer is permanently bound to its creating Engine and stores only a weak Engine reference.
Run requires both handles to remain live and rejects a Buffer from another Engine. After handle,
binding, and owner-thread validation, Run invalidates every previously borrowed event and usage view
from that Buffer, resets its count, and reuses the preallocated internal event storage without
constructing a per-Run
vector.

Capacity zero is an owner-thread-validated no-op: count remains zero and no reclamation, draining,
scheduling, or model work occurs. A permanently unhealthy Engine still returns its stored fatal
error.

A positive-capacity call is strictly drain-or-execute:

1. Validate the owner thread and reclaim abandoned Requests.
2. If `pending_events_` contains retained output, drain up to capacity in FIFO order and return.
   Do not execute another model transaction even when the caller buffer has spare slots.
3. Otherwise execute at most one static step or one dynamic transaction.
4. Move up to capacity produced events into the Buffer and retain overflow in `pending_events_`.

A transient allocation failure while reclaiming abandoned Requests leaves completed cleanup intact,
re-arms reclamation, and retries the remaining work at the next owner-thread boundary. An ownership
invariant failure during reclamation is fatal: `Run()` drains the retained terminal failure events
through the supplied Buffer before later calls rethrow the stored Engine error.

When capacity covers the complete affected batch, one call returns all of that transaction's
events. Capacity one expresses one-event-at-a-time behavior through the same implementation: the
first call executes once and later calls drain retained overflow without more model execution.
Each retained event owns its Request internally until it is copied; the public pointer remains
borrowed. A Buffer event holds the Request internally for the lifetime of that view, but the
returned `const OgaRequest*` is never an independently owned handle and is invalidated with the
event.

A committed step emits zero or more ordered token events and may emit a terminal event or combine
`TurnFinished` with its final visible token event. EOS can emit terminal completion without a
visible token. Event delivery does not mutate the Request's retained logical sequence. A
speculative step emits each accepted draft and its correction or bonus as a separate token event,
up to `kMaxGeneratedTokensPerStep` events per affected Request. If speculative verification accepts
a visible draft immediately before an accepted EOS ends the Turn, the accepted visible token and
terminal completion share that combined event; the EOS itself is never emitted as a token.

A chunked partial-prefill transaction can commit cache and Request progress without producing an
event. `Run()` then returns count zero while `HasPendingRequests()` remains true. It does not loop
internally to force a token event.

This distinction is important:

- One positive-capacity call either drains retained events or attempts at most one model
  transaction.
- Draining a previously committed batch does not change model or cache state.
- `Request::Close()` purges undrained events for that Request.
- `HasPendingRequests()` first reclaims abandoned Requests, then is true while either the pending
  event queue or scheduler contains work.

If the engine has previously encountered a fatal transaction or execution failure, `Run()` rethrows
the stored error instead of attempting more work.

## Speculative drafts

The dynamic Engine can verify a continuation together with a request's next decode token. A caller
can propose tokens before the next step with `Request::SetDraftTokens` (`OgaRequestSetDraftTokens`
in C or `Request.set_draft_tokens` in Python). When `model.mtp` names an auxiliary draft head, the
Engine also maintains an internal shadow Request and automatically proposes a chained greedy
continuation after each committed target step. Alternatively, `model.dflash2` runs a block drafter
over the target's packed auxiliary hidden states. A model cannot configure both automatic drafters.

Query the internal `Engine::MaxDraftTokensPerStep` capability through
`OgaEngineMaxDraftTokensPerProposal`, `OgaEngine::MaxDraftTokensPerProposal`, or
`Engine.max_draft_tokens_per_proposal` before proposing work. Zero means that the decoder does not
return one logits row per packed token or that its cache/state cannot commit an accepted prefix.
The reported value is a capability limit, not a guarantee that every proposed token fits the next
step's global token budget or current cache capacity. Automatic drafters also leave room for the
target's correction token and cap each proposal by the Request's session limit and remaining Turn
token budget.

The request must already belong to the Engine, have completed prefill, and be ready to decode.
Verification supports greedy target selection and random target sampling with a positive `top_k`;
proposals remain deterministic. Guidance,
repetition penalty, no-repeat-ngram processing, and active minimum-length processing are not
supported. Passing an empty sequence clears a pending proposal.

For a decode with K scheduled drafts, the packed input is the request's one unprocessed token
followed by the K drafts. The decoder must return K+1 logits rows for that request. Row `i` predicts
draft `i`; verification accepts the longest prefix whose tokens equal the target argmax. The first
nonmatching row supplies the replacement token, or row K supplies a bonus token when every draft is
accepted. Under random target sampling, each target row is sampled and the deterministic draft is
accepted when it matches that sample; the first mismatch is the correction token. Accepted drafts
and the correction or bonus are published together, so one model run can
publish several ordered token events. If the caller's event buffer cannot hold all of them, `Run()`
retains the overflow and drains it on subsequent calls before executing the model again. Callers
must process all returned events before reusing the buffer.

Draft rows are optional scheduler work. Admission first reserves one mandatory token for every
selected request, then distributes remaining `max_scheduled_tokens` and cache capacity to drafts.
If the optimistic draft width does not fit, the scheduler shortens or removes draft work without
dropping another selected request's mandatory decode. Prefill never verifies drafts.

A committed decode consumes the complete proposal, including drafts omitted by budgeting. A
retryable rollback restores the request and leaves the proposal pending for retry. During commit,
the cache reservation is narrowed from the scheduled K+1 slots to the accepted prefix plus the
request's mandatory token; paged KV and fixed recurrent state publish at that same boundary.

### Engine-hosted MTP head: operational contract

`model.mtp` turns the head on automatically for every request the dynamic Engine decodes. Server
authors should size capacity and handle failures against the following behaviors.

**Auxiliary memory accounting.** The head is a second paged pool that always holds the same block
count as the target pool, so both are sized from one budget. With
`engine.dynamic_batching.gpu_utilization_factor`, the auxiliary bytes per block are folded into the
capacity computation. With an explicit `engine.dynamic_batching.num_blocks`, that value is the
combined budget: the target pool is scaled down so target plus head together cost what the
configured block count would have cost without a head. A `num_blocks` too small to leave at least
one block for each pool is rejected at Engine construction.

**Failure isolation.** The target decode is mandatory; MTP drafting is optional acceleration. A
recoverable head failure (cache pressure, shape mismatch, binding or session error) releases only
MTP state, and the target step still commits — without drafts for that step — and increments
`standard_fallback_steps` and `mtp_failures` in the speculative statistics. The first failure is
logged, and three consecutive failures disable automatic MTP drafting for that Engine. Only a
contract violation, or a failure to roll MTP state back, marks the Engine unhealthy. A persistent
head failure therefore degrades to ordinary decoding without repeatedly running a broken head.

**Shadow lifecycle.** The head's shadow Request mirrors only the suffix the target committed during
the current turn. Beginning a continuation and canceling a turn both drop the shadow and release its
auxiliary blocks, so the next drafted step rebuilds it from the new turn's suffix. Closing the
request releases it as well.

**Mixed prefill batches.** Drafts are proposed only for requests that committed a decode token in
the step. A request that was prefilling, finished its turn, or has a proposal the Engine cannot
verify is skipped for that step and picks drafting back up on its next decode step.

**Seeded sampling.** `search.random_seed` reproduces output for a given decode path, not across
decode paths. A verified drafted step draws its target tokens from the Request's own host random
stream because acceptance is sequential, while an ordinary batched step draws from the device
sampler state. Whether drafts are admitted depends on batch composition and cache pressure, so a
seeded run is only bit-reproducible against another run with the same supplied drafts and scheduling
path. Disabling `model.mtp` does not make caller-provided drafts scheduling-independent; exact
repeatability requires replaying the same proposals and schedule.

**Telemetry thread ownership.** `Engine::GetSpeculativeStats` (`OgaEngineGetSpeculativeStats`,
`Engine.get_speculative_stats`) reads counters that `Run()` mutates without synchronization. Like
every other Engine query, it must be called from the Engine's owner thread and throws otherwise.

## One dynamic step in detail

`Engine::RunDynamic()` coordinates the complete transaction.

### 1. Identify active and waiting requests

`DynamicBatchScheduler::PlanStep()` skips `TurnComplete` residents and builds candidates from
executable residents plus waiting requests.

The cache manager checks whether those candidates fit alongside dormant turn-complete requests. If
retained residency prevents admission or cache growth, the plan reports capacity backpressure; the
application decides which conversation to release with `Close()`.

### 2. Build the initial step plan

The scheduler snapshots requests that already belong to the paged cache. Executable residents may
be `Active` or `Assigned`; an `Assigned` resident is a queued continuation.

It then snapshots nonresident waiting requests from the scheduler pool. These are `Assigned` and are marked as newly admitted candidates.

The scheduler orders candidates with decodes first. Order remains stable among
decodes and among prefills. Each candidate initially contributes one provisional
token so cache feasibility can be decided before a large prefill consumes the
global token budget.

For each candidate, the scheduler records a `RequestStepPlan` containing:

- The request identity.
- The sequence length before the transaction.
- The provisional number of unprocessed tokens.
- The number of cache slots required after the model run.
- Whether the work is prefill.
- Whether the request is newly admitted.

At this point, the plan contains candidates. It has not yet been reduced to the requests that fit or expanded to the final token counts.

### 3. Plan paged-cache resources and admission

`PagedKeyValueCache::PlanStepResources()` evaluates the candidate plan against:

- The configured maximum batch size.
- The provisional scheduled-row limit.
- The number of free physical blocks.
- Blocks already committed to active requests.
- Additional blocks each request needs for this step.
- The maximum block-table width supported by graph-capture buffers, when graph capture is enabled.

The planner considers candidates in the scheduler's decode-first order. It may
skip an infeasible candidate and continue to later candidates, so a blocked
prefill does not hide smaller work that can run. It matches residents by request
identity rather than block-table position, and the plan may contain an ordered
subset of residents.

Scheduled rows cannot exceed `max_batch_size`. Separately, all committed
residents, including omitted ones, count toward residency capacity. A new
admission is allowed only when the committed resident count plus selected new
admissions remains within `max_batch_size`.

Selected entries are compacted to the beginning of the plan. Deferred entries remain unchanged in committed state:

- An existing request keeps its current block table and stays `Active`.
- A new request stays `Assigned` and owns no cache blocks.
- Both can be considered again by the next engine step.

The planner distinguishes temporary capacity pressure from a request that can never fit:

- `CapacityDeferred` means the request could run later if capacity becomes available.
- `UnserviceableRequest` means the request would exceed the total cache or supported block-table width even if the cache were otherwise empty.

If at least one request fits, the step is executable even when other requests were deferred or found unserviceable. This allows useful work to continue instead of letting one request block the whole engine.

### 4. Finalize the packed token layout

After cache planning selects the batch, each decode keeps its one-token
contribution. Every selected prefill also keeps its provisional token, then
prefills expand in stable order while tokens remain. A prefill contribution is
bounded by its remaining prompt, `search.chunk_size` when configured, and
`max_scheduled_tokens`.

The scheduler recomputes cache targets and assigns each request:

- `packed_token_offset`: the first row occupied by the request in the flattened token input.
- `logits_row_index`: the row containing logits for the request's final input token.

For example, if the selected requests contribute 3, 1, and 2 tokens:

```text
flat input rows:       [A0 A1 A2 B0 C0 C1]
packed offsets:         0        3  4
selected logits rows:   2        3     5
```

Only the logits for the last input token of each request are used to sample its next token.

`RequestStepPlan::unprocessed_token_count` is authoritative on the dynamic path.
`ScheduledRequests` validates that every count is positive and no larger than
the request's remaining tokens, then binds those counts before decoder input or
chunk-completion logic reads the request.

The scheduler also marks whether the step is eligible for CUDA graph capture. A capturable step must be a pure decode step where every request contributes exactly one token. Prefill and mixed prefill/decode steps have variable token counts and run eagerly.

### 5. Reserve the complete cache transaction

`PagedCacheManager::ReserveStep()` creates a composite reservation: a `PagedCacheReservation` and, when the model has `fixed_conv` or `fixed_recurrent` groups, a `FixedStateReservation`. Both use the same scheduled row order.

The paged sub-reservation:

- Reserves every new physical block required by the selected plan.
- Records how far each request's committed slot count would advance.
- Creates provisional block tables for newly admitted requests.
- Leaves the committed block tables and allocated-request list unchanged.

Reservation is all-or-nothing. If all planned blocks cannot be reserved, the engine treats that as an execution contract failure because planning had already declared the step executable.

The fixed sub-reservation, when present, admits the same rows in the same order. A request that already owns a committed fixed slot keeps it; every other request is admitted provisionally into a free slot. After resource selection and token-budget assignment, an all-resident batch is ordered by fixed slot before packed offsets are calculated. When those slots form a contiguous interval, the pool normalizes any minority bank rows to a canonical active bank, then binds that bank interval directly as model input and the corresponding inactive-bank interval as model output. New admissions and fragmented intervals retain the gather/output staging fallback. Free slots have no visible bank, so admission aligns their bank selector with a coherent resident cohort; this prevents stale parity from a previous owner from unnecessarily disabling the next direct reservation.

Physical execution ordering does not change event ordering. Each row retains its logical scheduler
rank, and the Engine restores that order when it publishes the transaction's events after commit.

The fixed `target_tokens` mirror the paged `target_cache_slots`, so both states commit at one token boundary. `StepPlan::fixed_state` records the fixed row count, new-slot count, and binding footprint; `Engine::RunDynamic()` then proves the reservation matches that plan exactly -- required flag, row count, new-slot count, bytes, and per-row request identity -- and fails fatally on any mismatch. A composite reservation wraps exactly one paged reservation and the Engine holds at most one at a time, so the paged split-commit contract (its constructor reserves its own `committed_tables_` headroom, and the pool permits a single live reservation) holds without any cross-reservation aggregate check.

While the reservation is active, decoder input preparation can view a combined block table containing:

- Blocks already committed to each active request.
- Blocks held by this transaction's reservation.

This allows the model to write into the future cache layout without publishing that layout as committed state.

After reservation, `ScheduledRequests` validates the selected requests and preflights their
generated-output index capacity. If that allocation fails, the reservation is released before
request checkpointing, model execution, or cache commit.

### 6. Checkpoint request and sampler state

Before model execution, `ScheduledRequests::BeginTransaction()` checkpoints every selected request's search state.

If the device supports transactional batched sampling, the scheduler-owned sampler state is checkpointed as well. Each request keeps its own persistent sampler state, including its random stream, even though sampling work can be batched.

These checkpoints allow the engine to undo mutations caused by logits processing and sampling if a later part of the step fails.

### 7. Prepare paged cache state for the model

`DecoderModelExecutor::Decode()` first asks the cache manager to prepare the step.

On the transactional dynamic path, `PagedCacheManager::PrepareStep()` updates the model-facing KV-cache state using the active reservation and the block-table width selected during planning. This does not commit the reservation.

The physical key and value cache tensors are long-lived. What changes per step is the block table that maps each request's logical sequence blocks to physical cache blocks.

`ExecutionContext` also exposes the fixed slot handles, state bindings, and binding-byte count in the exact scheduled request order. `HybridDecoderIO` adopts the complete `VarlenDecoderIO` contract and appends these fixed bindings without changing packed token order or logits processing.

### 8. Pack variable-length model inputs

`SimpleDecoder` chooses `VarlenDecoderIO` for paged-only models and `HybridDecoderIO` (which
composes `VarlenDecoderIO`) when fixed state groups are present.

`VarlenDecoderIO` builds three coordinated inputs:

```text
input_ids                    all requests' unprocessed tokens concatenated
cumulative_sequence_lengths  boundaries of each request in the flat token array
past_sequence_lengths        committed KV length and write position for each request
```

When the graph declares `position_ids`, it also builds one absolute int64 position per packed
token. Request row `i` starts at its committed `past_sequence_lengths[i]`, so a token at local
offset `j` receives position `past_sequence_lengths[i] + j`. Models may consume either `[num_tokens]`
or Qwen MRoPE `[3, num_tokens]` geometry; the latter repeats each text position across all three
MRoPE axes for the text-only decoder path. Multimodal per-axis coordinates are outside the Engine
contract. The StepPlan row, token count, and packed offset are validated before this optional
tensor is copied to the model-input device.
Packed models with this input execute eagerly because the current persistent CUDA graph buffers
do not own stable position-ID storage.

Using the previous example:

```text
input_ids = [A0 A1 A2 B0 C0 C1]
cumulative_sequence_lengths = [0, 3, 4, 6]
past_sequence_lengths = [A_past, B_past, C_past]
```

This representation allows one model invocation to mix requests with different sequence lengths and different amounts of pending work. A step can contain a long prefill for one request and single-token decoding for other requests without padding every request to the same query length.

The decoder also prepares a CPU `int32[3]` attention metadata input:

```text
[max_query_len_bound, max_kv_len_bound, max_kv_len_lower_bound]
```

The first two values are upper bounds. The third is a lower bound on the longest per-request KV sequence and lets the paged attention operator decide whether split-KV decode is worthwhile. Supplying these values lets the operator choose its backend and size its work without reading sequence lengths back from the device. Models using this contract must expose the three-element input; the Engine no longer emits the legacy two-element form. This requires ONNX Runtime commit `0d291bc5d39d8e62150c2c30f174812834344b48` or a later package containing the three-element PagedAttention metadata contract.

Pure-decode steps and mixed prefill/decode steps may select different numerically valid CUDA
backends. Floating-point reassociation can therefore flip a near-tied greedy token when batch
composition changes even though request state and continuation are correct. Qualification should
use continuation/replay invariants, tolerant logits or top-k comparisons, and task-level quality;
bitwise token equality across batch compositions is available only when every selected backend is
batch-invariant.

### 9. Run the decoder once

`SimpleDecoder::Decode()` calls the ONNX Runtime decoder session once for the complete scheduled batch.

The outputs include one logits row for every flattened input token. The decoder state is attached to `ScheduledRequests` so post-processing can select the rows that belong to each request.

For eligible pure decode shapes, the decoder may capture or replay a CUDA graph. Prefill and other variable shapes run eagerly. Eager steps report exact bounds, so the KV upper and lower values are equal. A captured CPU input is read only while the graph is captured, so captured steps instead report bounds valid for every replay in the graph's block-table bucket. Capturable decode steps reserve exactly `ceil(current_kv_length / block_size)` blocks, allowing the Engine to report query upper bound `1`, KV upper bound `block_table_columns * block_size`, and KV lower bound `1` for the minimum or a capacity-clamped sub-minimum bucket, or `(preceding_power_of_two_columns * block_size) + 1` for larger and truncated final buckets. If a future reservation policy allocates ahead of the live KV length, the Engine conservatively reports a lower bound of `1`.

### 10. Select logits and stage next tokens

`VarlenDecoderIO::ProcessLogits()` uses each plan entry's `logits_row_index` to return one vocabulary-sized logits span per request.

`ScheduledRequests::GenerateNextTokensForTransaction()` then:

1. Applies each request's logits processors to its own logits row.
2. Samples a next token using the scheduler-owned batched sampler when supported, otherwise uses the per-request search path.
3. Runs the per-request sequence and EOS handling.
4. Produces a `RequestStepResult` for each request.

The reusable batched-sampling plan records each sampled Request's scheduled result index. Batched
completion therefore places results directly in linear time without searching the scheduled batch
again for every sampled Request.

The result records:

- The sampled token, if a token was appended.
- Whether a token was appended.
- Whether the turn is complete.

These results are staged. The request's host token mirror, processed-length counter, per-turn
generated count, and internal lifecycle state are not updated yet.

Requests that appended a token or became complete produce staged events. Events are not exposed until the complete transaction commits.

### 11. Commit the transaction

The commit order in `Engine::RunDynamic()` is deliberate:

1. Prepare the reservation: validate every ownership and capacity precondition of both sub-reservations. A fallback reservation copies each fixed staged output into its slot's inactive persistent bank. A direct reservation has already written full-step outputs there; if speculative verification keeps only a prefix, compact replay reconstructs the accepted state from the untouched active bank into that same inactive bank. Preparation synchronizes without publishing anything. This is the last step that performs fallible device work, and a failure here is fatal even though committed state is intact, because a partially written inactive bank cannot be proven consistent for a retry. The publish steps below do only host-side work; they can still throw on a state-machine misuse (for example a double publish), and the Engine treats any such throw as fatal too.
2. Commit the request search checkpoints and sampler checkpoint.
3. Publish the reservation: paged occupancy first, then the fixed bank flip.
4. Commit each request's lightweight bookkeeping.
5. Publish the staged events.

Everything after step 1 crosses the commit boundary and is never retried. This is a fail-stop
publication interval: a throw is fatal and the Engine is not used again; already published
components are not rolled back.

Committing the cache reservation:

- Flips each selected fixed slot to its freshly prepared bank, advances its state generation and committed token boundary, and publishes provisional fixed-slot ownership (`FixedStateReservation::PublishCommit()` is `noexcept`).
- Adds reserved blocks to existing block tables.
- Adds provisional block tables for newly admitted requests.
- Advances committed cache-slot counts.
- Adds newly admitted requests to the cache manager's allocated-request list.

Paged publication (`PagedCacheReservation::CommitValidated()`) is deliberately not `noexcept`, but for this single validated reservation it performs no fallible allocation or device work, so it cannot leave paged and fixed state at different token boundaries in practice; the Engine still treats any throw at this point as fatal.

Committing request bookkeeping:

- Appends the staged token to the host token mirror.
- Increments the current turn's generated-token count only when that output token was appended.
- Sets `processed_sequence_length_` to the sequence length that existed before sampling.
- Changes the status to `Active` or `TurnComplete`.

For a new request or queued continuation, this commit is the point where it moves from `Assigned`
to `Active` or `TurnComplete`. The dynamic transaction path does not need a separate visible
scheduling state between those states.

Finally, the engine swaps `staged_events_` into `pending_events_`. The first event is returned immediately, and later calls drain the rest without another model run.

## Constrained decoding and tool calling

Engine requests use the guidance configuration carried by their `GeneratorParams`. This allows
concurrent requests to use different JSON schemas, regular expressions, or Lark grammars. Tool
definitions and chat-template rendering remain application concerns; the Engine constrains the
generated token stream but does not parse tool calls from the decoded output.

Each guided request owns an independent constrained-logits processor. Its mask is applied before
minimum-length, repetition-penalty, and no-repeat-ngram processing, matching Generator ordering.
The processor snapshots its guidance and search configuration when the request is created, so later
mutation of the caller-owned `GeneratorParams` cannot change an active grammar. After sampling, the
selected token advances that request's grammar cursor.

The grammar cursor participates in the same transaction as search and paged-cache state. A step
checkpoints it before sampling, retains the advanced cursor on commit, and restores the checkpoint
on rollback. Draining an already-pending event does not advance the cursor again.

The model caches immutable tokenizer and compiled-grammar assets, while requests retain independent
copy-on-write cursors. After a dynamic step commits, the Engine submits one parallel llguidance job
for dirty cursors so CPU mask work overlaps the next model forward. Contiguous CUDA logits rows use
one host-to-device mask transfer and one mask kernel; unguided and partial-prefill rows receive
pass-through masks. If speculative mask submission fails, the committed step remains valid and the
dirty cursor retries mask construction when the next step needs it. Transient submission and
future-delivery failures can recover this way; an error reported by llguidance permanently poisons
that cursor and continues to fail rather than allowing generation with a stale mask.

Guidance fast-forward tokens are not currently supported by the Engine. Requests that enable them
are rejected because each forced token would also need a corresponding model execution and paged
KV-cache advancement inside the transaction.

## Decoded stop strings

`OgaTurnOptionsSetStopStrings` copies a Turn's decoded UTF-8 stop strings, validated and bounded by
`StopStringMatcher` (at most 16 entries, 16 KiB total, each entry nonempty valid UTF-8). An empty
array clears/disables stop strings; a null array is rejected. Matching is exact byte matching, with
no normalization, trimming, or case folding, and only considers text this Request generates during
the active turn -- prompt tokens, continuation input, and earlier-turn history never reach the
matcher.

When a turn enables stop strings, `Engine::BeginTurn` resolves the Engine-owned `Tokenizer`
(created once, on the first stop-enabled `BeginTurn`, via `Model::CreateTokenizer()`, and shared by
every Request) and builds this turn's complete, ready-to-use `StopStringController` before touching
the Request, its MTP shadow, or anything else -- so a tokenizer/stream construction failure leaves
everything completely unchanged for the caller to retry. That controller owns one `TokenizerStream`
and one `StopStringMatcher`; it decodes each generated token exactly once and feeds the resulting
bytes to the matcher. The `TokenizerStream` is freshly created for this turn's controller (see
`StopStringController`'s constructor and `RebuildStream()`), so its decoding context begins at the
first generated token of the active turn: the turn's prompt/continuation input tokens are never fed
to it, not even solely to seed detokenizer state before the first generated token is observed. This
is the exact generated-output decoding boundary a host must mirror if it runs its own incremental
decoder to hide stop text from published output -- see "Stop strings" in `docs/engine_c_api_spec.md`
for why a host that instead decodes its own copy of the whole prompt+generation as one continuous
stream can get different bytes for the first generated token (context-sensitive first-piece
spacing) than what the Engine's matcher actually saw. Ownership passes into
`RequestTurnAdmission::pending_stop_controller` (null
for a no-stop turn, which never touches the tokenizer or a controller at all -- the no-stop fast
path is unchanged from before this feature existed): `Request::PrepareTurnAdmission` and
`Request::SaveStateForNewTurnTransaction` never construct or otherwise touch the Request's stop
controller, only checkpoint bookkeeping for it. Only `Request::CommitTurnAdmission` installs the
prebuilt controller as the live `stop_controller_`, discarding whatever the Request had before
(a prior turn's controller, or null); `Request::RollbackTurnAdmission` (and simply an admission
attempt never reaching commit) instead just discards the not-yet-installed controller, so a failed
attempt -- whether it would have enabled stop strings or cleared them -- never touches the
Request's actual, already-committed stop_controller_ at all. This is what "modularity" means here:
Engine performs the fallible construction and infrastructure lookup, Request performs only the
plain, infallible move at the commit boundary.

`Request::StageGeneration` observes a newly generated token through the controller only when the
step actually appended one (see below for why an EOS token is not observed). Among the reasons this
observation can preempt, precedence is `StopString`, then `TurnLimit`, then `ContextLimit`: a token
that would otherwise end the turn on the turn or context limit still reports `StopString` when it
also completes a match. The raw token whose decoded bytes complete the match is retained, emitted as
a visible token, and counted toward generated usage exactly like any other token; bytes after the
match inside that token's decoded text are not trimmed from the Engine's raw history. The match, and
its caller-facing index into the turn's stop-string list, are not externally visible until the
step's transaction commits. A publication layer that hides stop text must run its own incremental
decoded-text matcher and hold back possible stop prefixes, then use the reported index to verify and
trim its published bytes; the Engine does not expose a byte offset within the match-completing
token.

The controller's own token history participates in the same transaction as Search, RNG, and
guidance state. Every `SaveState*ForTransaction()` variant checkpoints the controller's committed
token count; a direct `RestoreStateForTransaction()` or the queued
`QueueStateRestoreForTransaction()`/`CompleteStateRestoreForTransaction()` pair (the latter is where
the actual replay happens, matching how the queued path defers other non-trivial restoration) undoes
a speculatively observed token by recreating the `TokenizerStream` from scratch and replaying every
retained current-turn token through it, in order -- the ORT Extensions detokenizer cache backing a
stream cannot be cloned, so full replay is the only correctness-preserving way to undo a step. A
replay failure is treated exactly like any other rollback failure in this document: fatal, marking
the Engine unhealthy. `Request::CommitStep` copies the staged match's index into the Request only
after the transaction commits. `Request::MatchedStopStringIndex()` is otherwise always -1: a
committed StopString match makes the Request `TurnComplete` in the very same step that stages its
terminal event, and cancellation (`CanCancelFromEngine`) and fatal-failure handling
(`Engine::MarkUnhealthyAndThrow`) only ever force-terminate a Request that is still executable --
neither can run on a Request whose turn already ended in a committed match, so neither ever has an
existing match to clear; the field is simply never set on that path to begin with. The engine
invariant validator (`ValidateRequestInvariants`) enforces the resulting contract directly: a
Request's finish reason is `StopString` if and only if its matched index is nonnegative.

**Recovery cost bound.** A retryable whole-batch abort (see "Recoverable rollback" below) rolls back
every Request in the failing batch together, not just one. Every stop-enabled Request among them
that staged an observed token this step must independently recreate its `TokenizerStream` and
replay its full current-turn token history, so one retryable abort's recovery cost for the
stop-string component alone is `O(active stop-enabled requests in the batch x each request's
generated current-turn token count)` -- linear in the number of such requests, and, per request,
linear in how deep into the turn it already is. This follows from the required per-request replay;
this repository does not include a replay benchmark or publish a measured per-token constant. A
Request many thousands of tokens into a long turn pays a proportionally larger replay on every
retryable abort that touches it, and a batch with several such requests pays that cost once per
request, every time. No batching or amortization across requests exists for this replay today.
Rollout qualification for stop strings at scale should measure this cost with representative
tokenizers and turn depths, and monitor observed abort frequency alongside typical current-turn
depth rather than assuming replay cost is negligible.

Stop strings require an Engine configured for dynamic batching. Static batching completes generation
through `ScheduledRequests::GenerateNextTokens`, a non-transactional path that cannot stage, roll
back, or replay a match, so `Engine::BeginTurn` rejects a stop-enabled turn on a static-batching
Engine before any Request mutation.

Stop strings are incompatible with speculative draft verification for the specific Request whose
turn enables them, but this Engine capability is not disabled for other Requests. Every draft
producer -- manual `Request::SetDraftTokens` callers and the automatic in-Engine MTP drafter alike
-- converges on `Request::DraftTokenValidationError()`, which returns a rejection reason whenever
the request has an active stop controller. A manual caller sees this as a thrown error;
`Engine::PrepareMtpStep`'s existing per-request skip check (already used for guidance and other
per-request draft-ineligibility reasons) uses the same function to silently exclude the request from
receiving automatic drafts, so its target step keeps running the plain one-token-per-step path while
every other Request on the same Engine continues to use speculative decoding normally. See
"Speculative drafts" above for the shared verification path this reuses.

## Rollback and failure handling

The dynamic path separates failures into recoverable batch failures and fatal engine failures.

### Recoverable rollback

Request validation and post-processing failures are treated as retryable batch aborts. Model execution can also explicitly report a retryable failure. A recognized execution-memory capacity failure follows the same rollback path but has its own outcome.

Rollback performs both parts:

1. Restore request search state and sampler state from their checkpoints.
2. Release the composite reservation: discard any staged fixed outputs and provisional fixed slots (leaving resident slots untouched), and release every block held by the paged-cache reservation.

Step 1 restores every request in the failing batch, not just one; for a stop-enabled request that
staged an observed token this step, restoring its search state also replays its stop-string
controller (see "Decoded stop strings" above), so a batch containing several stop-enabled requests
deep into their turns pays that per-request replay cost once per request on every retryable abort --
see that section's "Recovery cost bound" for the resulting complexity and rollout-qualification
guidance.

After a successful rollback, committed request state and committed cache state match the state
before the step began. `Run()` translates `RetryableBatchAbort` into a `Retryable` event and
`ExecutionCapacityExceeded` into a `CapacityBlocked` event; the engine remains healthy. Calling
`Run()` again with unchanged memory availability and workload composition may produce the same
failure.

Planning allocation failures occur before reservation or request mutation. They propagate to the
caller without marking the Engine unhealthy, so a later `Run()` may retry. A
`StepPlanningConsistencyError`, by contrast, proves that committed paged and fixed ownership
disagree and is fatal.

In particular, rollback does not change the turn's generated-token count, lifecycle status,
pending-event queue or continuation state. A retry therefore observes
the same turn boundary and output stream as if the failed attempt had not run.

The model may have written data into reserved cache memory and into fixed output staging or an inactive direct bank before the failure. Discard is safe because neither was published as committed ownership or state: reserved blocks were never added to committed block tables, and the fixed active bank was never changed. Future users overwrite that storage before treating it as valid.

### Fatal failure

The engine becomes unhealthy when it cannot prove that all components still agree on state. Examples include:

- Reservation fails after an executable plan was produced.
- The reservation does not match the planned fixed-state resources (required flag, row count, new-slot count, staging bytes, or per-row request identity).
- Preparing the composite commit fails (for example a fixed staging copy fails while writing an inactive bank).
- An unknown model execution failure occurs.
- Rollback itself fails.
- Any part of the commit boundary fails.
- Planning detects inconsistent committed paged and fixed state.
- The scheduler returns an invalid planning outcome.
- Abandoned-Request cleanup detects inconsistent cache or scheduler ownership.

The engine stores the fatal error and marks every executable Turn complete with reason `Failed`.
Before rethrowing the stored error on later `Run()` calls, it emits one
`TurnFinished | Failed` event per affected Turn with the Turn's Request, ID, usage, and Engine
failure code. A fatal failure with no affected Turn emits one request-less Engine failure event.
At publication time, an externally abandoned Request is omitted from request-bearing events on a
best-effort basis; final-handle release may race with that snapshot, but any event already selected
retains the Request safely. Other executable Turns are still terminalized. `HasPendingRequests()`
returns true while these retained fatal events need draining. Request creation and `BeginTurn()`
also reclaim abandonment; an invariant failure from either operation throws the fatal
`EngineStepError` while retaining terminal events for the next positive-capacity `Run()`.
Continuing would risk using request search state and cache block tables from different logical
steps.

### Step outcomes

`step_plan.h` defines the main outcome categories:

| Outcome | Meaning |
| --- | --- |
| `NoWork` | No runnable work remains |
| `CapacityDeferred` | Pending work exists, but none of it fits the current free capacity |
| `UnserviceableRequest` | A request cannot fit the configured cache even at full availability |
| `Committed` | An executable plan was produced and is expected to commit |
| `RetryableBatchAbort` | The transaction was rolled back and may be retried |
| `ExecutionCapacityExceeded` | Execution exceeded available memory; the transaction was rolled back and the engine remains healthy |
| `ExecutionContractFailure` | Collaborators disagreed about planned or committed state |
| `FatalExecutionFailure` | Execution or rollback failed in a way that makes continued use unsafe |

When some requests fit and others are deferred, the step still executes the fitting subset. `capacity_deferred` is also recorded in transaction metrics for that successful planning pass.

If a continuation `BeginTurn()` fails while appending tokens and its Search checkpoint also cannot be restored, the request is closed and the Engine is marked fatally unhealthy. Reusing either would risk combining committed KV state with corrupted Search state.

## Paged KV-cache ownership

The paged cache has three useful ownership states:

### Free blocks

The block pool owns these blocks. They can be reserved by a future transaction.

### Transaction-reserved blocks

A `PagedCacheReservation` temporarily owns these blocks. They can appear in model-facing block tables for the current run, but they are not part of committed request ownership.

### Committed request blocks

A request's `PagedCacheBlockTable` owns these blocks. The table records:

- The request identity.
- The number of slots containing committed tokens.
- The ordered physical blocks used by the request.

Every physical block must be in exactly one of these states. The invariant helpers under `engine_invariants.*` validate total accounting, single ownership, valid block identifiers, reservation accounting, and consistency between request progress and cache progress. For a composite model, `ValidateCompositeStateInvariants()` additionally requires paged and fixed committed ownership to name the same requests and each request's fixed committed-token boundary to equal its paged committed-slot count.

Paged-cache publication exposes a validate/publish split used by the current composite transaction.
Paged-only execution can still use the `Commit()` convenience wrapper.
`ValidateCommit()` is repeatable and mutates nothing; it verifies request ownership, token
boundaries and scalar table/pool mutation generations, unique committed request tables, empty
reserved blocks, reserved-span accounting, touched-block mappings and occupancy, resident window
rings, and preallocated vector capacity. It also snapshots every committed resident table in order,
including requests omitted from the step, and requires its request identity, generation, block
mapping storage, committed-token boundary, and full/window mapping sizes to remain unchanged
through publication. Table replacement advances the destination generation even when vector
storage is reused. These
checks are independent of already-committed context blocks; their work is proportional to resident
request tables, scheduled requests, current growth, and the fixed window ring.
`CommitValidated()` allocation-free preflights every resident snapshot, delta, and captured growth
mapping again before publishing the already-validated block handles and advancing committed slots.
`Commit()` remains the single-reservation convenience wrapper that calls both phases.

A future composite transaction must validate every participating state reservation before
publishing any of them. Changing cache ownership, slot occupancy, block identity, or vector
headroom after validation invalidates the publish preconditions and is a fatal
transaction-contract error. A transaction must aggregate all paged-cache changes that share a
block pool into one `PagedCacheReservation`. This is a caller-enforced precondition: pool
generations reject overlapping reservations that allocate, free, or advance block occupancy.
`Block` identity and occupancy cannot be assigned through exposed handles; only the pool, cache,
and reservation mutation paths can advance occupancy, and each records the change in the pool
generation.

## Sliding-window paged layers

The runtime can store selected sliding-window layers in a fixed ring instead of growing their KV
cache through the full context. This path is enabled only when all of the following are true:

- `model.decoder.sliding_window` identifies the window size and layer indices.
- `model.decoder.inputs.block_table_windowed` is configured and is an input of the exported model.
- `search.chunk_size` is configured to a value greater than zero.
- At least one decoder layer keeps the full sequence.

Models without the windowed block-table input continue to use the normal full-sequence paged cache
for every layer. The layer indices must be within the decoder layer range and the window size must be
positive.

Let $C$ be the model chunk size, $W$ the sliding-window size, and $B$ the paged-cache block size. A
step can write $C$ new positions while its earliest query still reads $W - 1$ preceding positions,
so each request receives

$$
R = \left\lceil \frac{C + W - 1}{B} \right\rceil
$$

window blocks at admission. The window blocks come from a separate pool and remain a fixed cost for
the request. The window block table repeats those $R$ block IDs: column $j$ uses block
$j \bmod R$, so position $p$ resolves to slot $p \bmod (R B)$. Positions outside the live window
are overwritten in place.

The dynamic scheduler treats $C$ as a physical per-request query limit. A request-level
`chunk_size` that is absent, zero, or larger than $C$ is capped to $C$; a smaller positive override
is preserved. This prevents a transactional step from wrapping the ring while older positions are
still live.

Window blocks participate in the same reservation, rollback, commit, removal, and invariant checks
as full-context blocks, but accounting is validated independently for the two pools. CUDA graph
capture is supported: the runtime builds a persistent window block table with the same bucketed
shape as the full-context table.

## Fixed request-state pools

`FixedStatePool` is a model-independent owner for the manifest's `fixed_conv` and
`fixed_recurrent` groups. It validates the manifest
against session metadata and, for every fixed binding, additionally checks that
the input and output describe one per-request row of identical, statically known,
non-batch geometry with a batch axis (axis 0) that is either dynamic or a positive
fixed extent. Geometry validation recognizes fixed extents so it can diagnose asymmetric
bindings, but the continuous-batching pool requires a dynamic batch axis and rejects every
positive fixed extent before allocating device storage. It also rejects a zero batch,
two different fixed batch extents on input and output,
dynamic or mismatched non-batch axes, and dtype mismatches. Every dtype and
per-request row size is derived from that validated geometry, and state is
allocated with the model state-device allocator. The initial implementation
supports CPU and CUDA; other devices are rejected until they provide the required
offset-copy and completion guarantees.

Each resident request owns a stable slot identified by request identity and an
allocation generation. A released and reused slot receives a new generation, so
old handles are rejected. Each slot also tracks a `state_generation` (bumped on
every commit) and a `committed_tokens` count (the processed-token length baked
into its committed state). Admission is reservation-driven: the caller passes a
`request_id` and a `target_tokens` per scheduled row, and the pool infers
ownership. A request that already owns a committed slot is treated as resident and
keeps that slot; any other identity is admitted provisionally into a free slot
that is not discoverable as committed ownership until commit. There is no separate
allocation surface.

A reservation accepts requests in scheduled row order and exposes ordered handles,
bindings, and target tokens. A contiguous resident interval binds its active bank
directly as model input and its inactive bank as model output. If resident banks
differ, the pool first copies only minority rows into a canonical bank and changes
their selectors after synchronization; this representation-only normalization does
not advance request state. New admissions and fragmented intervals gather committed
or zero state into contiguous model inputs and use distinct staged outputs. No path
binds an output over visible committed state. The reservation reports its input/output
binding footprint through the retained `PlannedStagingBytes` API; direct views overlap
storage already counted by `PersistentBytes`.

Every tensor is backed by two persistent `[capacity, row...]` banks, and each slot
records which bank is currently active (holds its visible committed state). This
double buffering lets the commit be split into three phases so that a composite
Engine transaction can validate and stage all of its resources, synchronize once,
and then publish them at a single infallible boundary:

- **`ValidateCommit()`** is `const`, mutates nothing, and performs every host-side
  precondition check before device work begins. It proves every checkpointed slot
  is exactly where the reservation
  left it (ownership, identity, generation, and reservation id), that publishing it
  cannot overflow a generation, and that no row regresses below its slot's
  `committed_tokens`.
- **`PrepareCommit()`** is the fallible device-completion phase: it re-validates,
  copies fallback outputs into the **inactive** bank, or replays a partially accepted
  compact update over a direct output, and synchronizes. A fully accepted direct
  output needs no state copy. The active (visible) bank is never touched, so a
  failure leaves committed state exactly as it was; it drains the device and marks
  the pool unhealthy because the inactive banks may be left partially written. The
  reservation becomes `Prepared`.
- **`PublishCommit()`** is `noexcept` and performs no fallible or device work: it
  flips each slot to its freshly written bank, advances `state_generation`, sets
  `committed_tokens` to the row's `target_tokens`, and publishes provisional
  ownership. Because a prepared commit only wrote invisible banks, its output is
  invisible until this flip.

`Commit()` is a convenience wrapper that runs the three phases in order. `Discard()`
is valid from either `Reserved` or `Prepared`: it preserves every resident slot's
active state, generation, and committed tokens, and returns only provisional slots
to the free pool. A failed `PrepareCommit` also returns the reservation's
provisional slots to the pool and marks the pool unhealthy (it leaves state
consistent but cannot prove the inactive banks are whole), and `Discard()` on a
failed reservation is a no-op.

Only one reservation is live at a time. A reservation holds the pool's
single-reservation lock and its input/output binding backing for its whole
lifetime, and both are released when the reservation object is destroyed (an
uncommitted or prepared reservation also discards itself then). The retained
`ActiveStagingBytes` API tracks the live binding footprint, which overlaps persistent
banks for a direct reservation. The caller admits the next batch by letting
the previous reservation go out of scope after committing or discarding it; a
committed reservation's accessors stay valid until then because they read
reservation-owned storage.

Zeroing is minimized. Only the reusable per-tensor zero row is zeroed at
construction; it is the gather source for freshly admitted rows. A slot's active
bank is written by the commit that publishes it before it is ever gathered, and a
slot's inactive bank is only ever written (never read) before it becomes active, so
the persistent banks never need construction-time zeroing. The pool therefore does
not zero on release, and does not zero gather or staged tensors. It reports its
persistent bytes (both state banks and both staging buffers), reusable zeroing scratch bytes, and
the active reservation's input/output binding footprint separately. The two-bank design trades roughly double
the persistent footprint for a publish that is a host-only bank flip with no device
copy, which is what lets `PublishCommit()` be `noexcept`.

`PagedCacheManager` owns this pool (with `max_batch_size` slots) whenever the
decoder has `fixed_conv` or `fixed_recurrent` groups, and integrates it into the dynamic transaction:
`PlanStepResources()` plans fixed rows atomically with paged blocks, `ReserveStep()`
reserves both, `PrepareCommit()`/`Commit()` stage and publish both, and
`Deallocate()` validates and then releases both through a no-throw publication boundary. Explicit
removal and abandoned-request reclamation release a request's fixed slot together with its paged
blocks; a completed turn remains resident until one of those lifecycle events. `HybridDecoderIO` composes the packed
`VarlenDecoderIO` inputs and logits handling with the reservation's fixed input and
output bindings. Fixed-state models execute eagerly because packed position and
transaction-scoped state bindings are not yet part of the CUDA graph compatibility contract.

The pool allocates capacity-sized gather and output staging buffers at construction, before paged
KV auto-sizing queries available memory. Fallback reservations create exact-row tensor views over
that storage; eligible resident reservations instead view exact intervals of the persistent banks.
This keeps fallback staging inside the model's memory budget and prevents a planned step from
failing later because its fixed-state device buffers could not be allocated.

Fixed slots and paged block tables each maintain a preallocated flat request index. Ownership
publication updates these indexes without allocation, while planning resolves every resident and
selected row in O(1)-average lookup time. The composite request/paged/fixed consistency sweep is
therefore linear in resident and scheduled request counts rather than quadratic; this claim does
not cover unrelated queue ordering or bulk-removal compaction.

## Prefill, decode, and mixed batches

The engine uses the same transaction flow for prefill and decoding.

### Prefill

A prefill contributes as many pending prompt tokens as the remaining global
budget allows, bounded by `search.chunk_size` when configured. A prompt can
therefore span several committed steps even when `search.chunk_size` is absent.

Admitting a new prefill still reserves blocks for its whole prompt. Only the
executed contribution advances committed slots. This prevents another request
from taking the capacity needed to finish an admitted partial prefill.

### Decode

An active request normally contributes one token: the token sampled by its previous committed step.

### Mixed batch

The scheduler can place prefill and decode requests in the same model invocation if cache and batch capacity allow it. `VarlenDecoderIO` packs their different query lengths without padding.

Mixed batches are not eligible for the pure-decode CUDA graph path, but they remain valid eager executions.

## Batched sampling

Continuous batching requires each request to retain independent search behavior, but the GPU sampling work can still be batched.

The scheduler owns a reusable `BatchedSampler`. Each request owns its persistent `BatchedSamplerState`, so request-specific random streams survive changes in batch order and membership.

Sampler state is prepared as part of scheduler admission rather than being installed incrementally
on the Request. CUDA sampler-index acquisition is likewise failure-atomic: free-list capacity is
reserved before an index is published, initialization failure leaves the pool unchanged, and a
failure constructing the owning state wrapper releases the acquired index without allocation.

`ScheduledRequests` resolves each request's sampling configuration, groups compatible work inside the sampler, and submits the logits rows together. The implementation can handle heterogeneous request parameters without turning the requests into one shared `Search`.

If batched sampling or transactional sampler checkpoints are unsupported on the active device or search implementation, the code falls back to per-request sampling while preserving the same engine transaction boundary.

Logits processing remains per request. Minimum length, repetition penalty, no-repeat n-gram processing, EOS handling, maximum length, and sequence ownership continue to use each request's own state.

## CUDA graph capture

CUDA graph capture is an optimization for stable decode shapes.

A dynamic step is graph-capture eligible only when every selected request:

- Is past prefill.
- Contributes exactly one token.

The cache planner buckets block-table columns to powers of two, with a minimum bucket width of eight columns. The graph identity combines batch size with that block-table width bucket, because both values affect the captured input shape and launch configuration.

Graph buffers are allocated once at configured limits and reshaped as static views. Stable device addresses allow ONNX Runtime to replay the captured graph.

Prefill and mixed-token steps use graph id `-1`, which tells the CUDA execution provider to run eagerly.

An Engine-hosted DFlash 2 drafter consumes the packed auxiliary hidden-state tensor named by
`model.dflash2.main_aux_hidden_states`. Single-token target decode steps bind that output through
persistent graph buffers and remain eligible for CUDA graph capture; prefill and draft-verification
steps remain eager. Engine construction validates the output's rank, element type, and static width
against the drafter input before allocating cache resources.
The drafter run is synchronous because its packed inputs and outputs are owned by one proposal
call, so `model.dflash2.run_options` cannot disable execution-provider synchronization.
The direct drafter session also uses graph id `-1`: its variable-shaped proposal tensors are
allocated per call and therefore cannot be captured safely. If this optional post-commit drafter
run fails, the Engine discards any partial proposal and still publishes the already committed target
events. A transient failure is retried on the next step; three consecutive failures disable DFlash 2
for the Engine. `dflash2_failures` and `dflash2_disables` report those events.

The drafter keeps a fixed sliding-window ring per request from that request's first decode step
until it is closed, so its pool is sized for `max_batch_size` rings. A request that first decodes
while every ring is taken is skipped for the rest of its life and decodes without DFlash 2 drafts;
the drafter keeps serving the requests that already hold a ring. This makes `max_batch_size` the
DFlash 2 service-capacity limit across both active and idle long-lived requests, not merely the
per-step scheduler limit. `dflash2_admission_misses` reports requests denied a ring because that
capacity was occupied.

## Backpressure and fairness

Continuous batching does not mean every pending request runs on every step.

Backpressure can come from:

- Maximum batch size.
- Maximum scheduled tokens.
- Free paged-cache blocks.
- Per-request block growth.
- Graph-capture block-table width limits.

The current ordering policy is:

1. Consider decodes first, preserving their resident order.
2. Consider prefills afterward, preserving resident order and then scheduler-pool order.
3. Give every feasible selected row one token before expanding selected prefills in stable order.

Requests skipped because of token, row, or temporary cache capacity remain pending for a later step. Phase 1 does not rotate service or promote waiting prefills, so sustained decode demand can starve prefill work.

If no request can run because of temporary capacity, `RunDynamic()` reports `CapacityDeferred` instead of returning `nullptr`. Returning `nullptr` would incorrectly tell the caller that no work remains.

## Static engine path

The static engine path is intentionally separate.

`StaticBatchScheduler` builds a fixed batch, `StaticCacheManager` allocates one cache configuration for that batch, and `StaticBatchDecoderIO` prepares traditional batched inputs. Individual requests cannot release cache resources until the static batch is complete.

`RunStatic()` performs decode and sampling directly without the dynamic transaction and reservation protocol.

A resident static request queued by `BeginTurn()` returns to `Active` without reallocating the
batch. Static cache rows still cannot be released independently, and an all-turn-complete batch may
be recycled for new work. Static continuation is therefore valid only while the original
single-request batch remains resident. The per-turn generated-token budget applies on this path
too, although static execution does not use the dynamic reservation/checkpoint transaction.

Close or abandonment logically removes a Request from scheduling and purges its undelivered events.
A closed Request that is already resident in a static batch nevertheless remains part of that
shared physical allocation until the batch is recycled. It is not sampled or returned again, but
its row and cache allocation can remain alive for the lifetime of the batch. The Request's Search,
parameters, host token mirror, and length metadata remain available only to keep that physical row
safe; sampling-only state is released at logical close. When the whole batch is recycled, the Engine
releases the retained state on its owner thread and the closed Request becomes a lightweight
tombstone.

Changes to shared types such as `Request`, `ScheduledRequests`, `ModelExecutor`, or `SimpleDecoder` should be checked against both paths. This document should be updated only where behavior is shared or where the dynamic path changes.

## Experimental low-level public API shape

The language bindings currently expose the same basic low-level loop. Production hosts are expected to wrap this surface or use its replacement rather than expose it as their stable API:

```python
request_options = og.RequestOptions()
request_options.set_max_session_tokens(4096)
request = engine.create_request(generation_params, request_options)
turn_options = og.TurnOptions(request)
turn_options.set_max_generated_tokens(128)
turn_id = request.begin_turn(initial_tokens, turn_options)

event_buffer = engine.create_event_buffer(8)
while engine.has_pending_requests():
    for event in engine.run(event_buffer):
        if event.flags & og.EngineEventFlags.FAILED:
            if (
                event.request is None
                or event.error_code != og.EngineErrorCode.REQUEST_UNSERVICEABLE
            ):
                raise RuntimeError(
                    f"Engine failure: flags={event.flags}, error_code={event.error_code}"
                )
            # This Request cannot make progress, but the Engine and other Requests remain healthy.
            event.request.close()
            continue
        if event.request is None:
            if event.flags & (
                og.EngineEventFlags.CAPACITY_BLOCKED
                | og.EngineEventFlags.RETRYABLE
            ):
                # No progress committed. The Engine remains healthy and pending work can retry.
                continue
            raise RuntimeError(
                f"Engine-level event: flags={event.flags}, error_code={event.error_code}"
            )
        if event.flags & og.EngineEventFlags.TOKEN:
            # event.request is a borrowed identity alias; event.turn_id is Request-local.
            token = event.token
            # Stream or process the token.
        if event.flags & og.EngineEventFlags.TURN_FINISHED:
            reason = event.finish_reason

turn_options.set_max_generated_tokens(64)
turn_id = request.begin_turn(next_turn_tokens, turn_options)

# Repeat engine.run(event_buffer), then close when continuation is no longer needed.
request.close()
```

An event buffer with capacity one provides capacity-one behavior; larger buffers return the
complete transaction output when it fits. A run returning no events can represent committed
partial-prefill progress, so callers
continue while `has_pending_requests()` remains true. Flags are a bitmask, so callers test the
`TOKEN` bit rather than comparing flags for equality; `TOKEN | TURN_FINISHED` can be combined, and
`event.token` is consumed only when `TOKEN` is set. `event.request` is a borrowed alias of the
caller-owned Request handle, while `event.turn_id` identifies the Request-local Turn. A
turn-complete dynamic request remains cache-resident until explicit close, which releases dynamic
cache ownership immediately.

## Keeping this document current

When changing the engine, update the relevant sections of this document in the same pull request.

The document needs review when a change affects any of the following:

- Request states or sequence counters.
- Admission order, fairness, preemption, or prefill chunking.
- Batch planning or packed token layout.
- Cache block accounting, reservation, commit, release, or eviction.
- Model inputs required by paged attention.
- Sampling, logits processing, or random-state ownership.
- Transaction checkpoints, rollback, failure classification, or commit ordering.
- Ready-result behavior or public engine API semantics.
- CUDA graph eligibility or graph shape bucketing.

Prefer describing current behavior directly. If a design is proposed but not implemented, label it clearly as future work or keep it in a separate design document. Remove or revise statements that stop matching the code.

Tests under `test/cpp/engine/` provide focused coverage for scheduler planning, paged-cache resources, request and cache invariants, transaction rollback, fatal failures, and Engine-event draining. When behavior changes, update both the tests and this document so they continue to describe the same contract.
