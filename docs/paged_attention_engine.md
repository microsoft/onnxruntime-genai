# Paged Attention Engine

This document describes how the paged attention engine and continuous batching work in the current codebase. It is intended to be the living architecture guide for this part of ONNX Runtime GenAI.

Update this document whenever a change affects request admission, scheduling, paged KV-cache ownership, model input packing, sampling, transaction behavior, request completion, or failure handling. Historical design and performance work can remain in separate documents, but this document should always describe the behavior that exists on the current branch.

## Scope

The `Engine` can use either static batching or dynamic batching. This document focuses on the dynamic path used by models configured with `engine.dynamic_batching`, because that path provides continuous batching and uses the paged KV cache.

The current dynamic path manages paged KV decoder state together with per-request search and sampler state. A standalone `FixedStatePool` exists for `fixed_conv` and `fixed_recurrent` manifest groups, but the dynamic Engine still rejects those groups and does not construct, bind, or commit the pool. Future execution support must be selected from model capabilities rather than model names, and every Engine-owned mutable state must participate in the same transaction boundary.

> **Experimental low-level API:** An Engine creates model-bound Requests. `BeginTurn()` queues both
> initial input and later continuation input and snapshots an opaque `TurnParams` object. Request
> creation can also snapshot a prompt-plus-generated session-token limit. Successful Turns receive
> zero-based, monotonically increasing request-local IDs. `Run()` performs synchronous progress and
> returns one typed `EngineEvent`; token and completion payloads are selected by event flags.
> `CancelTurn(turn_id)` stops only the named Turn, and `Close()` releases Engine resources.
> `OgaCreateEngine` retains shared ownership of the underlying Model, so the caller may release its
> `OgaModel` handle after successful Engine creation. The Model remains alive until Engine teardown
> and the release of any other retaining objects.
>
> **Single-owner requirement:** One host-owned thread must perform all Engine and Request operations,
> including request creation, `BeginTurn()`, `Run()`, `CancelTurn()`,
> `Close()`, and handle destruction. Other host threads marshal commands and copied inputs to that owner. The
> Engine enforces its owner thread and has no worker thread. Final
> handle abandonment publishes an atomic marker as a safety net; cleanup runs at the next serialized
> Engine boundary.

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
drain one pending Engine event, if one exists
    |
plan a runnable batch
    |
reserve all KV-cache growth needed by that plan
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
return committed events one at a time
```

The step is transactional. Planning and reservation do not immediately change committed request or cache state. If a recoverable failure occurs before commit, the engine restores the request search state and releases the reserved cache blocks. A failure during the commit boundary is considered fatal because the engine can no longer guarantee that all cooperating components agree on the committed state.

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
- `Engine::Run()` calls `RunDynamic()`.

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

Fixed groups remain unsupported by the dynamic Engine: `ValidateDynamicEngineCompatibility` rejects `fixed_conv` and `fixed_recurrent` until request-owned fixed-state transactions are integrated. A standalone `FixedStatePool` (below) can already validate, allocate, and transactionally stage those groups, but the Engine does not construct it or bind it to model execution, so they still only describe graph state.

## Request lifecycle

A `Request` is one sequence. Engine requests currently require:

- `search.batch_size == 1`
- `search.num_beams == 1`
- At least one input token in every `BeginTurn()`

The engine creates throughput by batching several independent requests, not by placing several sequence rows inside one request.

The important request states are:

```text
Unassigned (Created) -- BeginTurn(tokens, turn_params) --> Assigned (Queued) -- schedule --> Active
                              ^                                           |
                              |                                           | turn stops
                              +----- BeginTurn(tokens, turn_params) -- TurnComplete

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

### `Assigned`

This is the queued state. A successful first or later `BeginTurn()` moves a Request here while its
copied input waits for execution. `BeginTurn()` is rejected while already queued or active. Input must leave room for at least one generated token below the request's internal
`max_total_tokens`.

Public `max_session_tokens` (stored internally as `max_total_tokens`) is the cumulative total
sequence limit for the entire Request: the initial
prompt, generated output, and every continuation input all count against the same limit. It defaults
to `max_length`, cannot exceed `max_length`, and is not reset by `BeginTurn()`.

Opaque nullable `TurnParams` are separate. Null, or setting `max_generated_tokens` to zero, means
that no per-Turn limit applies beyond the cumulative Request limit. The current Turn counts only
generated tokens appended as output. Initial
input, continuation input, and the previously generated token replayed during continuation prefill
do not count. A later turn may begin whenever its input still leaves room for at least one generated
token under the cumulative limit, and it may choose a different per-Turn limit.

`OgaRequestOptions` is a public sized Version 1 structure, not an opaque handle. Callers
zero-initialize it, set `struct_size`, and leave `reserved` zero. Null options and zero
`max_session_tokens` use the Request's snapshotted `OgaGeneratorParams.search.max_length`. That
search value normally defaults from the model context length, but a caller may set it lower before
Request creation. Undersized structures fail before mutation; larger structures are accepted and
trailing bytes are ignored. `OgaTurnParams` is opaque and reusable; `BeginTurn()` snapshots
supported values.

### `Active`

The current turn is executable and owned by the Engine. It normally has one unprocessed token at
the beginning of a decode step: the token sampled by the previous step.

### `TurnComplete`

The current Turn reached an end condition. A `TurnFinished` event reports `Eos`, `StopSequence`,
`MaxGeneratedTokens`, `MaxSessionTokens`, `Cancelled`, or `Failed`, plus per-Turn usage.
`StopSequence` is reserved until stop sequences are implemented. A generated EOS token is not
appended to the logical sequence or emitted as a token event.
The next continuation fragment is therefore responsible for any turn-boundary tokens required by
the model's chat template.

`BeginTurn(tokens, turn_params)` appends the next input fragment and moves a resident Request back
to `Assigned`. The completed Turn's terminal event must first be drained; calling `BeginTurn()`
while any event for the Request is pending fails without mutation. Turn-scoped
generated count and limit reset only after all validation, input allocation, Search append, and
scheduler preparation succeeds.

Cancellation does not rewind the Search sequence or committed cache. Accepted continuation input
and generated output therefore remain part of the logical request. A canceled resident request can
begin a later turn after its terminal notification is drained. A first turn canceled before
admission has no cache state, but it can likewise begin a later turn: the retained initial input and
new continuation input are prefetched together. If retained model state has otherwise ceased to be
resident, continuation still fails. Callers that need to discard canceled input or release capacity
must `Close()` and create a new Request.

Every completed Turn publishes exactly one terminal event. A final visible token and completion may
be combined in one event. An unserviceable Request receives `TurnFinished | Failed`; fatal Engine
failure returns a request-null `Failed` event and makes later progress calls rethrow the stored
failure.

A turn-complete request remains Engine-owned. Once admitted, it normally remains resident; a turn
canceled before initial admission has not allocated cache state. `Close()` ends logical ownership
immediately and releases dynamic cache resources; static batch storage may remain until batch
recycling. If every external handle is released instead, the request is marked abandoned and
reclaimed at the next serialized Engine boundary, including `HasPendingRequests()`.

Planning skips turn-complete residents and does not release their cache. Retained requests still
consume paged-cache blocks and a batch slot, so applications must call `Close()` when they no
longer need continuation when deterministic immediate reclamation is required.

### `Close()`

`Close()` is legal from every state, is sequentially idempotent, and moves the request to terminal
`Closed`. On the dynamic path, close immediately erases scheduler membership and releases
committed paged-cache ownership. The Engine also removes any undrained pending events for that
request. Final-handle abandonment performs the same logical removal and event purge when the Engine
next reaches an owner-thread boundary.

Events already copied by the host remain host-owned. Destroying the Engine terminalizes every bound
Request; surviving external handles are closed tombstones and still must be destroyed.

### `Closed`

`Closed` is distinct from `Unassigned` because close may already have destroyed residency and
scheduler ownership. Returning to `Unassigned` would imply that the same logical sequence could be
submitted as a new request. A closed static-batch row may remain physically allocated until the
batch is recycled, but it is no longer sampled or returned.

## The request length counters

Request progress uses these values:

| Value | Meaning |
| --- | --- |
| `CurrentSequenceLength()` | Number of tokens currently held by the request's search sequence |
| `processed_sequence_length_` | Number of sequence tokens already represented in the committed KV cache |
| `turn_prompt_tokens_` | Input tokens accepted by the current Turn |
| `turn_generated_tokens_` | Generated output tokens committed during the current turn; reset only by a successful `BeginTurn()` |
| `current_turn_id_` | Request-local `uint64_t` ID assigned after successful Turn admission; starts at 0 and increases monotonically |
| `finish_reason_` | Strongly typed reason for current-turn completion; `None` while generation is in progress |

Turn IDs use the complete `uint64_t` range. Once `UINT64_MAX` has been assigned, a later
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

The low-level engine API advances through repeated calls to synchronous `Run()`.

Before executing new work, `Run()` drains `pending_events_`. One model invocation may produce an
event for several Requests, but `Run()` returns one event at a time. Remaining events are returned
by later calls without another model execution. Each event retains its Request internally until
delivery; the public pointer is borrowed and is valid only while the caller retains its owned
Request handle.

A committed step emits a token event, a terminal event, or one combined
`Token | TurnFinished` event. EOS can emit terminal completion without a visible token. Event
delivery does not mutate the Request's retained logical sequence.

This distinction is important:

- One call to `Run()` does not always mean one model invocation.
- Draining a previously committed batch does not change model or cache state.
- `Request::Close()` purges undrained events for that Request.
- `HasPendingRequests()` first reclaims abandoned Requests, then is true while either the ready
  queue or scheduler contains work.

If the engine has previously encountered a fatal transaction or execution failure, `Run()` rethrows
the stored error instead of attempting more work.

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

`PagedCacheManager::ReserveStep()` creates a `PagedCacheReservation`.

The reservation:

- Reserves every new physical block required by the selected plan.
- Records how far each request's committed slot count would advance.
- Creates provisional block tables for newly admitted requests.
- Leaves the committed block tables and allocated-request list unchanged.

Reservation is all-or-nothing. If all planned blocks cannot be reserved, the engine treats that as an execution contract failure because planning had already declared the step executable.

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

### 8. Pack variable-length model inputs

`SimpleDecoder` chooses `VarlenDecoderIO` for the dynamic path.

`VarlenDecoderIO` builds three coordinated inputs:

```text
input_ids                    all requests' unprocessed tokens concatenated
cumulative_sequence_lengths  boundaries of each request in the flat token array
past_sequence_lengths        committed KV length and write position for each request
```

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

The result records:

- The sampled token, if a token was appended.
- Whether a token was appended.
- Whether the turn is complete.

These results are staged. The request's host token mirror, processed-length counter, per-turn
generated count, and internal lifecycle state are not updated yet.

Requests that appended a token or became complete produce staged events. Events are not exposed until the complete transaction commits.

### 11. Commit the transaction

The commit order in `Engine::RunDynamic()` is deliberate:

1. Commit the request search checkpoints and sampler checkpoint.
2. Commit the paged-cache reservation.
3. Commit each request's lightweight bookkeeping.
4. Publish the staged events.

Committing the cache reservation:

- Adds reserved blocks to existing block tables.
- Adds provisional block tables for newly admitted requests.
- Advances committed cache-slot counts.
- Adds newly admitted requests to the cache manager's allocated-request list.

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

Guidance fast-forward tokens are not currently supported by the Engine. Requests that enable them
are rejected because each forced token would also need a corresponding model execution and paged
KV-cache advancement inside the transaction.

## Rollback and failure handling

The dynamic path separates failures into recoverable batch failures and fatal engine failures.

### Recoverable rollback

Request validation and post-processing failures are treated as retryable batch aborts. Model execution can also explicitly report a retryable failure. A recognized execution-memory capacity failure follows the same rollback path but has its own outcome.

Rollback performs both parts:

1. Restore request search state and sampler state from their checkpoints.
2. Release every block held by the paged-cache reservation.

After a successful rollback, committed request state and committed cache state match the state before the step began. The caller receives an `EngineStepError` with either `RetryableBatchAbort` or `ExecutionCapacityExceeded`, and the engine remains healthy. Calling `Run()` again with unchanged memory availability and workload composition may produce the same capacity failure.

In particular, rollback does not change the turn's generated-token count, lifecycle status,
pending-event queue or continuation state. A retry therefore observes
the same turn boundary and output stream as if the failed attempt had not run.

The model may have written data into reserved cache memory before the failure. Releasing the reservation is still safe because those blocks were never added to committed block tables. Future users of those blocks overwrite the relevant slots before treating them as valid cache contents.

### Fatal failure

The engine becomes unhealthy when it cannot prove that all components still agree on state. Examples include:

- Reservation fails after an executable plan was produced.
- An unknown model execution failure occurs.
- Rollback itself fails.
- Any part of the commit boundary fails.
- The scheduler returns an invalid planning outcome.

The engine stores the fatal error and rethrows it on later `Run()` calls. Continuing would risk using request search state and cache block tables from different logical steps.

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

Every physical block must be in exactly one of these states. The invariant helpers under `engine_invariants.*` validate total accounting, single ownership, valid block identifiers, reservation accounting, and consistency between request progress and cache progress.

Paged-cache publication exposes a validate/publish split for the composite transaction introduced
by the next Engine integration step. The current Engine continues to use the `Commit()` convenience
wrapper.
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
fixed extent. A fixed batch on either the input or the output constrains the whole
binding: it is adopted as the pool's fixed batch size and every reservation must
match it, and the pool refuses to construct if that batch cannot fit in capacity.
It rejects a zero batch, two different fixed batch extents on input and output,
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
bindings, and target tokens. It gathers each resident row from the slot's active
persistent bank and each freshly admitted row from a single reusable zeroed row,
into contiguous model inputs, and allocates distinct staged output tensors. It
never binds an output over committed state. All host validation and slot planning
complete before any device copy is enqueued; once gather copies are in flight the
pool synchronizes before returning, and a failure drains the device and marks the
pool unhealthy without leaving partially mutated slots. The reservation reports its
`PlannedStagingBytes` (gather plus staged output) up front.

Every tensor is backed by two persistent `[capacity, row...]` banks, and each slot
records which bank is currently active (holds its visible committed state). This
double buffering lets the commit be split into three phases so that a composite
Engine transaction can validate and stage all of its resources, synchronize once,
and then publish them at a single infallible boundary:

- **`ValidateCommit()`** is the fallible host-side preflight. It is `const`, mutates
  nothing, and proves every checkpointed slot is exactly where the reservation
  left it (ownership, identity, generation, and reservation id), that publishing it
  cannot overflow a generation, and that no row regresses below its slot's
  `committed_tokens`.
- **`PrepareCommit()`** is the fallible device-staging phase: it re-validates, then
  copies each staged output row into the slot's **inactive** bank and synchronizes.
  The active (visible) bank is never touched, so a failure leaves committed state
  exactly as it was; it drains the device and marks the pool unhealthy because the
  inactive banks may be left partially written. The reservation becomes `Prepared`.
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
single-reservation lock and its gather/output staging memory for its whole
lifetime, and both are released when the reservation object is destroyed (an
uncommitted or prepared reservation also discards itself then). `ActiveStagingBytes`
therefore tracks the live reservation. The caller admits the next batch by letting
the previous reservation go out of scope after committing or discarding it; a
committed reservation's accessors stay valid until then because they read
reservation-owned storage.

Zeroing is minimized. Only the reusable per-tensor zero row is zeroed at
construction; it is the gather source for freshly admitted rows. A slot's active
bank is written by the commit that publishes it before it is ever gathered, and a
slot's inactive bank is only ever written (never read) before it becomes active, so
the persistent banks never need construction-time zeroing. The pool therefore does
not zero on release, and does not zero gather or staged tensors. It reports its
persistent bank bytes (both banks), reusable zeroing scratch bytes, and active
gather/output staging bytes separately. The two-bank design trades roughly double
the persistent footprint for a publish that is a host-only bank flip with no device
copy, which is what lets `PublishCommit()` be `noexcept`.

This pool is not yet part of the Engine transaction, scheduler capacity model, or
decoder bindings, and the dynamic Engine still rejects fixed-state groups. Those
integrations must commit paged KV, fixed state, request/search state, and ready
events at one boundary before hybrid execution is enabled.

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
its row, Request/Search storage, and cache allocation can remain alive for the lifetime of the
batch. Static cleanup therefore must not be described or tested as immediate per-Request
deallocation.

Changes to shared types such as `Request`, `ScheduledRequests`, `ModelExecutor`, or `SimpleDecoder` should be checked against both paths. This document should be updated only where behavior is shared or where the dynamic path changes.

## Experimental low-level public API shape

The language bindings currently expose the same basic low-level loop. Production hosts are expected to wrap this surface or use its replacement rather than expose it as their stable API:

```python
request = engine.create_request(generation_params, max_session_tokens=4096)
turn_params = og.TurnParams(request)
turn_params.set_max_generated_tokens(128)
turn_id = request.begin_turn(initial_tokens, turn_params)

while engine.has_pending_requests():
    event = engine.run()
    if event.flags & og.EngineEventFlags.TOKEN:
        # event.request is a borrowed identity alias; event.turn_id is Request-local.
        token = event.token
        # Stream or process the token.
    if event.flags & og.EngineEventFlags.TURN_FINISHED:
        reason = event.finish_reason

turn_params.set_max_generated_tokens(64)
turn_id = request.begin_turn(next_turn_tokens, turn_params)

# Repeat engine.run(), then close when continuation is no longer needed.
request.close()
```

One event is returned at a time. Flags are a bitmask, so callers test the `TOKEN` bit rather than
comparing flags for equality; `TOKEN | TURN_FINISHED` can be combined, and `event.token` is consumed
only when `TOKEN` is set. `event.request` is a borrowed alias of the caller-owned Request handle,
while `event.turn_id` identifies the Request-local Turn. A turn-complete dynamic request remains
cache-resident until explicit close, which releases dynamic cache ownership immediately.

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
