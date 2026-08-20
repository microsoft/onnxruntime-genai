# Paged Attention Engine

This document describes how the paged attention engine and continuous batching work in the current codebase. It is intended to be the living architecture guide for this part of ONNX Runtime GenAI.

Update this document whenever a change affects request admission, scheduling, paged KV-cache ownership, model input packing, sampling, transaction behavior, request completion, or failure handling. Historical design and performance work can remain in separate documents, but this document should always describe the behavior that exists on the current branch.

## Scope

The `Engine` can use either static batching or dynamic batching. This document focuses on the dynamic path used by models configured with `engine.dynamic_batching`, because that path provides continuous batching and uses the paged KV cache.

The current dynamic path manages paged KV decoder state together with per-request search and sampler state. It does not bind or transactionally checkpoint hybrid recurrent, convolutional, or other mutable model state. Future support for such state must be selected from model capabilities rather than model names, and every Engine-owned mutable state must participate in the same transaction boundary.

> **Transitional low-level API:** `AddTokens()` plus `AddRequest()`, `Continue()`, repeated `Step()` calls, token-at-a-time unseen-output access, and `Remove()` are a transitional host-facing surface. The production host API is expected to wrap or replace these operations; do not treat their current shape as the final high-level contract.
>
> **Serialization requirement:** Except for releasing an external request handle, every call on an `Engine` and on any `Request` owned by that engine must be externally serialized with `Engine::Step()`. This includes completion and unseen-output access as well as lifecycle mutation. Final handle release only publishes an atomic abandonment marker; cleanup runs at the next serialized Engine boundary. The API is otherwise not thread-safe, and idempotent terminal removal only makes sequential retries harmless.

The main implementation is under `src/engine/`:

| Responsibility | Main files |
| --- | --- |
| Top-level orchestration and transaction handling | `engine.h`, `engine.cpp` |
| Request lifecycle and per-request search state | `request.h`, `request.cpp` |
| Batch planning and request admission | `scheduler.h`, `scheduler.cpp`, `decode_first_scheduler_policy.*` |
| KV-cache capacity planning and ownership | `cache_manager.h`, `cache_manager.cpp`, `paged_key_value_cache.h`, `paged_key_value_cache.cpp` |
| Speculative cache reservation | `paged_cache_reservation.h`, `paged_cache_reservation.cpp` |
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
return an already-ready request, if one exists
    |
plan a runnable batch
    |
reserve all KV-cache growth needed by that plan
    |
preflight generated-output bookkeeping
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
return ready requests one at a time
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
- `Engine::Step()` calls `StepDynamic()`.

Dynamic batching limits scheduled rows with `max_batch_size` and limits the total
query tokens in one model run with `max_scheduled_tokens`. The token limit defaults
to 2048. Both limits are positive and independent.

Without dynamic batching, the engine uses the older static batching path. Static batching allocates and advances a batch as a unit. It does not use the transaction flow described below.

## Request lifecycle

A `Request` is one sequence. Engine requests currently require:

- `search.batch_size == 1`
- `search.num_beams == 1`
- At least one input token before the request is added to the engine

The engine creates throughput by batching several independent requests, not by placing several sequence rows inside one request.

The important request states are:

```text
Unassigned (Created) -- submit --> Assigned (Queued) -- schedule --> Active
                                      ^                              |
                                      |                              | turn stops
                                      +---- Continue(tokens) ---- TurnComplete

Assigned (Queued) ---+
Active --------------+-- Remove() --> Closed
TurnComplete --------+
```

### `Unassigned`

The request is not owned by an engine. `AddTokens()` accumulates the initial prompt in this state.
`Continue()` is not valid until a submitted request reaches `TurnComplete`.

### `Assigned`

`Engine::AddRequest()` validates the request, calls `Request::Assign()`, and adds it to the scheduler pool.

This is the queued state. `Engine::AddRequest()` moves a new request here before first admission.
`Continue()` also moves a cache-resident `TurnComplete` request here while its next input waits for
execution.

For a new request, assignment moves the prompt into `Search`, creates the host-side token mirror,
initializes the sequence counters, and records the owning Engine. `AddTokens()` and `Continue()` are
both rejected while already queued. Input must leave room for at least one generated token below
`max_length`.

`max_length` is the cumulative total sequence limit for the entire session: the initial prompt, generated output, and every continuation input all count against the same limit. `Continue()` does not reset it, and it is not a per-turn generation budget.

### `Active`

The current turn is executable and owned by the Engine. It normally has one unprocessed token at
the beginning of a decode step: the token sampled by the previous step.

### `TurnComplete`

The current generation turn reached an end condition, such as EOS or maximum length. Generated
output remains available. `IsTurnComplete()` is the API for testing this state; it does not mean permanent request termination or removal.

A generated EOS/stop token is not appended to the logical sequence or returned as unseen output.
The next continuation fragment is therefore responsible for any turn-boundary tokens required by
the model's chat template.

`Continue(tokens)` appends the next input fragment and moves a resident request back to `Assigned`.
`AddTokens()` remains an initial-input-only operation.

A submitted request remains owned by its Engine and resident at `TurnComplete`. `Remove()` releases that ownership immediately. If every external handle is released instead, the request is marked abandoned and reclaimed before the Engine's next `AddRequest()` or `Step()` boundary.

Planning skips turn-complete residents and does not release their cache. Retained requests still
consume paged-cache blocks and a batch slot, so applications must call `Remove()` when they no
longer need continuation when deterministic immediate reclamation is required.

### `Remove()`

`Remove()` is legal from `Assigned`, `Active`, and `TurnComplete`, and moves the request to
terminal `Closed`. On the dynamic path, removal immediately erases scheduler membership and releases
committed paged-cache ownership. The Engine also removes any undrained ready-queue entries for that
request.

Calling `Remove()` for an already terminal `Closed` request is an engine-agnostic idempotent no-op because the request no longer has an owner. Removing an `Unassigned` request remains invalid, as does asking an Engine other than the request's owner to remove a nonterminal request. This idempotence does not relax the external serialization requirement.

### `Closed`

`Closed` is distinct from `Unassigned` because removal may already have destroyed residency and
scheduler ownership. Returning to `Unassigned` would imply that the same logical sequence could be
submitted as a new request. A closed static-batch row may remain physically allocated until the
batch is recycled, but it is no longer sampled or returned.

## The request length counters

Three views of request progress are important:

| Value | Meaning |
| --- | --- |
| `CurrentSequenceLength()` | Number of tokens currently held by the request's search sequence |
| `processed_sequence_length_` | Number of sequence tokens already represented in the committed KV cache |
| `seen_sequence_length_` | High-water sequence index of generated output consumed by the API caller; copied into invariant snapshots rather than used to select the next output token |

Generated-output delivery uses separate bookkeeping because continuation input creates gaps in the
logical sequence:

| Value | Meaning |
| --- | --- |
| `tokens_host_` | Host-side mirror of the complete logical sequence, including prompt, generated output, and continuation input |
| `unseen_token_indices_` | Positions of generated tokens in `tokens_host_`; continuation-input positions are never added |
| `next_unseen_token_index_` | Cursor into `unseen_token_indices_`; entries at and after this cursor have not been consumed |

Unread generated output is one globally ordered stream for the request across all turns. The unseen-output API does not tag tokens with a turn, so callers that need per-turn attribution must track the boundaries themselves.

The generated-token index queue is not reserved to `max_length`. Before a scheduled request can
execute, it compacts a consumed prefix when that avoids growth or when the consumed prefix is at
least as large as the unread suffix. It then reserves geometrically from the actual unread count
plus the maximum indices the step can append. An Engine request has one sequence, so a
chunk-complete step reserves one append and a partial-prefill step reserves none.

This preparation runs in both static and dynamic `ScheduledRequests` construction. The static
scheduler also prepares newly admitted rows before publishing its immediate cache allocation and
prepares queued residents before moving them to `Active`; construction then finds the required
capacity already available. Preparation therefore finishes before model execution or search-state
advancement and before any request or cache state commits. `Request::CommitStep()` can consequently
remain allocation-free and `noexcept`; static `CompleteGeneration()` uses the same prepared
capacity while retaining its existing append behavior.

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

## `Engine::Step()` and ready-result draining

The current transitional low-level engine API advances through repeated calls to `Step()`.

Before executing new work, `Step()` checks `ready_requests_`. One model invocation may produce a token for several requests, but `Step()` returns only one `Request` pointer. The remaining ready requests stay in the ready queue and are returned by later `Step()` calls without another model execution.

This distinction is important:

- One call to `Step()` does not always mean one model invocation.
- Draining a previously committed batch does not change model or cache state.
- `Engine::RemoveRequest()` purges undrained entries for that request from the ready queue.
- `HasPendingRequests()` is true while either the ready queue or scheduler contains work.

If the engine has previously encountered a fatal transaction or execution failure, `Step()` rethrows the stored error instead of attempting more work.

## One dynamic step in detail

`Engine::StepDynamic()` coordinates the complete transaction.

### 1. Identify active and waiting requests

`DynamicBatchScheduler::PlanStep()` skips `TurnComplete` residents and builds candidates from
executable residents plus waiting requests.

The cache manager checks whether those candidates fit alongside dormant turn-complete requests. If
retained residency prevents admission or cache growth, the plan reports capacity backpressure; the
application decides which conversation to release with `Remove()`.

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

The decoder also prepares attention metadata with bounds for the maximum query length and maximum KV length in the step. These bounds help the paged attention operator choose its backend and size its work without reading sequence lengths back from the device.

### 9. Run the decoder once

`SimpleDecoder::Decode()` calls the ONNX Runtime decoder session once for the complete scheduled batch.

The outputs include one logits row for every flattened input token. The decoder state is attached to `ScheduledRequests` so post-processing can select the rows that belong to each request.

For eligible pure decode shapes, the decoder may capture or replay a CUDA graph. Prefill and other variable shapes run eagerly.

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
- Whether generation is complete.

These results are staged. The request's host token mirror, processed-length counter, and internal lifecycle state are not updated yet.

Requests that appended a token or became complete are placed in a staged ready list. The list is not exposed until the complete transaction commits.

### 11. Commit the transaction

The commit order in `Engine::StepDynamic()` is deliberate:

1. Commit the request search checkpoints and sampler checkpoint.
2. Commit the paged-cache reservation.
3. Commit each request's lightweight bookkeeping.
4. Publish the staged ready list.

Committing the cache reservation:

- Adds reserved blocks to existing block tables.
- Adds provisional block tables for newly admitted requests.
- Advances committed cache-slot counts.
- Adds newly admitted requests to the cache manager's allocated-request list.

Committing request bookkeeping:

- Appends the staged token to the host token mirror.
- Sets `processed_sequence_length_` to the sequence length that existed before sampling.
- Changes the status to `Active` or `TurnComplete`.

For a new request or queued continuation, this commit is the point where it moves from `Assigned`
to `Active` or `TurnComplete`. The dynamic transaction path does not need a separate visible
scheduling state between those states.

Finally, the engine swaps the staged ready list into `ready_requests_`. The first ready request is returned immediately, and later calls drain the rest without another model run.

## Constrained decoding and tool calling

Engine requests use the guidance configuration carried by their `GeneratorParams`. This allows
concurrent requests to use different JSON schemas, regular expressions, or Lark grammars. Tool
definitions and chat-template rendering remain application concerns; the Engine constrains the
generated token stream but does not parse tool calls from the decoded output.

Each guided request owns an independent constrained-logits processor. Its mask is applied before
minimum-length, repetition-penalty, and no-repeat-ngram processing, matching Generator ordering.
After sampling, the selected token advances that request's grammar cursor.

The grammar cursor participates in the same transaction as search and paged-cache state. A step
checkpoints it before sampling, retains the advanced cursor on commit, and restores the checkpoint
on rollback. Draining an already-ready request does not advance the cursor again.

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

After a successful rollback, committed request state and committed cache state match the state before the step began. The caller receives an `EngineStepError` with either `RetryableBatchAbort` or `ExecutionCapacityExceeded`, and the engine remains healthy. Calling `Step()` again with unchanged memory availability and workload composition may produce the same capacity failure.

The model may have written data into reserved cache memory before the failure. Releasing the reservation is still safe because those blocks were never added to committed block tables. Future users of those blocks overwrite the relevant slots before treating them as valid cache contents.

### Fatal failure

The engine becomes unhealthy when it cannot prove that all components still agree on state. Examples include:

- Reservation fails after an executable plan was produced.
- An unknown model execution failure occurs.
- Rollback itself fails.
- Any part of the commit boundary fails.
- The scheduler returns an invalid planning outcome.

The engine stores the fatal error and rethrows it on later `Step()` calls. Continuing would risk using request search state and cache block tables from different logical steps.

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

If `Continue()` fails while appending tokens and its Search checkpoint also cannot be restored, the request is closed and the Engine is marked fatally unhealthy. Reusing either would risk combining committed KV state with corrupted Search state.

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

If no request can run because of temporary capacity, `StepDynamic()` reports `CapacityDeferred` instead of returning `nullptr`. Returning `nullptr` would incorrectly tell the caller that no work remains.

## Static engine path

The static engine path is intentionally separate.

`StaticBatchScheduler` builds a fixed batch, `StaticCacheManager` allocates one cache configuration for that batch, and `StaticBatchDecoderIO` prepares traditional batched inputs. Individual requests cannot release cache resources until the static batch is complete.

`StepStatic()` performs decode and sampling directly without the dynamic transaction and reservation protocol.

A resident static request queued by `Continue()` returns to `Active` without reallocating the
batch. Static cache rows still cannot be released independently, and an all-turn-complete batch may
be recycled for new work. Static continuation is therefore valid only while the original
single-request batch remains resident.

A closed request that is already resident in a static batch remains physically retained until that shared batch is recycled. It is not sampled or returned again, but its Request/Search storage can remain alive for the lifetime of the batch.

Changes to shared types such as `Request`, `ScheduledRequests`, `ModelExecutor`, or `SimpleDecoder` should be checked against both paths. This document should be updated only where behavior is shared or where the dynamic path changes.

## Transitional low-level public API shape

The language bindings currently expose the same basic low-level loop. Production hosts are expected to wrap this surface or use its replacement rather than expose it as their stable API:

```python
request.add_tokens(initial_tokens)
engine.add_request(request)

while engine.has_pending_requests():
    ready_request = engine.step()
    if ready_request is not None:
        while ready_request.has_unseen_tokens():
            token = ready_request.get_unseen_token()
            # Stream or process the token.

if request.is_turn_complete():
    request.continue_with(next_turn_tokens)

# Repeat engine.step(), then close the conversation when continuation is no longer needed.
# Explicit removal releases resources immediately; final-handle release otherwise defers cleanup
# until the next add_request() or step() boundary.
engine.remove_request(request)
```

One ready request may be returned several times over its lifetime as new tokens become available. A
turn-complete dynamic request remains cache-resident until explicit removal, which releases dynamic
cache ownership immediately. The unseen-output accessors return generated tokens one at a time in global request order without turn tags.

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

Tests under `test/engine/` provide focused coverage for scheduler planning, paged-cache resources, request and cache invariants, transaction rollback, fatal failures, and ready-result draining. When behavior changes, update both the tests and this document so they continue to describe the same contract.
