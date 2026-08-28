# Experimental Engine C API redesign

> **Status:** Implemented experimental contract.
>
> This specification describes the experimental Engine C API. It is
> deliberately independent of the classic Generator API except where retaining
> `OgaGeneratorParams` avoids an unnecessary runtime refactor.

## Goals

The public ownership and execution hierarchy is:

```text
Model
  `-- Engine                 one shared runtime configured by model.config.engine
        `-- Request          one resident conversation/session
              `-- Turn       one queued, active, or completed generation invocation
```

The redesign must:

- preserve current request-level generation configurability;
- make Request and Turn limits explicit;
- identify every Turn;
- cancel only the intended Turn;
- return typed, caller-buffered Engine events;
- deliver tokens only through events;
- keep ABI-visible structures fixed-width and extensible;
- preserve the single-host-owner-thread contract.

## C API

### Opaque handles

```c
typedef struct OgaEngine OgaEngine;
typedef struct OgaRequest OgaRequest;
typedef struct OgaTurnParams OgaTurnParams;
```

`OgaTurnParams` is opaque because sampling, stop-sequence, and guidance settings are expected to
evolve and may eventually contain copied pointer-backed data. Adding setter functions is ABI-safe
and avoids public field-presence rules for values where zero is meaningful.

`OgaGeneratorParams` remains in Request creation initially. Internally, `Search`, RNG, sampling,
guidance, allocation limits, and model execution currently consume `GeneratorParams`. Replacing it
with an Engine-bound opaque `OgaRequestParams` is deferred until that ownership is refactored.

### Request options

```c
typedef struct OgaRequestOptions {
  uint32_t struct_size;
  uint32_t reserved; /* Must be zero. */
  uint64_t max_session_tokens; /* 0 = snapshotted generation_params.search.max_length. */
} OgaRequestOptions;
```

Rules:

- `OgaRequestOptions` is a public fixed-field structure, not an opaque handle.
- The caller zero-initializes the structure and sets `struct_size`.
- An undersized structure is rejected before Request mutation.
- A larger structure is accepted; unknown trailing bytes are ignored.
- `reserved` must be zero.
- Conversion from `uint64_t` to internal sizes is checked before mutation.
- Null options or zero `max_session_tokens` use the Request's snapshotted
  `OgaGeneratorParams.search.max_length`. That search value normally defaults from the model context
  length, but the caller may set it lower before Request creation.
- A nonzero `max_session_tokens` may not exceed that snapshotted `search.max_length`.
- The value counts the initial input, generated tokens, and continuation input over the complete
  resident Request.

### Turn parameters

```c
OgaResult* OgaRequestCreateTurnParams(
    OgaRequest* request,
    OgaTurnParams** out);

void OgaDestroyTurnParams(OgaTurnParams* params);

OgaResult* OgaTurnParamsSetMaxGeneratedTokens(
    OgaTurnParams* params,
    uint64_t max_generated_tokens);

OgaResult* OgaTurnParamsSetTemperature(
    OgaTurnParams* params,
    float temperature);

OgaResult* OgaTurnParamsSetTopP(
    OgaTurnParams* params,
    float top_p);

OgaResult* OgaTurnParamsSetTopK(
    OgaTurnParams* params,
    int32_t top_k);

OgaResult* OgaTurnParamsSetSeed(
    OgaTurnParams* params,
    uint64_t seed);

OgaResult* OgaTurnParamsSetStopSequences(
    OgaTurnParams* params,
    const OgaSequences* stop_sequences);

OgaResult* OgaTurnParamsSetGuidance(
    OgaTurnParams* params,
    const char* guidance_type,
    const char* guidance_data);
```

Initial implementation status:

| Setting | Initial behavior |
| --- | --- |
| `max_generated_tokens` | Implemented end to end; zero uses the configured/default limit |
| `temperature`, `top_p`, `top_k`, `seed` | Setter returns an explicit not-implemented error |
| token-ID stop sequences | Setter accepts the existing ragged `OgaSequences` collection, returns an explicit not-implemented error, and does not retain or dereference it |
| guidance | Setter returns an explicit not-implemented error and does not retain/dereference input |

Unsupported requested behavior must never be accepted and ignored. `OgaTurnParams` may be reused, but
`OgaRequestBeginTurn` snapshots all supported values before returning. A params object is bound to
the Request that created it; passing it to another Request is rejected.

### Finish reasons and event flags

```c
typedef enum OgaFinishReason {
  OgaFinishReason_None = 0,
  OgaFinishReason_Eos = 1,
  OgaFinishReason_StopSequence = 2, /* Reserved; not emitted initially. */
  OgaFinishReason_MaxGeneratedTokens = 3,
  OgaFinishReason_MaxSessionTokens = 4,
  OgaFinishReason_Cancelled = 5,
  OgaFinishReason_Failed = 6
} OgaFinishReason;

typedef enum OgaEngineEventFlags {
  OgaEngineEventFlag_None = 0,
  OgaEngineEventFlag_Token = 1u << 0,
  OgaEngineEventFlag_TurnFinished = 1u << 1,
  OgaEngineEventFlag_CapacityBlocked = 1u << 2,
  OgaEngineEventFlag_Failed = 1u << 3,
  OgaEngineEventFlag_Retryable = 1u << 4
} OgaEngineEventFlags;
```

`OgaEngineEventFlags` is a bitmask, not a mutually exclusive type. In particular, a final model step
may emit one event with both `Token` and `TurnFinished`.

The public token loop is `Tokens(request, turn_id)`: callers pump `OgaEngineRun`, iterate the
populated event prefix, test the `Token` bit rather than comparing flags for equality, treat
`event.request` as a borrowed identity alias for their owned Request handle, use `event.turn_id` as
the Request-local Turn identity, and consume `event.token` only when `Token` is set.

`OgaFinishReason_StopSequence` reserves the intended vocabulary but is not emitted until stop
sequences are implemented end to end.

### Usage and events

```c
typedef struct OgaTurnUsage {
  uint64_t prompt_tokens;
  uint64_t generated_tokens;
  uint64_t cached_prompt_tokens; /* Cross-request prefix-cache hits; currently unimplemented, so always 0 */
} OgaTurnUsage;

typedef enum OgaErrorCode {
  OgaErrorCode_None = 0,
  OgaErrorCode_CapacityDeferred = 1,
  OgaErrorCode_ExecutionCapacityExceeded = 2,
  OgaErrorCode_RetryableExecution = 3,
  OgaErrorCode_RequestUnserviceable = 4,
  OgaErrorCode_EngineContractFailure = 5,
  OgaErrorCode_EngineExecutionFailure = 6
} OgaErrorCode;

typedef struct OgaEngineEvent {
  uint32_t struct_size;
  uint32_t flags;

  OgaRequest* request; /* Borrowed; NULL for Engine-level events. */
  uint64_t turn_id;

  int32_t token;
  uint32_t reserved;

  OgaFinishReason finish_reason;
  OgaErrorCode error_code;
  OgaTurnUsage usage;
} OgaEngineEvent;
```

The error codes describe public behavior while preserving the distinctions needed to map existing
`EngineStepError` categories. Numeric values are stable ABI and must not be renumbered.

Payload validity is determined by flags:

| Flags | Valid payload |
| --- | --- |
| `None` | Reserved zero value; no-work is represented by `out_event_count == 0` |
| `Token` | `request`, `turn_id`, and `token` |
| `TurnFinished` | `request`, `turn_id`, `finish_reason`, and `usage` |
| `Token \| TurnFinished` | Final visible token and terminal payload from the same committed step |
| `CapacityBlocked` | `error_code` is `CapacityDeferred` or `ExecutionCapacityExceeded`; Engine and Request remain reusable |
| `Retryable` | `error_code` is `RetryableExecution`; no progress committed and the Engine remains reusable |
| `TurnFinished \| Failed` | Request/Turn failure; finish reason is `Failed` |
| `Failed` without `TurnFinished` | Engine-level failure; `request` is null |

Invalid arguments, invalid structure sizes, incompatible handles, and owner-thread misuse return
`OgaResult`; they do not produce `Failed` events. Capacity pressure is operational and does not set
`Failed`.

`prompt_tokens` is the number of input IDs accepted by the current `BeginTurn`.
`generated_tokens` is the number of visible output tokens committed for the Turn.
`cached_prompt_tokens` has the exact contract documented in the structure comment. Scheduler
`max_scheduled_tokens` is not usage: it is a per-step budget shared across Requests.

`OgaEngineEvent` records are caller-allocated. The caller sets every slot's `struct_size` before
each call. The implementation validates the complete buffer before making progress, preserves each
size, writes no bytes beyond the Version 1 fields, and fully initializes every Version 1 field in
the populated prefix.

### Engine, Request, and Turn operations

```c
OgaResult* OgaCreateEngine(
    OgaModel* model,
    OgaEngine** out);

void OgaDestroyEngine(OgaEngine* engine);

OgaResult* OgaEngineCreateRequest(
    OgaEngine* engine,
    const OgaGeneratorParams* generation_params,
    const OgaRequestOptions* request_options,
    OgaRequest** out);

OgaResult* OgaRequestBeginTurn(
    OgaRequest* request,
    const OgaTurnParams* turn_params,
    const int32_t* input_ids,
    uint64_t input_ids_count,
    uint64_t* out_turn_id);

OgaResult* OgaRequestCancelTurn(
    OgaRequest* request,
    uint64_t turn_id,
    bool* out_cancelled);

OgaResult* OgaRequestClose(OgaRequest* request);

void OgaDestroyRequest(OgaRequest* request);

OgaResult* OgaEngineRun(
    OgaEngine* engine,
    void* event_buffer,
    size_t event_capacity,
    size_t event_stride,
    size_t* out_event_count);

OgaResult* OgaEngineHasPendingRequests(
    OgaEngine* engine,
    bool* out);
```

On successful `OgaCreateEngine`, the Engine retains shared ownership of the underlying Model. The
caller may immediately release its `OgaModel` handle with `OgaDestroyModel`; that releases only the
caller's handle ownership. The Model remains alive until the Engine and any other retaining objects
are destroyed. The input Model handle must be valid during `OgaCreateEngine` and is not usable by the
caller after that handle is released.

Destroying an Engine closes all bound Requests and purges events. Surviving Request handles remain
valid closed tombstones that the caller must still destroy. A resident static batch is released as
shared Engine storage during teardown, not through independent per-Request deallocation.

Input IDs are copied before `BeginTurn` returns. The public count is fixed-width and is converted to
`size_t` only after range validation. A pointer/count pair is canonical because an Engine Request is
one sequence; accepting multi-sequence `OgaSequences` would obscure that constraint.

Turn IDs:

- start at zero independently for every Request;
- increase monotonically;
- are assigned only after successful admission;
- are returned through `out_turn_id`;
- are repeated on every event for that Turn;
- are checked for exhaustion before admission mutates state.

`CancelTurn` behavior:

| Request/Turn state | Result |
| --- | --- |
| ID matches queued or active Turn | Cancel synchronously and set `out_cancelled = true` |
| ID identifies the completed Turn | Success with `false` |
| ID is stale, unknown, or newer than the current Turn | Success with `false`; never cancel another Turn |
| Request has never started or is closed | Error |

Cancellation is owner-thread-only, preserves committed Request/Search/KV state and visible tokens,
and publishes a `TurnFinished` event with reason `Cancelled`.

`OgaEngineRun` uses caller-owned reusable storage:

- `event_capacity` is the number of records available.
- `event_stride` is the byte distance between record starts. It must be aligned for
  `OgaEngineEvent` and at least `sizeof(OgaEngineEvent)`.
- `out_event_count` receives the number of records written in the buffer prefix and is set to zero
  before later validation or Engine work. It must not overlap the event storage.
- Every positive-capacity slot must be aligned, have
  `sizeof(OgaEngineEvent) <= struct_size <= event_stride`, and have `reserved == 0`.
- The complete storage layout, including checked `event_capacity * event_stride`, is validated
  before Request reclamation, event draining, scheduling, mutation, or model progress. An invalid
  later slot therefore cannot cause partial event writes or partial Engine progress.
- The caller's `struct_size` is preserved for every returned record. Version 1 fields are fully
  initialized, trailing bytes are untouched, and unused slots are untouched.

Capacity zero is an owner-thread-validated no-op. `event_buffer` may be null, `event_stride` is
ignored, the count becomes zero, and the call performs no reclamation, event drain, scheduling, or
model work. A permanently unhealthy Engine still returns its stored fatal error.

## State machine

```text
Created
   | BeginTurn -> turn_id
   v
TurnActive -- Token events ------------------------------------+
   |                                                          |
   +-- Token | TurnFinished(reason) --> TurnComplete           |
   |                                      |                   |
   +-- CancelTurn(same turn_id) -----------+                   |
                                          | BeginTurn          |
                                          +-------------------+

Any state -- Close --> Closed
```

There is no Request `Continue` operation. Initial input, tool results, and later user input all use
`BeginTurn`.

There is no public Request rewind operation. Classic `Generator::RewindToLength` is unrelated.
Dynamic Engine transactions may restore Search checkpoints and release uncommitted cache
reservations after a failed step or failed continuation admission. Cancellation and completion do
not rewind committed state.

## Event production and delivery

The implementation replaces ready-Request retention and unseen-token delivery with typed pending
events:

```cpp
struct PendingEngineEvent {
  std::shared_ptr<Request> request;
  uint64_t turn_id{};
  uint32_t flags{};
  int32_t token{};
  GenerationFinishReason finish_reason{};
  TurnUsage usage{};
  EngineErrorCode error_code{};
};
```

At Engine construction:

```cpp
pending_events_.reserve(cache_manager_->MaxBatchSize());
```

`reserve` allocates capacity without constructing events. A committed step produces at most one
combined event per affected Request, so the maximum retained count is bounded by the scheduled batch
size. Reserving before execution prevents normal event publication from allocating after model and
cache state have committed.

Conceptual positive-capacity `Run` flow:

```cpp
size_t Engine::Run(std::span<EngineEvent> output) {
  ReclaimAbandonedRequestsAndEvents();

  if (HasRetainedEvents())
    return DrainPendingEvents(output);

  if (!scheduler_->HasPendingRequests())
    return 0;

  ExecuteAndCommitAtMostOneStep();
  PopulatePendingEventsFromCommittedResults();
  return DrainPendingEvents(output);
}
```

The operation is drain-or-execute:

- If retained events exist at call entry, the call drains up to capacity in FIFO order and performs
  no model transaction, even if output capacity remains.
- Otherwise the call executes at most one static step or one dynamic transaction.
- If that committed step produces events for Requests A, B, and C and capacity is at least three,
  all three are returned by the same call.
- If capacity is one, A is returned and B/C remain retained. Later calls drain B/C without model
  execution. Only after retained overflow is empty may a later call schedule more work.
- A chunked partial-prefill transaction may commit cache and Request progress while producing no
  event. In that case the call succeeds with count zero and `HasPendingRequests()` remains true. The
  Engine does not loop internally to force token output.

`OgaEngineHasPendingRequests` returns true when either pending events or schedulable work exist.
Before answering, it reclaims Requests whose final public handle was released. This is an
owner-thread Engine boundary, so abandonment cannot leave stale schedulable work or retained events
hidden behind a false result.

The Engine API has no public asynchronous event queue. Hosts that want uninterrupted inference copy
events into an application-owned bounded queue and continue pumping the owner thread. If the host
does not drain, Engine progress pauses and memory remains bounded.

Closing a Request removes its undelivered internal events. Events already copied by the application
are application-owned and must be discarded there if no longer useful. Destroying an Engine closes
its Requests and discards pending events.

Close and final-handle abandonment both logically remove a Request from scheduling and purge its
undelivered events. On the dynamic path, committed paged-cache ownership is released immediately.
On the static path, a resident row is part of a shared batch allocation and cannot be physically
released per Request; the closed/abandoned Request is no longer scheduled or returned, but its row,
Request/Search storage, and shared cache allocation may remain until the whole batch recycles.

The event's `request` is borrowed. It remains valid only while the caller retains the owned Request
handle. Internal pending events hold a `shared_ptr<Request>` until delivery, but that does not relax
the public owned-handle lifetime contract.

## Token delivery

Engine events are the only public generated-token delivery mechanism. Remove:

```c
OgaRequestHasUnseenTokens
OgaRequestGetUnseenToken
```

and remove their internal unseen-index FIFO bookkeeping.

At successful step commit, the Engine already knows the Request, Turn ID, selected token, terminal
state, finish reason, and usage. It captures those values in `PendingEngineEvent`. One committed
step produces at most one combined event per affected Request. `OgaEngineRun` copies the available
FIFO prefix directly to caller storage and retains overflow.

`tokens_host_` and the Search sequence remain authoritative resident conversation state. Event
delivery does not remove tokens from that state. Because token and Turn ID are captured together at
commit, no `GeneratedTokenRecord` or per-Turn token range is required.

Callers must never read `event.token` merely because a record was returned. They check
`event.flags & OgaEngineEventFlag_Token`; the same event can also carry
`OgaEngineEventFlag_TurnFinished`.

## Failure and capacity behavior

- Every `EngineStepError` is mapped by its typed `StepOutcomeKind`; error strings are never parsed to
  determine behavior.
- Retryable capacity pressure produces `CapacityBlocked` without poisoning the Engine.
- A non-capacity retryable abort produces `Retryable`; the transaction is fully rolled back before
  publication.
- An unserviceable Request produces `TurnFinished | Failed`, is removed from scheduler/cache
  admission, and does not poison unrelated Requests.
- A fatal Engine failure produces `Failed` with a null Request and prevents later model execution.
- Detailed diagnostics continue to use the library's established result/error facilities.
- Output validation, API misuse, and owner-thread violations return `OgaResult` with count zero and
  no Engine progress.
- Capacity, retryable, unserviceable, contract-failure, and first fatal-execution outcomes remain
  typed events. After the fatal event is delivered, later `Run` calls return the stored fatal error.

The required mapping is:

| Internal `StepOutcomeKind` | Event flags | `OgaErrorCode` | Engine reusable | Request reusable |
| --- | --- | --- | --- | --- |
| `NoWork` | `None` | `None` | Yes | Unchanged |
| `Committed` | `Token`, `TurnFinished`, or both | `None` | Yes | Yes |
| `CapacityDeferred` | `CapacityBlocked` | `CapacityDeferred` | Yes | Yes |
| `ExecutionCapacityExceeded` | `CapacityBlocked` | `ExecutionCapacityExceeded` | Yes | Yes |
| `RetryableBatchAbort` | `Retryable` | `RetryableExecution` | Yes | Yes |
| `UnserviceableRequest` | `TurnFinished \| Failed` | `RequestUnserviceable` | Yes | No current-Turn progress |
| `ExecutionContractFailure` | `Failed` | `EngineContractFailure` | No | No further execution |
| `FatalExecutionFailure` | `Failed` | `EngineExecutionFailure` | No | No further execution |

`UnserviceableRequest` is terminal for the affected Turn because retrying the unchanged Request
would loop forever. The Engine resolves the internal Request pointer, removes it from admission,
sets its finish reason to `Failed`, and publishes exactly one terminal event.

`ExecutionContractFailure` represents a broken scheduler/transaction invariant.
`FatalExecutionFailure` represents execution or rollback failure that leaves Engine state
untrustworthy. Both mark the Engine permanently unhealthy before publishing the event.

Implementation should centralize this policy in a typed mapping helper rather than duplicating it
across planning, execution, post-processing, and commit paths.

## Deferred additions

### Request IDs

Initial events use borrowed pointer identity. Pointer comparison is a constant-time machine-word
comparison. A later version may append an Engine-local monotonic `request_id` to
`OgaEngineEvent` and add `OgaRequestGetId`. The single caller-sized event structure permits this
without writing beyond older callers' buffers. No ID lookup API is planned.

### Engine-bound Request parameters

A future `OgaRequestParams` may be created from an Engine and replace `OgaGeneratorParams` in
Request creation. It should preserve search setters while hiding internal `GeneratorParams`.
This is deferred until Search, RNG, sampling, guidance, and execution ownership are separated from
the classic Generator parameter type.

### Larger event records

Future event fields can be appended to `OgaEngineEvent`. Callers opt in by supplying a larger
record, setting `struct_size` to that record size, and using an aligned stride at least that large.
Version 1 preserves the size and trailing bytes rather than assuming the extra fields exist.

## Language surfaces

### C++

- RAII `OgaTurnParams`.
- `OgaRequest::BeginTurn` returns `uint64_t`.
- `OgaRequest::CancelTurn(uint64_t) -> bool`.
- `OgaEngine::Run(std::span<OgaEngineEvent>)` returns a
  `std::span<const OgaEngineEvent>` over the populated prefix.
- The convenience wrapper accepts exact Version 1 storage and initializes only `struct_size` and
  `reserved` in every reusable slot. Future-sized or padded records use the C API directly.
- No unseen-token methods.

### Python

- `Request.begin_turn(tokens, turn_params=None) -> int`.
- `Request.cancel_turn(turn_id) -> bool`.
- `TurnParams.set_stop_sequences(token_id_sequences)` accepts a ragged collection of token-ID
  sequences and currently raises the explicit not-implemented error.
- `Engine.run(max_events: int = 1) -> list[EngineEvent]`.
- `max_events=0` is the capacity-zero no-op and a negative value is rejected.
- The returned list contains only the populated prefix; its default capacity preserves
  one-event-at-a-time behavior while larger values expose complete transaction output when it fits.
- `EngineEvent` exposes flags, borrowed Request, Turn ID, optional token, finish reason, error code,
  and usage.
- No `has_unseen_tokens` or `get_unseen_token`.

Examples and integration tests must consume `event.token` and use flag checks. Managed wrappers must
not turn cancellation into a callback that races the owner-thread `Run`.

### Benchmarks and examples

- Keep existing `OgaGeneratorParams` setup for Request creation.
- Replace ready-Request lookup and FIFO draining with event handling.
- Preserve application-owned maps when additional metadata is needed.

## Implemented phases

1. Added fixed-width public `OgaRequestOptions`, opaque `OgaTurnParams`, renamed finish reasons, event flags,
   usage, and caller-sized event declarations.
2. Implemented Turn parameter creation, supported limit snapshotting, explicit unsupported setters,
   zero-based Turn IDs, and named cancellation.
3. Replaced internal ready-Request retention with pre-reserved pending event storage.
4. Captured token and terminal payload atomically at transaction commit on dynamic and static paths.
5. Replaced `OgaEngineRun` and removed unseen-token delivery.
6. Updated C++, Python, examples, benchmarks, architecture documentation, and tests.

Do not temporarily maintain two token-delivery paths unless required solely to keep an intermediate
commit buildable; the completed change exposes events only.

## Acceptance criteria

### ABI and validation

- Exact V1 sizes and offsets are asserted for supported ABIs.
- Null positive-capacity buffers, null count pointers, misaligned buffers, short or misaligned
  strides, multiplication overflow, undersized records, `struct_size > stride`, and nonzero
  reserved fields are rejected before mutation or progress.
- Every slot is validated before any slot is written; invalid later slots leave the complete event
  buffer unchanged.
- Larger structures and padded strides are accepted. Caller `struct_size`, trailing bytes, unused
  slots, and reusable headers are preserved.
- Capacity zero validates the owner thread and otherwise does no work.
- `uint64_t` counts that do not fit internal types are rejected.
- Unsupported Turn setters return explicit not-implemented errors.

### Request and Turn lifecycle

- Request settings are snapshotted at creation.
- Turn IDs begin at zero and failed admission does not consume an ID.
- IDs increase across continuation Turns and exhaustion is rejected before mutation.
- Named stale cancellation cannot cancel a successor Turn.
- Repeated cancellation is idempotent.
- Cancellation preserves committed state and reports exactly one terminal event.

### Event delivery

- No event output returns count zero; partial prefill can do so while work remains.
- One token event is emitted for every visible generated token.
- A final visible token and completion are emitted together.
- Every token and terminal event carries the correct Turn ID.
- A multi-Request transaction returns all events in one call when capacity suffices.
- Capacity one executes once and drains retained overflow on later calls without more inference.
- Draining retained events never executes a new transaction, even when the output buffer has spare
  capacity.
- Static multi-row output follows the same bulk delivery contract.
- `HasPendingRequests` remains true while events are retained.
- Event output validation failure causes no model progress.
- Closing one Request removes only that Request's retained events.
- Abandonment and Engine destruction leave no borrowed pending handles.

### Usage and reasons

- Prompt and generated counts are correct for initial, continuation, EOS, limit, and cancelled Turns.
- Cached prompt tokens are zero.
- EOS, max-generated, max-session, cancelled, and failed reasons are distinguished.
- StopSequence is never emitted until implemented.

### Failure and capacity

- Scheduler deferral emits `CapacityBlocked/CapacityDeferred` and leaves all state reusable.
- Execution capacity exhaustion emits `CapacityBlocked/ExecutionCapacityExceeded` after rollback.
- Retryable non-capacity abort emits `Retryable/RetryableExecution` after rollback.
- An unserviceable Request is removed from admission, emits exactly one
  `TurnFinished | Failed/RequestUnserviceable` event, and does not affect unrelated Requests.
- Contract failure emits `Failed/EngineContractFailure`, has no borrowed Request, and permanently
  prevents later execution.
- Fatal execution failure emits `Failed/EngineExecutionFailure`, has no borrowed Request, and
  permanently prevents later execution.
- API misuse returns `OgaResult` without a misleading operational event.

### Cross-surface behavior

- C and C++ compile smokes use the final declarations.
- Python unit and integration tests consume events only.
- Static and dynamic Engine tests cover token-only, terminal-only, and combined events.
- Examples and benchmarks contain no unseen-token calls.
- Windows and Linux/WSL builds cover the fixed-width ABI and wrappers.
