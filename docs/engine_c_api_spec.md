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

- preserve the supported request-level `OgaGeneratorParams` configuration for one-sequence,
  one-beam Engine Requests; unsupported modes remain rejected;
- make Request and Turn limits explicit;
- identify every Turn;
- cancel only the intended Turn;
- return typed, caller-buffered Engine events;
- deliver tokens only through events;
- keep ABI-visible structures concrete and fixed-width;
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
  uint64_t max_session_tokens; /* 0 = snapshotted generation_params.search.max_length. */
} OgaRequestOptions;
```

Rules:

- `OgaRequestOptions` is a public fixed-field structure, not an opaque handle.
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
| token-ID stop sequences | A null collection is rejected; a non-null collection is not dereferenced or retained and returns an explicit not-implemented error |
| guidance | Setter returns an explicit not-implemented error and does not retain/dereference input |

Unsupported requested behavior must never be accepted and ignored. `OgaTurnParams` may be reused, but
`OgaRequestBeginTurn` snapshots all supported values before returning. A params object is bound to
the Request that created it; passing it to another Request is rejected. The params object does not
keep the Request alive. Using it after the bound Request is closed or destroyed returns an error.

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

The public token loop pumps `OgaEngineRun`, iterates the populated event prefix, tests the `Token`
bit rather than comparing flags for equality, treats `event.request` as a borrowed identity alias
for the caller's owned Request handle, uses `event.turn_id` as the Request-local Turn identity, and
consumes `event.token` only when `Token` is set.

`OgaFinishReason_StopSequence` reserves the intended vocabulary but is not emitted until stop
sequences are implemented end to end.

### Usage and events

```c
typedef struct OgaTurnUsage {
  uint64_t prompt_tokens;
  uint64_t generated_tokens;
  uint64_t cached_prompt_tokens; /* Currently always 0. */
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
  uint32_t flags;

  OgaRequest* request; /* Borrowed; NULL for Engine-level events. */
  uint64_t turn_id;

  int32_t token;

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

Invalid arguments, incompatible handles, and owner-thread misuse return `OgaResult`; they do not
produce `Failed` events. Capacity pressure is operational and does not set `Failed`.

`prompt_tokens` is the number of input IDs accepted by the current `BeginTurn`.
`generated_tokens` is the number of visible output tokens committed for the Turn.
`cached_prompt_tokens` is currently always zero; no prefix-cache hit accounting is exposed.
Scheduler `max_scheduled_tokens` is not usage: it is a per-step budget shared across Requests.

`OgaEngineEvent` records are caller-allocated as a contiguous array. The implementation validates
the complete buffer range before making progress and fully initializes the populated prefix.

Every function returning `OgaResult*` returns null on success or an owned error object on failure;
the caller releases that object with `OgaDestroyResult`, and its diagnostic string is valid until
then. The C++ wrapper converts a non-null result to `std::runtime_error`. API misuse and validation
failures are reported this way; callers must use event flags and `error_code`, not diagnostic-string
parsing, to classify operational Run outcomes.

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
    OgaEngineEvent* events,
    size_t event_capacity,
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

The Engine records its owner thread when it is created. Engine operations, Request operations, and
Turn-parameter setters are not thread-safe and must be called serially from that owner thread.
Destroying a Request handle only publishes an atomic abandonment marker when it is the final public
handle; the Engine performs the actual close and cleanup at its next owner-thread boundary. This
deferred-release behavior does not permit concurrent access to the Engine or Request.

Destroying an Engine closes all bound Requests and purges events. Surviving Request handles remain
valid closed tombstones that the caller must still destroy. A resident static batch is released as
shared Engine storage during teardown, not through independent per-Request deallocation.

Request creation snapshots the supplied generation parameters. Engine Requests require
`search.batch_size == 1` and `search.num_beams == 1`; `top_p` must be in `[0, 1]`, and `top_k`
must be nonnegative. Guidance fast-forward tokens are unsupported, and guidance type and data
must either both be present or both be absent. The parameters must belong to the same `OgaModel`
instance used to create the Engine. Creation itself does not queue work.

Input IDs are copied before `BeginTurn` returns. The public count is fixed-width and is converted to
`size_t` only after range validation. A pointer/count pair is canonical because an Engine Request is
one sequence; accepting multi-sequence `OgaSequences` would obscure that constraint.
`input_ids_count` must be nonzero for a successful Turn. A null `input_ids` pointer is accepted only
with a zero count, which is then rejected by `BeginTurn` as an empty input.
On success, `OgaEngineCreateRequest` and `OgaRequestCreateTurnParams` return caller-owned handles
that must be released with their matching destroy functions. `BeginTurn` copies supported values
from `turn_params` and does not retain either the params handle or the caller's input buffer.

Turn IDs:

- start at zero independently for every Request;
- increase monotonically;
- are assigned only after successful admission;
- are returned through `out_turn_id`;
- are repeated on every event for that Turn;
- are checked for exhaustion before admission mutates state.

`BeginTurn` is valid for a new Request or after the current Turn is complete. Before a continuation
Turn is admitted, all undelivered events for that Request must be drained with `OgaEngineRun`.
Normal continuation requires the Request's model state to remain resident. A canceled Turn that has
not processed any model tokens may be restarted while nonresident; otherwise the normal residency
and device-specific continuous-decoding rules apply. Static batching supports continuous decoding
only when its resident batch contains one Request.

`CancelTurn` behavior:

| Request/Turn state | Result |
| --- | --- |
| ID matches queued or active Turn | Cancel synchronously and set `out_cancelled = true` |
| ID identifies the completed Turn | Success with `false` |
| ID is stale, unknown, or newer than the current Turn | Success with `false`; never cancel another Turn |
| Request has never started or is closed | Error |

Cancellation is owner-thread-only, preserves committed Request/Search/KV state and visible tokens,
and publishes a `TurnFinished` event with reason `Cancelled`. If an undelivered token event for
the same Turn already exists, cancellation merges the terminal fields into that event rather than
creating a second event. Cancellation is idempotent by Turn ID: the first matching queued or active
Turn returns true, and later calls return false.

`OgaEngineRun` uses caller-owned reusable storage:

- `event_capacity` is the number of records available.
- `out_event_count` receives the number of records written in the buffer prefix and is set to zero
  before later validation or Engine work. It must not overlap the event storage. The complete
  `size_t` object is checked for overlap before the implementation writes through the pointer, so
  the count remains unchanged for that specific validation error. Callers must not rely on the
  count value after any invalid call.
- For positive capacity, `events` points to a contiguous, aligned array of `OgaEngineEvent`.
- The complete storage range, including checked
  `event_capacity * sizeof(OgaEngineEvent)`, is validated before Request reclamation, event
  draining, scheduling, mutation, or model progress.
- Populated records are fully initialized, and unused slots are untouched.

The caller-owned buffer is the public output storage, not an allocation-free guarantee for the C
adapter: the current C implementation creates temporary internal event storage for a positive
capacity call. The C++20 span wrapper passes that caller storage to `OgaEngineRun`, so it has the
same C-adapter allocation behavior.

Capacity zero is an owner-thread-validated no-op. `events` may be null, the count becomes zero, and
the call performs no reclamation, event drain, scheduling, or model work. A permanently unhealthy
Engine still returns its stored fatal error.

## State machine

```text
RequestCreated/Unassigned
   | BeginTurn -> turn_id
   v
Assigned/Active -- Token event -------------------------------+
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

Request Rewind has not yet been implemented. Classic `Generator::RewindToLength` is a separate
Generator API and does not provide Request Rewind. Dynamic Engine transactions may restore Search
checkpoints and release uncommitted cache reservations after a failed step or failed continuation
admission, but that internal rollback is not Request Rewind. Cancellation and completion do not
rewind committed state.

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
  ReclaimAbandonedRequests();

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
- If planning identifies an unserviceable Request, the Engine publishes that Request's failure
  event first and does not execute otherwise fitting work in the same call. Remaining executable
  Requests are considered by a later `Run`.

`OgaEngineHasPendingRequests` returns true when either pending events or schedulable work exist.
Before answering, it reclaims Requests whose final public handle was released. This is an
owner-thread Engine boundary, so abandonment cannot leave stale schedulable work or retained events
hidden behind a false result. A false result does not close or release turn-complete Requests;
callers must close or abandon those handles explicitly.

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
  no Engine progress, except that an overlapping `out_event_count` is rejected before the count can
  safely be written.
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
sets its finish reason to `Failed`, and publishes exactly one terminal event. The Engine remains
healthy; a later `BeginTurn` for that Request is subject to the normal continuation residency rules
(and normally fails on the dynamic path because the failed Request was deallocated).

`ExecutionContractFailure` represents a broken scheduler/transaction invariant.
`FatalExecutionFailure` represents execution or rollback failure that leaves Engine state
untrustworthy. Both mark the Engine permanently unhealthy before publishing the event.

The current implementation centralizes this policy in its typed `EventFromStepError` mapping rather
than parsing diagnostic strings.

## Deferred additions

### Request IDs

Initial events use borrowed pointer identity. Pointer comparison is a constant-time machine-word
comparison. No separate Request ID or lookup API is planned.

### Engine-bound Request parameters

A future `OgaRequestParams` may be created from an Engine and replace `OgaGeneratorParams` in
Request creation. It should preserve search setters while hiding internal `GeneratorParams`.
This is deferred until Search, RNG, sampling, guidance, and execution ownership are separated from
the classic Generator parameter type.

## Language surfaces

### C++

- RAII `OgaTurnParams`.
- `OgaRequest::BeginTurn` returns `uint64_t`.
- `OgaRequest::CancelTurn(uint64_t) -> bool`.
- `OgaEngine::Run(OgaEngineEvent*, size_t)` returns the populated record count and is available in
  C++17 and later.
- When compiled as C++20 or later, `OgaEngine::Run(std::span<OgaEngineEvent>)` returns a
  `std::span<const OgaEngineEvent>` over the populated prefix. The C++17 wrapper still exposes
  the pointer/count overload.
- No unseen-token methods.

### Python

- `Request.begin_turn(tokens, turn_params=None) -> int`.
- `Request.cancel_turn(turn_id) -> bool`.
- `TurnParams.set_stop_sequences(token_id_sequences)` converts the ragged collection to a temporary
  `OgaSequences`; the current C API then raises the explicit not-implemented error. A null C
  collection is rejected before that status is returned.
- `Engine.run(max_events: int = 1) -> list[EngineEvent]`.
- `max_events=0` is the capacity-zero no-op and a negative value is rejected.
- The returned list contains only the populated prefix; its default capacity preserves
  one-event-at-a-time behavior while larger values expose complete transaction output when it fits.
- `EngineEvent` exposes flags, borrowed Request, Turn ID, optional token, finish reason, error code,
  and usage. The event object does not retain the Request; keep the original Python `Request`
  object alive while using `event.request`.
- `Request.begin_turn` requires one logical dimension and converts input to a C-contiguous `int32`
  array when necessary before calling the C++ wrapper. Read-only and strided arrays are accepted
  through that conversion; multidimensional arrays are rejected.
- No `has_unseen_tokens` or `get_unseen_token`.

Examples and integration tests must consume `event.token` and use flag checks. Managed wrappers must
not turn cancellation into a callback that races the owner-thread `Run`; no managed Engine wrapper
is part of the current source surface.

### Benchmarks and examples

- Keep existing `OgaGeneratorParams` setup for Request creation.
- Replace ready-Request lookup and FIFO draining with event handling.
- Preserve application-owned maps when additional metadata is needed.

## Implemented phases

1. Added public `OgaRequestOptions`, opaque `OgaTurnParams`, renamed finish reasons, event flags,
   usage, and caller-buffered event declarations.
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

- Published sizes and offsets are asserted for supported ABIs.
- Null positive-capacity buffers, null count pointers, misaligned buffers, multiplication overflow,
  address-range overflow, and overlapping output counts are rejected before mutation or progress.
- Populated records are fully initialized and unused slots are preserved.
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
