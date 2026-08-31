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
- return typed Engine events through reusable opaque storage;
- deliver tokens only through events;
- keep public Engine handles opaque and behavioral constants fixed-width;
- preserve the single-host-owner-thread contract.

## C API

### Opaque handles

```c
typedef struct OgaEngine OgaEngine;
typedef struct OgaEngineEvent OgaEngineEvent;
typedef struct OgaEngineEventBuffer OgaEngineEventBuffer;
typedef struct OgaRequest OgaRequest;
typedef struct OgaRequestOptions OgaRequestOptions;
typedef struct OgaTurnOptions OgaTurnOptions;
typedef struct OgaTurnUsage OgaTurnUsage;
```

Every new Engine API object is opaque. `OgaRequestOptions` and `OgaTurnOptions` can evolve through
setters without public field-presence rules. `OgaEngineEvent` and `OgaTurnUsage` expose data only
through getters, so the API publishes no layout, size, alignment, version, or stride contract.
`OgaEngineEventBuffer` owns the event objects and exposes borrowed views.

`OgaGeneratorParams` remains in Request creation initially. Internally, `Search`, RNG, sampling,
guidance, allocation limits, and model execution currently consume `GeneratorParams`. Replacing it
with an Engine-bound opaque `OgaRequestParams` is deferred until that ownership is refactored.

### Request options

```c
OgaResult* OgaCreateRequestOptions(OgaRequestOptions** out);

void OgaDestroyRequestOptions(OgaRequestOptions* options);

OgaResult* OgaRequestOptionsSetMaxSessionTokens(
    OgaRequestOptions* options,
    uint64_t max_session_tokens);
```

Rules:

- `OgaRequestOptions` is an opaque, reusable, caller-owned handle.
- Conversion from `uint64_t` to internal sizes is checked before mutation.
- Null options or zero `max_session_tokens` use the Request's snapshotted
  `OgaGeneratorParams.search.max_length`. That search value normally defaults from the model context
  length, but the caller may set it lower before Request creation.
- A nonzero `max_session_tokens` may not exceed that snapshotted `search.max_length`.
- The value counts the initial input, generated tokens, and continuation input over the complete
  resident Request.

### Turn options

```c
OgaResult* OgaRequestCreateTurnOptions(
    OgaRequest* request,
    OgaTurnOptions** out);

void OgaDestroyTurnOptions(OgaTurnOptions* options);

OgaResult* OgaTurnOptionsSetMaxGeneratedTokens(
    OgaTurnOptions* options,
    uint64_t max_generated_tokens);

OgaResult* OgaTurnOptionsSetTemperature(
    OgaTurnOptions* options,
    float temperature);

OgaResult* OgaTurnOptionsSetTopP(
    OgaTurnOptions* options,
    float top_p);

OgaResult* OgaTurnOptionsSetTopK(
    OgaTurnOptions* options,
    int32_t top_k);

OgaResult* OgaTurnOptionsSetSeed(
    OgaTurnOptions* options,
    uint64_t seed);

OgaResult* OgaTurnOptionsSetStopTokenIds(
    OgaTurnOptions* options,
    const OgaSequences* stop_token_ids);

OgaResult* OgaTurnOptionsSetStopStrings(
    OgaTurnOptions* options,
    const OgaStringArray* stop_strings);

OgaResult* OgaTurnOptionsSetGuidance(
    OgaTurnOptions* options,
    const char* guidance_type,
    const char* guidance_data);
```

Initial implementation status:

| Setting | Initial behavior |
| --- | --- |
| `max_generated_tokens` | Implemented end to end; zero uses the configured/default limit |
| `temperature`, `top_p`, `top_k`, `seed` | Setter returns an explicit not-implemented error |
| token-ID stop sequences and UTF-8 stop strings | Each has a distinct setter. A null collection is rejected; a non-null collection is not dereferenced or retained and returns an explicit not-implemented error |
| guidance | Setter returns an explicit not-implemented error and does not retain/dereference input |

Unsupported requested behavior must never be accepted and ignored. `OgaTurnOptions` may be reused,
but `OgaRequestBeginTurn` snapshots all supported values before returning. An options object is
bound to the Request that created it; passing it to another Request is rejected. The options object does not
keep the Request alive. Using it after the bound Request is closed or destroyed returns an error.

### Finish reasons and event flags

```c
typedef uint32_t OgaFinishReason;
#define OgaFinishReason_None ((OgaFinishReason)0)
#define OgaFinishReason_Eos ((OgaFinishReason)1)
#define OgaFinishReason_StopSequence ((OgaFinishReason)2)
#define OgaFinishReason_MaxGeneratedTokens ((OgaFinishReason)3)
#define OgaFinishReason_MaxSessionTokens ((OgaFinishReason)4)
#define OgaFinishReason_Cancelled ((OgaFinishReason)5)
#define OgaFinishReason_Failed ((OgaFinishReason)6)

typedef uint32_t OgaEngineEventFlags;
#define OgaEngineEventFlag_None ((OgaEngineEventFlags)0)
#define OgaEngineEventFlag_Token ((OgaEngineEventFlags)(1u << 0))
#define OgaEngineEventFlag_TurnFinished ((OgaEngineEventFlags)(1u << 1))
#define OgaEngineEventFlag_CapacityBlocked ((OgaEngineEventFlags)(1u << 2))
#define OgaEngineEventFlag_Failed ((OgaEngineEventFlags)(1u << 3))
#define OgaEngineEventFlag_Retryable ((OgaEngineEventFlags)(1u << 4))
```

`OgaEngineEventFlags` is a bitmask, not a mutually exclusive type. In particular, a final model step
may emit one event with both `Token` and `TurnFinished`.

The public types are fixed-width integer typedefs rather than C enums. Adding a constant therefore
cannot change a field or parameter's ABI width under compiler enum-size options.

The public token loop pumps `OgaEngineRun`, iterates the populated event prefix, tests the `Token`
bit rather than comparing flags for equality, treats the Request getter result as a borrowed
identity alias for the caller's owned Request handle, uses the Turn ID getter as the Request-local
Turn identity, and consumes the token getter only when `Token` is set.

`OgaFinishReason_StopSequence` reserves the intended vocabulary but is not emitted until stop
sequences are implemented end to end.

### Usage and events

```c
typedef struct OgaEngineEvent OgaEngineEvent;
typedef struct OgaEngineEventBuffer OgaEngineEventBuffer;
typedef struct OgaTurnUsage OgaTurnUsage;

typedef uint32_t OgaErrorCode;
#define OgaErrorCode_None ((OgaErrorCode)0)
#define OgaErrorCode_CapacityDeferred ((OgaErrorCode)1)
#define OgaErrorCode_ExecutionCapacityExceeded ((OgaErrorCode)2)
#define OgaErrorCode_RetryableExecution ((OgaErrorCode)3)
#define OgaErrorCode_RequestUnserviceable ((OgaErrorCode)4)
#define OgaErrorCode_EngineContractFailure ((OgaErrorCode)5)
#define OgaErrorCode_EngineExecutionFailure ((OgaErrorCode)6)

OgaResult* OgaEngineEventGetFlags(
    const OgaEngineEvent* event, OgaEngineEventFlags* out);
OgaResult* OgaEngineEventGetRequest(
    const OgaEngineEvent* event, const OgaRequest** out);
OgaResult* OgaEngineEventGetTurnId(
    const OgaEngineEvent* event, uint64_t* out);
OgaResult* OgaEngineEventGetToken(
    const OgaEngineEvent* event, int32_t* out);
OgaResult* OgaEngineEventGetFinishReason(
    const OgaEngineEvent* event, OgaFinishReason* out);
OgaResult* OgaEngineEventGetErrorCode(
    const OgaEngineEvent* event, OgaErrorCode* out);
OgaResult* OgaEngineEventGetUsage(
    const OgaEngineEvent* event, const OgaTurnUsage** out);

OgaResult* OgaTurnUsageGetPromptTokens(
    const OgaTurnUsage* usage, uint64_t* out);
OgaResult* OgaTurnUsageGetGeneratedTokens(
    const OgaTurnUsage* usage, uint64_t* out);
OgaResult* OgaTurnUsageGetCachedPromptTokens(
    const OgaTurnUsage* usage, uint64_t* out);
```

The error codes describe public behavior while preserving the distinctions needed to map existing
`EngineStepError` categories. Numeric values are stable ABI and must not be renumbered.

Payload validity is determined by flags:

| Flags | Valid payload |
| --- | --- |
| `None` | Reserved zero value; no-work is represented by Buffer count zero |
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

Event getters require non-null event and output pointers and return `OgaResult*` on misuse. They
initialize the output to its zero/null value before rejecting a null event. The count and indexed
Buffer accessors are the two deliberate direct-return exceptions: count returns zero for a null
Buffer, and indexed access returns null for a null Buffer or an out-of-range index.

`OgaEngineEventGetRequest` returns a const borrowed identity alias, never an owned Request handle.
`OgaEngineEventGetUsage` returns a borrowed usage view. Event and usage pointers are valid only
until the next validated `OgaEngineRun` using their Buffer or Buffer destruction.

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
    const OgaTurnOptions* turn_options,
    const int32_t* input_ids,
    uint64_t input_ids_count,
    uint64_t* out_turn_id);

OgaResult* OgaRequestCancelTurn(
    OgaRequest* request,
    uint64_t turn_id,
    bool* out_cancelled);

OgaResult* OgaRequestClose(OgaRequest* request);

void OgaDestroyRequest(OgaRequest* request);

OgaResult* OgaCreateEngineEventBuffer(
    OgaEngine* engine,
    size_t capacity,
    OgaEngineEventBuffer** out);

void OgaDestroyEngineEventBuffer(
    OgaEngineEventBuffer* buffer);

OgaResult* OgaEngineRun(
    OgaEngine* engine,
    OgaEngineEventBuffer* buffer);

size_t OgaEngineEventBufferGetCount(
    const OgaEngineEventBuffer* buffer);

const OgaEngineEvent* OgaEngineEventBufferGet(
    const OgaEngineEventBuffer* buffer,
    size_t index);

OgaResult* OgaEngineHasPendingRequests(
    OgaEngine* engine,
    bool* out);
```

Typical event draining creates the Buffer once and reuses it:

```c
OgaEngineEventBuffer* buffer = NULL;
OgaCreateEngineEventBuffer(engine, 4, &buffer);
while (has_pending) {
  OgaEngineRun(engine, buffer);
  size_t count = OgaEngineEventBufferGetCount(buffer);
  for (size_t i = 0; i < count; ++i) {
    const OgaEngineEvent* event = OgaEngineEventBufferGet(buffer, i);
    OgaEngineEventFlags flags = OgaEngineEventFlag_None;
    OgaEngineEventGetFlags(event, &flags);
    /* Read only payloads selected by flags. */
  }
}
OgaDestroyEngineEventBuffer(buffer);
```

On successful `OgaCreateEngine`, the Engine retains shared ownership of the underlying Model. The
caller may immediately release its `OgaModel` handle with `OgaDestroyModel`; that releases only the
caller's handle ownership. The Model remains alive until the Engine and any other retaining objects
are destroyed. The input Model handle must be valid during `OgaCreateEngine` and is not usable by the
caller after that handle is released.

The Engine records its owner thread when it is created. Engine operations, Request operations, and
Turn-option setters are not thread-safe and must be called serially from that owner thread.
Destroying a Request handle only publishes an atomic abandonment marker when it is the final public
handle; this final release may occur on another thread. The Engine strongly retains the Request and
performs the actual close, runtime-state release, and cleanup at its next owner-thread boundary.
This deferred-release behavior does not permit concurrent Request operations.

Destroying an Engine closes all bound Requests and purges events. Surviving Request handles remain
valid lightweight closed tombstones that the caller must still destroy. Teardown uses a no-throw
detach path that releases scheduler/cache ownership and Request runtime state without relying on the
already-expired weak Engine reference. A resident static batch is released as shared Engine storage.

An Event Buffer stores a weak Engine reference and does not keep its bound Engine alive. Run rejects
a Buffer created by another Engine. Both the Engine and Buffer handles must be live for every Run;
after Engine destruction, the Buffer remains owned by the caller and must still be destroyed without
being passed to Run. Buffer creation and Run honor the Engine owner thread. Buffer access, Run, and
destruction are serialized; getters may read immutable views but must not race a Run or destruction.

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
On success, `OgaCreateRequestOptions`, `OgaEngineCreateRequest`, and
`OgaRequestCreateTurnOptions` return caller-owned handles that must be released with their matching
destroy functions. Request and Turn creation copy supported values from their options handles and
do not retain either handle or the caller's input buffer.

Turn IDs:

- start at one independently for every Request; zero means no Turn;
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

`OgaEngineRun` uses an opaque, caller-owned reusable Buffer:

- Buffer creation allocates storage for exactly the requested event capacity.
- Run reuses that storage directly; it does not allocate a temporary event vector.
- After handle, binding, and owner-thread validation, Run invalidates the Buffer's prior event and
  usage views and resets its count.
- A successful Run exposes the populated prefix through `GetCount` and `Get`.
- Events beyond the Buffer capacity remain retained by the Engine and drain on later calls.
- The Buffer is permanently bound to the creating Engine and does not keep that Engine alive.

Capacity zero is an owner-thread-validated no-op. The Buffer count remains zero and the call
performs no reclamation, event drain, scheduling, or model work. A permanently unhealthy Engine
still returns its stored fatal error.

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

`reserve` allocates capacity without constructing events. A speculative transaction can produce up
to `max_draft_tokens_per_step + 1` token events per affected Request. Request creation grows the
retained-event capacity for the tracked Request count because fatal handling publishes a terminal
event for every executable Turn, including Requests outside the failed batch. This keeps event
publication allocation-free after model/cache commit and after the Engine becomes unhealthy.

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

Close and final-handle abandonment both logically remove a Request from scheduling, purge its
undelivered events, and release its Search, guidance, sampler, and parameter runtime state on the
owner thread. On the dynamic path, committed paged-cache ownership is released immediately. On the
static path, a resident row is part of a shared batch allocation and cannot be physically released
per Request; the closed/abandoned tombstone is no longer scheduled or returned, but its row and
shared cache allocation may remain until the whole batch recycles.

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

At successful step commit, the Engine already knows the Request, Turn ID, visible tokens, terminal
state, finish reason, and usage. It captures those values in `PendingEngineEvent`. A speculative
transaction emits accepted drafts followed by its correction or bonus token; only the final event
carries terminal state when the Turn finishes. `OgaEngineRun` moves the available FIFO prefix into
the reusable Buffer storage and retains overflow.

`tokens_host_` and the Search sequence remain authoritative resident conversation state. Event
delivery does not remove tokens from that state. Because token and Turn ID are captured together at
commit, no `GeneratedTokenRecord` or per-Turn token range is required.

Callers must never consume the token getter merely because an event was returned. They check
`flags & OgaEngineEventFlag_Token`; the same event can also carry
`OgaEngineEventFlag_TurnFinished`.

## Failure and capacity behavior

- Every `EngineStepError` is mapped by its typed `StepOutcomeKind`; error strings are never parsed to
  determine behavior.
- Retryable capacity pressure produces `CapacityBlocked` without poisoning the Engine.
- A non-capacity retryable abort produces `Retryable`; the transaction is fully rolled back before
  publication.
- An unserviceable Request produces `TurnFinished | Failed`, is removed from scheduler/cache
  admission, and does not poison unrelated Requests.
- A fatal Engine failure produces `TurnFinished | Failed` for every affected Turn and prevents
  later model execution. If no Turn is affected, it produces `Failed` with a null Request.
- Detailed diagnostics continue to use the library's established result/error facilities.
- Invalid Buffer use, API misuse, and owner-thread violations return `OgaResult` with Buffer count
  zero and no Engine progress.
- Capacity, retryable, unserviceable, contract-failure, and first fatal-execution outcomes remain
  typed events. After all fatal events are delivered, later `Run` calls return the stored fatal
  error.

The required mapping is:

| Internal `StepOutcomeKind` | Event flags | `OgaErrorCode` | Engine reusable | Request reusable |
| --- | --- | --- | --- | --- |
| `NoWork` | `None` | `None` | Yes | Unchanged |
| `Committed` | `Token`, `TurnFinished`, or both | `None` | Yes | Yes |
| `CapacityDeferred` | `CapacityBlocked` | `CapacityDeferred` | Yes | Yes |
| `ExecutionCapacityExceeded` | `CapacityBlocked` | `ExecutionCapacityExceeded` | Yes | Yes |
| `RetryableBatchAbort` | `Retryable` | `RetryableExecution` | Yes | Yes |
| `UnserviceableRequest` | `TurnFinished \| Failed` | `RequestUnserviceable` | Yes | No current-Turn progress |
| `ExecutionContractFailure` | `TurnFinished \| Failed` per affected Turn | `EngineContractFailure` | No | No further execution |
| `FatalExecutionFailure` | `TurnFinished \| Failed` per affected Turn | `EngineExecutionFailure` | No | No further execution |

`UnserviceableRequest` is terminal for the affected Turn because retrying the unchanged Request
would loop forever. The Engine resolves the internal Request pointer, removes it from admission,
sets its finish reason to `Failed`, and publishes exactly one terminal event. The Engine remains
healthy; a later `BeginTurn` for that Request is subject to the normal continuation residency rules
(and normally fails on the dynamic path because the failed Request was deallocated).

`ExecutionContractFailure` represents a broken scheduler/transaction invariant.
`FatalExecutionFailure` represents execution or rollback failure that leaves Engine state
untrustworthy. Both mark the Engine permanently unhealthy, mark every executable Turn failed, and
publish its terminal event before later `Run` calls return the stored fatal error. With no affected
Turn, the Engine publishes one request-less `Failed` event instead.

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

- RAII `OgaRequestOptions` and `OgaTurnOptions`.
- RAII `OgaEngineEventBuffer`, created once with `OgaEngine::CreateEventBuffer(capacity)`.
- `OgaRequest::BeginTurn` returns `uint64_t`.
- `OgaRequest::CancelTurn(uint64_t) -> bool`.
- `OgaEngine::Run(OgaEngineEventBuffer&)` returns the populated count.
- `OgaEngineEventBuffer::Get(index)` returns a borrowed event pointer.
- Event and usage wrappers expose getters only. `OgaEngineEvent::Request()` returns an optional
  non-owning `std::reference_wrapper<const OgaRequest>`; it never exposes an owning, mutable, or
  deletable handle for the borrowed Request.
- No unseen-token methods.

### Python

- `RequestOptions.set_max_session_tokens(value)` configures the cumulative Request limit.
- `Engine.create_request(params, options=None) -> Request`.
- `Request.begin_turn(tokens, turn_options=None) -> int`.
- `Request.cancel_turn(turn_id) -> bool`.
- `TurnOptions.set_stop_token_ids(token_id_sequences)` converts the ragged collection to a temporary
  `OgaSequences`; `TurnOptions.set_stop_strings(strings)` converts strings to a temporary
  `OgaStringArray`. The current C API returns the explicit not-implemented error for either
  non-null collection.
- `Engine.create_event_buffer(capacity) -> EngineEventBuffer`.
- `Engine.run(buffer) -> EngineEventBuffer`; the returned object is the same reusable Buffer and is
  a Python sequence over its populated borrowed event views. The binding releases the GIL for the
  complete native Run and reacquires it before returning or translating an exception.
- A zero-capacity Buffer is the capacity-zero no-op. Negative or unrepresentable capacities are
  rejected during Buffer creation.
- `EngineEvent` exposes flags, borrowed Request, Turn ID, optional token, finish reason, error code,
  and borrowed usage. Indexed event views keep their Buffer object alive. Both event and usage data
  are invalidated by the next Run using that Buffer, so applications copy any values that must
  persist.
- `Request.begin_turn` requires one logical dimension and converts input to a C-contiguous `int32`
  array when necessary before calling the C++ wrapper. Read-only and strided arrays are accepted
  through that conversion; multidimensional arrays are rejected.
- No `has_unseen_tokens` or `get_unseen_token`.

Examples and integration tests must consume tokens through event views and use flag checks. Managed wrappers must
not turn cancellation into a callback that races the owner-thread `Run`; no managed Engine wrapper
is part of the current source surface.

### Benchmarks and examples

- Keep existing `OgaGeneratorParams` setup for Request creation.
- Replace ready-Request lookup and FIFO draining with event handling.
- Preserve application-owned maps when additional metadata is needed.

## Implemented phases

1. Added opaque Engine option, event, usage, and reusable Buffer handles; kept finish reasons,
   event flags, and error codes fixed-width.
2. Implemented Turn option creation, supported limit snapshotting, explicit unsupported setters,
   nonzero Turn IDs, and named cancellation.
3. Replaced internal ready-Request retention with pre-reserved pending event storage.
4. Captured token and terminal payload atomically at transaction commit on dynamic and static paths.
5. Replaced `OgaEngineRun` and removed unseen-token delivery.
6. Updated C++, Python, examples, benchmarks, architecture documentation, and tests.

Do not temporarily maintain two token-delivery paths unless required solely to keep an intermediate
commit buildable; the completed change exposes events only.

## Acceptance criteria

### ABI and validation

- Finish reasons, event flags, and error codes are fixed-width typedefs.
- Event, usage, and Buffer layouts, sizes, offsets, alignments, versions, and strides are not
  public ABI.
- Null handles and null getter outputs are rejected safely.
- Direct count/get access is null- and bounds-safe.
- A Buffer from another live Engine is rejected before model progress.
- A Buffer remains safely destructible after its Engine is destroyed and is not passed to Run.
- Capacity zero validates the owner thread and otherwise does no work.
- `uint64_t` counts that do not fit internal types are rejected.
- Unsupported Turn setters return explicit not-implemented errors.

### Request and Turn lifecycle

- Request settings are snapshotted at creation.
- Turn IDs begin at one, zero means no Turn, and failed admission does not consume an ID.
- IDs increase across continuation Turns and exhaustion is rejected before mutation.
- Named stale cancellation cannot cancel a successor Turn.
- Repeated cancellation is idempotent.
- Cancellation preserves committed state and reports exactly one terminal event.

### Event delivery

- No event output returns count zero; partial prefill can do so while work remains.
- One token event is emitted for every visible generated token.
- A speculative decode transaction emits accepted draft tokens in order before its correction or
  bonus token; output-buffer overflow retains the remaining token events without executing again.
- A final visible token and completion are emitted together.
- Every token and terminal event carries the correct Turn ID.
- A multi-Request transaction returns all events in one call when capacity suffices.
- Capacity one executes once and drains retained overflow on later calls without more inference.
- Draining retained events never executes a new transaction, even when the output buffer has spare
  capacity.
- Static multi-row output follows the same bulk delivery contract.
- `HasPendingRequests` remains true while events are retained.
- Invalid Buffer use causes no model progress.
- Reusing a Buffer invalidates all prior event and usage views without allocating per Run.
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
- Contract failure emits one `TurnFinished | Failed/EngineContractFailure` event for every affected
  Turn and permanently prevents later execution.
- Fatal execution failure emits one `TurnFinished | Failed/EngineExecutionFailure` event for every
  affected Turn and permanently prevents later execution.
- A fatal failure with no affected Turn emits one request-less Engine failure event.
- API misuse returns `OgaResult` without a misleading operational event.

### Cross-surface behavior

- C and C++ compile smokes use the final declarations.
- Python unit and integration tests consume events only.
- Static and dynamic Engine tests cover token-only, terminal-only, and combined events.
- Examples and benchmarks contain no unseen-token calls.
- Windows and Linux/WSL builds cover the fixed-width ABI and wrappers.
