# Experimental Engine C API redesign

> **Status:** Implemented experimental contract.
>
> This specification describes the experimental Engine C API. It is independent of the classic
> Generator API: Engine Request creation does not accept `OgaGeneratorParams`, and the classic
> Generator surface is unchanged.

## Goals

The public ownership and execution hierarchy is:

```text
Model
  `-- Engine                 one shared runtime configured by model.config.engine
        `-- Request          one resident conversation/session
              `-- Turn       one queued, active, or completed generation invocation
```

The redesign must:

- keep ownership of configuration at the level that owns the state: model/Engine configuration for
  the runtime, `OgaRequestOptions` for resident-session policy, and `OgaTurnOptions` for one turn's
  generation policy;
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

Request creation takes no generation parameters. The Engine derives each Request's private search
configuration from its own model, forcing a single sequence and a single beam, so no caller-supplied
`GeneratorParams` field can reach the Engine and be silently ignored.

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
- Null options or zero `max_session_tokens` use the model-configured `search.max_length`, which
  normally defaults from the model context length.
- A nonzero `max_session_tokens` may not exceed that model-configured ceiling.
- This is the Request's one session limit: Search completion, cache sizing, speculative bounds, and
  the `MaxSessionTokens` finish reason all use it.
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

OgaResult* OgaTurnOptionsSetMinGeneratedTokens(
    OgaTurnOptions* options,
    uint64_t min_generated_tokens);

OgaResult* OgaTurnOptionsSetDoSample(
    OgaTurnOptions* options,
    bool do_sample);

OgaResult* OgaTurnOptionsSetTemperature(
    OgaTurnOptions* options,
    float temperature);

OgaResult* OgaTurnOptionsSetTopP(
    OgaTurnOptions* options,
    float top_p);

OgaResult* OgaTurnOptionsSetTopK(
    OgaTurnOptions* options,
    int32_t top_k);

OgaResult* OgaTurnOptionsSetRepetitionPenalty(
    OgaTurnOptions* options,
    float repetition_penalty);

OgaResult* OgaTurnOptionsSetNoRepeatNgramSize(
    OgaTurnOptions* options,
    int32_t no_repeat_ngram_size);

OgaResult* OgaTurnOptionsSetSeed(
    OgaTurnOptions* options,
    uint64_t seed);

OgaResult* OgaTurnOptionsClearSeed(OgaTurnOptions* options);

OgaResult* OgaTurnOptionsSetStopStrings(
    OgaTurnOptions* options,
    const OgaStringArray* stop_strings);

OgaResult* OgaTurnOptionsSetGuidance(
    OgaTurnOptions* options,
    const char* guidance_type,
    const char* guidance_data);

OgaResult* OgaTurnOptionsClearGuidance(OgaTurnOptions* options);

OgaResult* OgaTurnOptionsReset(OgaTurnOptions* options);
```

Every option is unset by default, and an unset option means "use the model-configured default for
this turn" -- never "keep what the previous turn used". Policy is resolved anew for every
`OgaRequestBeginTurn`:

| Setting | Unset behavior |
| --- | --- |
| `do_sample`, `temperature`, `top_p`, `top_k` | Model `search` defaults |
| `repetition_penalty`, `no_repeat_ngram_size` | Model `search` defaults, subject to scoring-device capability |
| `min_generated_tokens` | Zero; the model's session-absolute `search.min_length` is never reinterpreted as a per-turn value |
| `max_generated_tokens` | Unlimited except for the Request's session limit |
| `seed` | Continue the Request's existing host and device random streams |
| stop strings | Disabled |
| guidance | Disabled |

Rules:

- Zero unsets `max_generated_tokens` and `min_generated_tokens`: a turn cannot generate zero tokens,
  and a zero floor is the same thing as no floor.
- Zero `no_repeat_ngram_size` does *not* unset the option. It explicitly disables n-gram blocking
  for the turn, which is also what an unset value resolves to whenever the model's
  `search.no_repeat_ngram_size` is zero. On a model that configures a nonzero size, setting zero is
  how a turn opts out of it.
- Zero is a valid deterministic *seed*, so `OgaTurnOptionsClearSeed` is the only way to remove a
  pending reseed; it means "continue the existing stream", not "randomize again".
- `OgaTurnOptionsReset` is the whole-object unset mechanism: it restores every option, including
  `no_repeat_ngram_size`, to unset. There are deliberately no per-scalar clear entry points beyond
  `OgaTurnOptionsClearSeed` and `OgaTurnOptionsClearGuidance`, which exist only because zero and the
  empty grammar are meaningful values rather than "unset".
- `min_generated_tokens` masks the end-of-sequence token until the turn has generated that many
  tokens. It does not prevent stop strings, turn or session limits, cancellation, failure, or
  guidance termination. An extendable accepting grammar continues under the minimum, but once
  guidance permits EOS and no continuation token, its termination takes precedence rather than
  leaving the turn with no legal token.
- Admission rejects an explicitly set distribution scalar that contradicts a resolved greedy policy,
  so a caller never believes a turn sampled when it selected the top logit. A `temperature` other
  than 0 or 1, a nucleus `top_p` strictly between 0 and 1, and a `top_k` above 1 all contradict
  greedy selection. Values that request greedy selection themselves (`temperature == 0`,
  `top_k == 1`) or restrict nothing (`top_k == 0`, `top_p` of 0 or 1, `temperature` of 1) are
  accepted in any combination, so `do_sample = false` together with `top_k = 1` is valid while
  `do_sample = false` together with `temperature = 0.7` is not.
- Admission also rejects an explicit `do_sample = true` that a model search default silently
  overrides, because the model supplies `top_k == 1` or `temperature == 0`. The caller cannot see
  those defaults through the options object, so the error names the model-supplied cause and the
  Turn must override that exact field -- a `top_k` above 1, or a nonzero `temperature` -- to sample.
  A Turn that spells greedy out itself (`top_k = 1` or `temperature = 0` set on the Turn) has chosen
  it and is accepted alongside `do_sample = true`.
- A sampled Turn requires a positive `top_k` or a positive `top_p`. Both zero is rejected: it
  selects from nothing, and it is not the same request as greedy selection.
- A sampled Turn rejects a resolved `top_k` greater than the model vocabulary size. If that value
  came from the model's `search.top_k`, the error names the model default so the Turn can override
  it explicitly.
- `no_repeat_ngram_size` is rejected at admission on a scoring device whose search cannot apply it,
  rather than failing after the model has already run.
- Static batching completes generation on a non-transactional path, so it rejects stop strings and a
  per-turn seed at admission, before the Request is mutated.
- Dynamic batching also rejects a per-turn seed when the active batched sampler cannot checkpoint
  and restore its device RNG state.
- The complete resolved policy is validated before any Request or Engine mutation, so a rejected
  turn leaves the previous completed turn entirely reusable.

Unsupported requested behavior must never be accepted and ignored. `OgaTurnOptions` may be reused
and reapplies its configured fields to every turn it is passed to; `OgaTurnOptionsReset` restores
every option to unset. `OgaRequestBeginTurn` snapshots all values before returning. An options object
is bound to the Request that created it; passing it to another Request is rejected. The options
object does not keep the Request alive. Using it after the bound Request is closed or destroyed
returns an error.

### Seeds

A turn seed reseeds both the Request's host random stream and its device sampler state at the start
of the turn. The seed is full-width `uint64_t`; a value below 2^32 reproduces exactly what the
classic `search.random_seed` produced for the same value.

Every Request has a durable seed basis that its streams start from and that only a committed reseed
advances. A Request created from a model that configures `search.random_seed` uses that value; a
Request created from a model that leaves it unset (the default) draws a generated 64-bit basis once,
at creation, so an unseeded Request is not reproducible across processes while its own later turns
still continue one stream.

A greedy step consumes neither the Request's host random stream nor its persistent device sampler
state. Adding or lengthening a greedy Turn therefore does not shift the random draws used by a later
unseeded sampled Turn.

The reseed is applied inside the step transaction, strictly after every checkpoint and strictly
before the first random consumer, and it becomes durable only when that sampling step commits.
A rolled-back step restores both streams and leaves the reseed pending, so the retry reseeds
identically. A turn that is canceled, fails, or otherwise ends before a sampling step commits
discards its pending reseed without changing the durable basis, leaving the Request on exactly the
stream position it had before the turn.

Determinism is scoped to the same model, package, provider, platform toolchain, effective turn
policy, scheduling path, and speculative draft path.

### Guidance

Guidance is strictly turn-scoped. `OgaTurnOptionsSetGuidance` supplies one grammar
(`json_schema`, `regex`, or `lark_grammar`) for the next admitted turn; `OgaTurnOptionsClearGuidance`
and an omitted grammar both mean an unguided turn. Guidance is never inherited from a previous turn
or implicitly enabled from model or Request state.

Both strings are copied immediately. The setter validates the request shape and guidance type.
Build support and the grammar itself are validated at turn admission, before the Request is mutated,
so an unsupported or invalid grammar leaves the previous completed turn reusable. Every terminal path --
completion, stop match, cancellation, failure, and close -- releases the turn's grammar cursor, and a
rolled-back step restores it.

A guided turn does not accept speculative drafts; the next unguided turn is draft-eligible again.
Guidance and stop strings can be enabled together.

### Stop strings

`OgaTurnOptionsSetStopStrings` copies the given decoded UTF-8 stop strings immediately; the array
may be reused or destroyed afterward without affecting the options. An empty array (zero entries)
is a valid configuration that clears/disables stop strings; this is distinct from a nonempty array
containing an empty string member, which is invalid. Every entry in a nonempty array must itself be
a nonempty, valid UTF-8 string; the whole configuration may contain at most 16 entries totaling at
most 16 KiB (the same bounds `StopStringMatcher` enforces). Duplicate entries are preserved as
distinct, independently indexed entries. `OgaStringArray` stores NUL-terminated C strings and
therefore cannot preserve bytes after an embedded NUL; callers must reject or escape such input
before adding it to the array.

Matching is exact: no normalization, trimming, or case folding, and only text generated by this
Engine Request during the active turn is considered (never prompt tokens, continuation input, or
earlier-turn history). Concretely, matching decodes through a fresh detokenizer stream whose
decoding context begins at the first generated token of the active turn: prompt and continuation
input tokens for that turn are never fed to it, not even solely to seed detokenizer state before the
first generated token. This is the exact generated-output decoding boundary a host must mirror if it
runs its own incremental decoder to hide stop text from published output: some tokenizers apply
context-sensitive spacing to the first decoded piece of a stream (for example a leading-space
convention that a mid-stream piece would not get), so decoding a host's own copy of the whole
prompt+generation as one continuous stream can produce different bytes for the first generated token
than decoding a stream that starts fresh at that token the way the Engine's matcher does. When
several stop strings could match, the earliest-ending match wins, then
the earliest-starting/longest match, then the lowest original index. The raw token whose decoded
bytes complete the match is retained and counted as generated output; bytes after the match in that
token remain in the Request's raw history (the Engine never trims or rewrites committed tokens).
Callers that hide stop text from published output must incrementally decode and hold back possible
stop prefixes, then trim through the configured string identified by the matched index. The Engine
does not expose a byte offset within the match-completing token.

For the same generated token, precedence among reasons that reach Request-level classification is
`StopString`, then `MaxGeneratedTokens` (turn limit), then `MaxSessionTokens` (context/session
limit): a token that would otherwise end the turn on the turn or context limit still reports
`StopString` when it also completes a match. `Eos` is different: the search layer decides whether a
sampled token is EOS, and never appends an EOS token to a single-sequence Request's history, before
Request-level stop-string classification ever sees it, so an EOS token's bytes never reach the
matcher. In practice this is not a gap -- GenAI tokenizers default to `skip_special_tokens=true` and
register EOS as a special added token, so a real EOS token always decodes to zero bytes and could
never independently complete a stop match anyway. An active stop-string configuration does not
disturb ordinary EOS termination.

Stop strings require an Engine configured for dynamic batching: `OgaRequestBeginTurn` rejects a
stop-enabled turn on a static-batching Engine before any Request mutation, because static batching
completes generation through a non-transactional path that cannot stage, roll back, or replay a
match. A stop-enabled turn drafts and verifies with speculative decoding exactly like any other
turn: `OgaRequestSetDraftTokens` and every automatic in-Engine drafter (e.g. MTP) accept and verify
drafts normally, observing target-accepted tokens through the same stop-string matcher the ordinary
one-token path uses, in exact committed order, truncating at the first completed match and
discarding every later draft.

Turn admission itself is transactional with respect to stop strings, symmetrically for both
directions: a `BeginTurn` that would enable, change, or clear the Request's stop-string
configuration builds that outcome completely before any mutation and installs it only once the
whole admission attempt (continuation append, scheduler admission, and the rest) succeeds. If any
of that fails, `OgaRequestBeginTurn` throws and the Request's stop-string state -- whatever it was
before the call, from a prior committed turn -- is exactly as it was; a corrected retry then behaves
like any ordinary `BeginTurn`. The first stop-enabled admission on an Engine lazily creates the
shared tokenizer used by its Request-local streams. Loading tokenizer assets and initializing ORT
Extensions happen synchronously on the Engine owner thread, so that first admission can take longer
and briefly delay other Requests already decoding on the same Engine. Later stop-enabled admissions
reuse the shared tokenizer.

A committed stop match is staged and observed transactionally, in the same Request transaction as
Search/guidance/cache state: it is not externally visible until the step commits, and if the step
rolls back (directly or via the Engine's queued restore-and-complete path), the incremental
tokenizer stream backing the match is recreated and every previously committed current-turn
generated token is replayed through it, since the underlying detokenizer stream cannot be cloned. A
replay failure during rollback is treated as a fatal consistency failure, matching every other
rollback failure in this document. One rollback therefore performs work linear in each affected
stop-enabled Request's committed current-turn token count. Repeated retryable aborts as a turn grows
can accumulate quadratic detokenization work on the Engine owner thread, multiplied across affected
stop-enabled Requests in the batch; hosts should monitor abort frequency and turn depth when
qualifying this feature for long-running workloads. `OgaEngineEventGetMatchedStopStringIndex` and
`Request::MatchedStopStringIndex()` are always -1 for a cancellation or fatal-failure terminal
outcome: a committed StopString match makes the Request `TurnComplete` in the same step that stages
its terminal event, and both cancellation and fatal-failure handling only ever force-terminate a
Request that is still executable (not yet `TurnComplete`), so neither can observe, let alone
overwrite, an already-committed match -- the index is simply never set to begin with on that path.

### Finish reasons and event flags

```c
typedef uint32_t OgaFinishReason;
#define OgaFinishReason_None ((OgaFinishReason)0)
#define OgaFinishReason_Eos ((OgaFinishReason)1)
#define OgaFinishReason_StopString ((OgaFinishReason)2)
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

`OgaFinishReason_StopString` is emitted when a Turn's decoded stop strings complete a match; see
"Stop strings" above. `OgaEngineEventGetMatchedStopStringIndex` reads the matched entry's index and
writes -1 for every other terminal event.

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
OgaResult* OgaEngineEventGetMatchedStopStringIndex(
    const OgaEngineEvent* event, int32_t* out);
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
| `TurnFinished` | `request`, `turn_id`, `finish_reason`, `usage`, and `matched_stop_string_index` (-1 unless `finish_reason` is `StopString`) |
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
performs the logical close and cleanup at its next owner-thread boundary. A resident static row
retains the runtime state needed by an executable peer until the shared physical batch is
recycled. This deferred-release behavior does not permit concurrent Request operations.

Destroying an Engine closes all bound Requests and purges events. Surviving Request handles remain
valid lightweight closed tombstones that the caller must still destroy. Teardown uses a no-throw
detach path that releases scheduler/cache ownership and Request runtime state without relying on the
already-expired weak Engine reference. A resident static batch is released as shared Engine storage.

An Event Buffer stores a weak Engine reference and does not keep its bound Engine alive. Run rejects
a Buffer created by another Engine. Both the Engine and Buffer handles must be live for every Run;
after Engine destruction, the Buffer remains owned by the caller and must still be destroyed without
being passed to Run. Buffer creation and Run honor the Engine owner thread. Buffer access, Run, and
destruction are serialized; getters may read immutable views but must not race a Run or destruction.

Request creation takes no generation parameters. The Engine derives each Request's private search
configuration from its own model and forces the single-sequence invariants the Engine depends on:
`batch_size` and `num_beams` are one, the search length limit is the Request's `max_session_tokens`,
and guidance is off (guidance is per Turn, and fast-forward tokens are never enabled). Nothing about
sampling, guidance, or stop strings is fixed at creation.

Creation validates the model configuration it is about to derive from, before minting a Request:

- `search.max_length` must be greater than zero. It is the ceiling for `max_session_tokens`, which
  defaults to it and may be lower but never higher.
- `search.num_beams` must be one. Beam search is rejected rather than silently forced, because the
  Request would otherwise decode something the caller never asked for. `search.batch_size` is not
  rejected: the Engine batches Requests rather than rows, so it simply derives its own single-row
  search.
- `search.min_length` must be zero. It is a session-absolute floor, while the Engine's minimum is
  per Turn (`OgaTurnOptionsSetMinGeneratedTokens`).

The rejections name a route the caller can take without editing the model directory: overlay the
value on the `Config` before creating the Model, for example
`OgaConfigOverlay(config, "{\"search\":{\"num_beams\":1}}")`. Raising the session ceiling of a model
whose `search.max_length` is lower than its context length uses the same overlay route.

Per-Turn generation policy is validated separately, at each `OgaRequestBeginTurn`, before the Turn
mutates the Request. Creation itself does not queue work.

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
  int32_t matched_stop_string_index{-1};
  TurnUsage usage{};
  EngineErrorCode error_code{};
};
```

At Engine construction:

```cpp
const size_t max_step_events =
    cache_manager_->MaxBatchSize() * kMaxGeneratedTokensPerStep;
pending_events_.reserve(max_step_events);
staged_events_.reserve(max_step_events);
fatal_events_.reserve(max_step_events + 1);
```

`reserve` allocates capacity without constructing events. A committed step produces at most
`kMaxGeneratedTokensPerStep` events per affected Request, so normal retained output is bounded by
that limit times the scheduled batch size. Request creation grows dedicated fatal-event capacity
for that complete retained step plus every tracked Request because fatal handling may publish a
terminal event for executable Turns outside the failed batch. This keeps event publication
allocation-free after model/cache commit and after the Engine becomes unhealthy.

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
callers must close or abandon those handles explicitly. If reclamation detects an ownership
invariant failure, the Engine becomes unhealthy, terminal events are retained, and this operation
returns true so the host can drain them through `OgaEngineRun`. A transient allocation failure
returns an `OgaResult`, re-arms reclamation, and leaves any completed cleanup intact; the next
owner-thread boundary safely retries the remaining work.

Request creation and `OgaRequestBeginTurn` are also abandonment-reclamation boundaries. If
reclamation detects an invariant failure there, that operation returns an `OgaResult` while the
Engine retains terminal events for the next positive-capacity `OgaEngineRun`.

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
per Request; the closed/abandoned tombstone is no longer sampled or returned, but its row, shared
cache allocation, and row-essential runtime state may remain until the whole batch recycles.

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

At successful step commit, the Engine already knows the Request, Turn ID, selected tokens, terminal
state, finish reason, and usage. It captures those values in ordered `PendingEngineEvent` objects.
A speculative transaction emits accepted drafts followed by its correction or bonus token, up to
`kMaxGeneratedTokensPerStep` events per affected Request. Only the final event carries terminal
state when the Turn finishes. `OgaEngineRun` moves the available FIFO prefix into the reusable
Buffer storage and retains overflow.

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

## Language surfaces

### C++

- RAII `OgaRequestOptions` and `OgaTurnOptions`, the latter created from its Request.
- `OgaEngine::CreateRequest(const OgaRequestOptions* = nullptr)`.
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
- `Engine.create_request(*, options=None) -> Request`. `options` is keyword-only, so an older
  positional `create_request(params)` call fails loudly instead of binding generation parameters the
  Engine no longer accepts.
- `Request.begin_turn(tokens, turn_options=None) -> int`.
- `Request.cancel_turn(turn_id) -> bool`.
- `TurnOptions` mirrors the C setters: `set_max_generated_tokens`, `set_min_generated_tokens`,
  `set_do_sample`, `set_temperature`, `set_top_p`, `set_top_k`, `set_repetition_penalty`,
  `set_no_repeat_ngram_size`, `set_seed`, `clear_seed`, `set_stop_strings`, `set_guidance`,
  `clear_guidance`, and `reset`.
- `TurnOptions.set_stop_strings(strings)` converts a Python list of `str` to a temporary
  `OgaStringArray` and rejects (raises `ValueError`) any entry containing an embedded NUL byte
  before conversion, since the C string-array surface cannot represent bytes after one. An empty
  list clears/disables stop strings.
- `Engine.create_event_buffer(capacity) -> EngineEventBuffer`.
- `Engine.run(buffer) -> EngineEventBuffer`; the returned object is the same reusable Buffer and is
  a Python sequence over its populated borrowed event views. The binding releases the GIL for the
  complete native Run and reacquires it before returning or translating an exception.
- A zero-capacity Buffer is the capacity-zero no-op. Negative or unrepresentable capacities are
  rejected during Buffer creation.
- `EngineEvent` exposes flags, borrowed Request, Turn ID, optional token, finish reason,
  `matched_stop_string_index` (`None` unless the finish reason is `STOP_STRING`), error code, and
  borrowed usage. Indexed event views keep their Buffer object alive. Both event and usage data
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

- Configure the session limit through `OgaRequestOptions` and generation policy through
  `OgaTurnOptions`; Request creation takes no generation parameters.
- Replace ready-Request lookup and FIFO draining with event handling.
- Preserve application-owned maps when additional metadata is needed.

## Implemented phases

1. Added opaque Engine option, event, usage, and reusable Buffer handles; kept finish reasons,
   event flags, and error codes fixed-width.
2. Implemented Turn option creation, per-turn policy snapshotting, nonzero Turn IDs, and named
   cancellation.
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
- Every declared Turn setter is wired end to end; none returns success before the Engine honors it.

### Request and Turn lifecycle

- Request settings are snapshotted at creation, and a model configuration the Engine cannot honor --
  a nonzero `search.min_length`, a `search.num_beams` other than one, or a nonpositive
  `search.max_length` -- is rejected there rather than forced.
- Turn policy is resolved anew for every Turn from model defaults plus that Turn's explicit
  overrides, and is validated before any Request mutation.
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
- StopString reports the matched entry's index; every other reason reports -1.

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
