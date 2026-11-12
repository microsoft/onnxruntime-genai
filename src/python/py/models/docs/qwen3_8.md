# Qwen3.8 Model Builder Design

Qwen3.8 builders live in the [`builders/qwen3_8/`](../builders/qwen3_8/) package. They inherit from the base `Model` infrastructure for graph initialization, node/value creation, initializers, quantization, and serialization.

## Components

- `model.py` coordinates the package and exports the standalone Engram component.
- `text.py` builds the decoder, sparse attention, hidden-state-dependent Engram processing, and hyper connections.
- `mtp.py` builds the one-layer MTP head with the same text-building infrastructure.
- `graph.py` supplies the shared companion builder.
- `embedding.py`, `engram.py`, and `vision.py` build their corresponding components.

Component loaders remain in `loaders/`. Shared standard-ONNX decompositions are registered from `builders/expansions/standard.py`; model-specific graph emission remains in the owning builder.

## Index Sharing

Paged MTP graphs declare index-sharing inputs and outputs during initialization. Sparse attention emits the projection-row Gather, index merge, packed indexer, and status output directly. Index sharing is not a separate graph or a conversion pass over a serialized model.

The existing `max_draft_tokens` setting controls the exported draft budget, bounded by the kernel's supported capacity. Main and MTP graphs retain their ordinary cache and state contracts.

## Embeddings and Storage

`Qwen4ExpEngramModel` owns all token hashing and embedding-table lookup for non-paged and paged execution. The decoder always receives `engram_embeddings`; it contains no hash operator, quantized table lookup, or Engram embedding table. The former `Qwen38Engram` decoder helper is removed.

The decoder retains projections, hidden-state-dependent gating, padding masking, convolution, and residual addition because those execute inside the decoder layer. Its small token-history bindings preserve the runtime state interface, and standard-ONNX gathers capture accepted-prefix token history for speculative rollback without duplicating hashing or lookup.

Non-paged decoder token history is an unchanged input/output value using the same binding name, not an `Identity` node. Paged decoding continues to forward the standalone component's `present` history directly. Named hidden-state and companion outputs use their actual producer values rather than added pass-through nodes.

The same standalone lookup pipeline serves both decoder layouts: `NGramHashMapping`, `GatherBlockQuantized`, an optional dtype `Cast`, and the final output `Reshape`. Token IDs and history are rank two, and embeddings are rank three. The Engine invokes this component per request and packs its output rows for the paged decoder; no dense-to-varlen shape adaptation is needed inside either graph. Each exported component owns one external-data file, and only the standalone Engram graph stores the Engram lookup table.

## Configuration and Layout

Operator settings belong in the existing sparse-indexer, PLE, hyper-connection, attention, and quantization dictionaries. Weight packing is selected explicitly for the deployment target, not inferred from the export host. Integer QMoE, NVFP4, and FP8 have distinct layout contracts; see [packing options](../README.md#qmoe-weights-prepacked) and [quantization design](../quantization/QUANTIZATION.md).

For model-specific command-line examples, see [Model-specific Export Options](model-specific-options.md). General architecture and naming guidance remain in [DESIGN.md](../DESIGN.md).

## Sampled DFlash2 proposals

On Qwen3.8-27B with temperature 1.0, top-k 20 and top-p 0.95 (MMLU-Pro, 800 prompts, batch 1),
`lattice` averaged 5.01 tokens per target step against 4.78 for `greedy_path` and 4.66 for
`independent`, and decoded 3.1x faster than `none`.

These measurements motivate the default `sampled_proposal` policy. See the [Engine guide](../../../../../docs/paged_attention_engine.md) for the general proposal and verification contracts.

## Static backend that selects its own tokens

The `qwen38_flash_graph` model type (Qwen3.8-Flash CUDA-graph decode with a built-in multi-token-prediction loop: an MTP head
proposes up to three draft tokens per cycle, a DFlash2 drafter up to seven) runs on the static path through a `ModelExecutor`
that chooses its own tokens, instead of attaching decoder state with logits. Its state is fixed in-place KV, a
recurrent-state bank ring and a device-side draft -> verify -> commit loop that cannot be paged or cropped to an arbitrary
token. The device holds `model.qwen38_graph.max_batch_size` slots (default 1), one resident Request per slot.
`CreateEngineBackend` (`src/engine/backend_factory.h`) and `CreateQwen38GraphDependencies`
(`src/engine/qwen38_graph_executor.h`) assemble it:

- Single-slot native prefix reuse is opt-in: set `engine.dynamic_batching.prefix_caching` to `true` and
  `engine.dynamic_batching.max_batch_size` to `1` (the dynamic-batching default is 16).
  Set `model.qwen38_graph.prefill.chunk` (or `engine.dynamic_batching.max_scheduled_tokens`) explicitly:
  if both `prefill.chunk` and `search.chunk_size` are unset, creating this section supplies its default
  2,048-token scheduling budget as the prefill chunk, replacing the separate dense-prefill default of 8,192.
  Keep the effective chunk identical in cache-off/on cold-TTFT comparisons.
  Within one Engine, it restores committed prompt checkpoints across turns and separate Requests, including after `Close()`;
  it does not restore speculative decode state. Exact hits reuse the saved next-token prediction without an empty
  prefill. Partial hits ingest only the remaining suffix. `cached_prompt_tokens` counts reused tokens from the current
  turn's input, not retained earlier-turn history.
  Separate dense-prefill checkpoints save the dense graph's populated state. They omit the idle, zero-initialized
  decode KV/index buffers and recurrent banks: after restoration, `FastHandoff()` rebuilds those packed buffers from
  the dense state before decoding. Unified prefill populates the packed buffers directly and must checkpoint them.
  The existing `max_snapshots` and `snapshot_cache` VRAM/RAM/disk limits bound checkpoint storage; without explicit
  tier budgets or a count, at most one device-resident checkpoint is retained. `max_snapshots: 0` disables publication.
  Native `device_memory_budget_bytes`, `device_memory_headroom_bytes`, and `gpu_utilization_factor`
  follow the [native memory-budget rules](../../../../../docs/paged_attention_engine.md#native-bolt-prefix-caching):
  configure a byte budget before model loading with explicit headroom. Fixed weights, native buffers,
  and warmed graphs are charged first; the remainder caps VRAM checkpoints without resizing the target's
  fixed layout. Without a byte budget, the utilization factor limits remaining snapshot VRAM instead.
  RAM/disk snapshot limits remain independent. Budget validation also runs with prefix caching disabled.
  With warning logging enabled, an explicit opt-in that is inactive because of multiple slots, zero snapshot count,
  or all-zero tier budgets emits a warning.
  DFlash checkpoints must retain enough target features to reconstruct its complete context window (or start at
  position zero). A short final prefill chunk can therefore reuse an earlier eligible checkpoint instead of the
  entire prompt. Intermediate captures are skipped except for the last eligible DFlash checkpoint when the final
  chunk cannot reconstruct the drafter. Several-slot configurations retain replay-only behavior.
  Reuse beyond the first chunk requires an effective `prefill.chunk >= sliding_window + block_size`
  for the DFlash drafter; unified prefill otherwise defaults to 2,048 tokens. For a 2,048-token sliding
  window, that default is too small, so a repeated long prompt reuses only the first chunk.
  Unified prefill caps chunks at 8,192 tokens (4,096 when KV capacity exceeds 65,536); setting a larger
  configured value cannot bypass that cap. With warning logging enabled, Engine initialization warns
  once when the effective chunk is smaller than the drafter context window.
  The H100 prefix-reuse checks reported in #2937 used an explicit 8,192-token prefill chunk, including
  the 8,192- and 10,450-token checkpoint results; those are not results for the default chunk.
- MoE offload requires an explicit `session.moe.offload.mode` of `required`, `auto`, `1` or `on` (case-insensitive).
  A budget alone does not enable offload, matching ORT's disabled default; `disabled`, `0` and `off` preserve that behavior.
  Only enabled offload changes target-graph selection, synthetic warmup options or session budgets.
  By default, offload uses a fixed-layout estimate for the Flash-Next NVFP4 package. `ORT_QWEN38_ADAPTIVE_CTX_BUDGET`
  keeps it on when unset, empty, `1` or `true`, turns it off when `0` or `false`, and makes model creation fail on any
  other value. The estimate assumes 12 attention layers,
  two KV heads of 256 channels, one-byte K/V caches, int64 position IDs, and a 128-float index-key row per four positions
  per layer. It charges 16 KiB per position per request slot, relative to one slot of 33032 positions. These are not bytes
  measured from the selected graph, and do not account for every state tensor, workspace or allocator overhead.
  The filename tier estimates include the package's padding:

  | Filename component | Estimated KV positions |
  | --- | ---: |
  | `_2k` | 2312 |
  | `_8k` | 8456 |
  | `_16k` | 16648 |
  | `_32k` | 33032 |
  | `_64k` | 65800 |
  | `_128k` | 133128 |

  Other names, including `model_dynamic_ctx.onnx`, use the configured `model.qwen38_graph.kv_capacity` for the estimate.
  The loaded graph remains authoritative for execution capacity. With model logging enabled, adaptive offload reports
  a capacity mismatch, but cannot resize budgets on an already-created session. Different cache layouts/dtypes or padding
  require disabling adaptive budgets and configuring suitable explicit budgets; this estimate does not guarantee that a
  model fits in device memory.
- The slot count is an integer in 1..8 (`Config::Model::Qwen38Graph::kMaxBatchSize`): the JSON parser rejects any other
  `max_batch_size` and names the key and the value, and the other settings that can choose the count
  (`engine.dynamic_batching.max_batch_size`, `search.batch_size`) are held to the same range at model creation instead
  of being clamped silently. `engine.static_batching` never chooses the count: it configures the Engine's static path in
  general, which this backend has always ignored, so a package that carries it (with the Engine's default of 4 or a
  value of its own) keeps its single slot and keeps loading on a stack that cannot serve several. Only a setting that
  asks for slots opts in. Several slots are served by the batched fold target and only
  work with: the folded MTP stack (`fold.enabled`), drafted by the DFlash2 drafter (`model.dflash2.filename`) when the stack has an
  internal head (`fold.head_r2.filename`), whose state and drafting are not slot-addressed; the unified fast prefill, i.e. `prefill.enabled` without a separate dense
  graph (the prompts are ingested chunk by chunk through the fold target, slot by slot; a configuration with
  `prefill.enabled` false or with a dense prefill graph is single-slot); a compact-state target (the exporter's
  `state_update.*` outputs; a ring-bank target with `present_mid*` outputs keeps one bank per row count for the whole batch and
  is single-slot); and a plane library with the `batched_prep` capability (`qwen38_plane_prep_batched`, which takes the row
  count and the slot count as arguments) and the `ctx_slot_stride` capability (slot `s` of its `ctx_out` is at word `16 * s`,
  see the next item). The model checks the first two before the plane library is loaded and any weight is
  uploaded, the plane capabilities when the library has been loaded and before any weight is uploaded, and the compact-state
  property on the output names of the target graph, which it reads from the model file (`ReadQwen38OnnxOutputNames`: the
  protobuf framing of `ModelProto.graph` and `GraphProto.output`, skipping every initializer) before any weight is uploaded
  too. A model file the reader cannot parse (an ORT-format model, say) is left to the same check on the output names of the
  target session, which runs as soon as that session exists, before the other sessions and any device state are created.
  Each failure is a `std::runtime_error` that names the setting. A plane library without `batched_prep` or `ctx_slot_stride`
  still loads packages with one slot.
- The fold stack reads the n-gram contexts of the verify rows of every slot from one buffer (`ctx_a`) that
  `com.microsoft::SpeculativePrefixCommit` takes as its `prefixes` input. The op divides the size of the input by the slot count
  and reads slot `s` at `s` times that many words, and the engine binds `batch * 16` words (2 words for each of the 8 rows of the
  largest verify step), so every writer of the buffer puts slot `s` at word `16 * s` (`QWEN38_PLANE_CTX_WORDS_PER_SLOT`)
  whatever the number of verify rows (3 to 8): the resident and the host-staged paths of `qwen38_plane_prep_batched` do, and the
  words of a slot past its `2 * rows` are left untouched. The one-slot prep writes slot 0 only. A plane library from before
  this layout wrote slot `s` at word `2 * rows * s` (the same offset only for 8 rows) and does not report `ctx_slot_stride`:
  the model refuses it for several slots, because the n-gram context a slot commits would otherwise be read from the wrong
  words with no error.
- The fold model takes exactly the four graph inputs of the `SpeculativePrefixCommit` contract of ORT #33474: `fold.ctl`,
  `fold.ctx_commit`, `fold.ctx_a` and `fold.ptrs`. The engine owns a device buffer for each of them and for nothing else, so
  when it builds the target it refuses a model with any other `fold.*` input, such as the `fold.ctx_b` of the node of an earlier
  export, with a `std::runtime_error` that names the input; export the model package again for the contract.
- The compact `state_update` replay corrects the recurrent, convolution and indexer state of a slot after a partial
  acceptance. The verify step reads the state bank `k` and writes the state it ends in, after all of its rows, to the other
  bank `k ^ 1` (the two banks alternate from one cycle to the next); the replay recomputes the state of the rows that were
  accepted from bank `k`, which the step left untouched, and the updates the step captured, and writes it over the state in
  bank `k ^ 1`, so it reads and writes different buffers. A `CausalConvWithState` state is replayed in either physical layout,
  `[batch, channels, history]` (what the exporter writes) or `[batch, history, channels]`: `ValidateQwen38ReplayDescriptor`
  takes the channel count from the last dimension of the `state_update` value, finds the layout from the dimension of the
  state that has it, and
  `MakeQwen38ConvReplayDescriptor` passes it to the plane in `Qwen38StateReplayDesc::layout`. A state with as many steps as
  channels cannot be told apart and is taken as channel-first. Only a package with a history-first conv state needs a plane
  library with the `conv_history_first_replay` capability; the model checks it while it builds the target, before any request,
  and the error names the capability, the state and the library. A channel-first package keeps loading with a library without it.
- `FixedSlotCacheManager` (`Qwen38GraphCacheManager`, `src/engine/fixed_slot_scheduler.h`) holds one row per slot (`SlotOf`,
  `ResidentRequestCount`). `FixedSlotScheduler` queues further Requests and replaces a slot as soon as its turn is over, so a
  short prompt or answer does not wait for a long one, and `CacheManager::SupportsContinuousMultiSlotDecoding()` lets a
  resident Request continue its turn while others are resident. `RewindToStartOfTurn` releases the slot; a continued turn
  (`BeginTurn` on the resident Request) and the replay after a rewind re-ingest retained history (through the same
  chunked prefill as a new prompt), because the look-ahead cycle and the bank ring cannot be rolled back.
  With single-slot native prefix reuse enabled, a matching committed prompt checkpoint can skip the cached prefix.
- `Qwen38GraphExecutor::SelectsOwnTokens()` is true. One Engine step is one prompt chunk (the size the device asks for,
  `Qwen38DecodeDevice::PrefillChunkTokens()`: `model.qwen38_graph.prefill.chunk` when set, otherwise 8192 tokens with the
  dense prefill (`prefill.enabled`) and 2048 per step through the decode graphs, see `EffectiveQwen38PrefillChunkTokens`; the
  device clamps it to the capacity of the loaded graph, which a context tier can leave below the configured chunk, so that
  case is not a configuration error;
  the decode-graph path needs an even chunk; a step that only advances the prompt delivers no event, so `CancelRequest`
  stays responsive; the dense prefill state continues across the chunks and is handed over to the decode state in the step that
  ingests the last one) or one whole cycle (with `fold.enabled`: one CUDA-graph launch per contiguous active-slot interval).
  The executor tells the device the
  prompt length of the turn (`Qwen38DecodeDevice::ExpectPrefillTokens`, after `Reset()` and before the first chunk; with
  several slots `ExpectPrefillTokensSlot(slot, total)`, after `ResetSlot(slot)` and before the first `PrefillChunkSlot` of
  that slot); the device hands it to the ORT QMoE kernels with every chunk of that turn and drops it at
  `Reset()`/`ResetSlot()`/`BeginDecode()`/`BeginDecodeSlot()`. The value is state of the device and, with several slots, of
  the slot: nothing is published through the process environment by the executor, so Engines of different models and slots
  that ingest their prompts in interleaved chunks never see each other's value. The device itself publishes the QMoE hints
  (`ORT_QMOE_PREFILL_NEW_TURN`, `ORT_QMOE_PREFILL_TOTAL_TOKENS`, `ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS`,
  `ORT_QMOE_PREFILL_HANDOFF_DONE`, `ORT_QMOE_IN_WARMUP`), only for the duration of one ORT Run (`Qwen38OrtEnvScope`, one process-wide lock,
  the previous values are restored): the values of a prompt chunk are computed from that chunk's slot alone, never kept or
  max-merged across Runs, and a Run publishes the new-turn and hand-off notices of the slots it serves only (a prompt chunk
  serves its slot, a batched decode the slots that decode in it).
  Every Run also carries `ep.cuda.qmoe_in_warmup=1` during synthetic warmup and `0` afterwards; current ORT reads that entry
  rather than `ORT_QMOE_IN_WARMUP`. Reused run options are updated only when the value changes, including the reset to `0`
  before live runs. With warmup enabled (the default), the first offloaded target warmup for each direction uses its normal
  `gpu_graph_id` and captures while the device is muted. Subsequent synthetic warmups use `gpu_graph_id=-1`, including
  batched shapes: captured replays bypass the per-run control and would otherwise score and promote dummy-token experts.
  Captured graphs are provider- and thread-owned. Acquiring the device on another thread (a model loaded on one thread and
  driven by an Engine on another) therefore recreates the capturing sessions and the decode state on the acquiring thread and
  repeats the muted warmup there before any live request; a thread that reuses an exited thread's operating-system ID still
  counts as a new thread. Engines on the owning thread reuse its captures. A failed recreation discards the partial state and
  releases the lease, so the next acquisition retries.
  With several slots a step advances every slot (`Qwen38GraphSession::StepBatch`): it commits the in-flight batched cycle of
  the decoding slots, prefills the next chunk of the slots that have not started decoding and enqueues the next batched
  cycle. The executor declares its step plan via `ScheduledRequests::BeginBackendStep` and stages owned `BackendRowResult`
  records (`PrefillProgress` and/or `SelectedTokenBlock`, up to `kMaxGeneratedTokensPerStep` candidate tokens) via
  `ScheduledRequests::StageBackendRow`; `GenerateNextTokens` validates the whole batch before mutating any Request, commits
  candidates through `Request::CommitExternalTokens` (observing stop strings, EOS, turn limit, and context limit in order),
  and records `TokenCommitAck` counts (consumed, appended, and staged for events) settled through `TokenProducingSession`
  before events are published. Stop strings are therefore accepted on this static path; every other static executor still
  rejects them.
- The next cycle is enqueued before a step returns, so the GPU runs it while the Engine delivers the previous tokens. A stop
  string can only be known after delivery, so one cycle may be wasted; the next request and Engine teardown wait for it with
  `Qwen38DecodeDevice::Drain()`, and so does a rewind of a single-slot device. With several slots a rewind's `ReleaseSlot()` does
  the same when a cycle is unchecked: it drains the device and reads the status of every slot before the slot is returned, so a
  failing look-ahead fails the rewind before the Request is mutated (the failure is kept for retries and the next step). The
  replacement of a slot (`ReleaseSlotForTeardown()`: a cancellation, a new Request taking over a finished turn) does not wait;
  the new prompt waits for the device in `BeginSlot()`, before it resets the slot. The record of such a
  cycle is never consumed, so its status is read from the device instead: `CheckDecodeStatus()` right after `Drain()` reports
  the error bits accumulated by every cycle since the last
  `Reset()`, also when a record was left unconsumed. The executor always runs `Drain()` and then `CheckDecodeStatus()` before
  `Reset()` (which clears the bits): a cancellation or stop-string match that leaves a cycle behind therefore still fails the
  next request or the rewind when that cycle went wrong. The same holds for `ResetSlot()` with several slots: the bits of a
  batched cycle are kept per slot, `ResetSlot()` clears the ones of its slot, and `CheckDecodeStatus()` reads every slot that
  started a decode (slot 0 or not) and names the slots with bits set. A slot handed to a new request, whose previous request
  was canceled or finished while a batched cycle was in flight, is therefore only reset once the stream was drained and the
  status of all the slots was read (a status that was already read since the last cycle, as at the end of a chain of cycles,
  is not read again). A device error is fatal to the Engine, as every executor failure is, so a reported status is never
  consumed twice, and a fatal execution failure poisons the mutable device context across Engine leases until the model is
  reloaded. At Engine teardown the executor drains the device and reads its status too, but nobody is left to report an error
  to, so a failure poisons the device context instead (the next Engine over the model fails).
- Ready slots decode between prompt chunks, including when a new long prompt joins resident decoders. Phase 1 defers the
  next cycle only until phase 2 has ingested one chunk per prefilling slot; phase 2 then enqueues all ready decoders.
  The device splits active slots into contiguous intervals and uses fixed-address bindings for each interval, including
  its offset KV, indexer, recurrent state, control block, context and result ring. A slot outside those intervals is never
  an idle row of a target Run, so neither its prompt KV nor its recurrent state is overwritten. Each prefilling slot keeps
  its own recurrent bank independent of the decode ring phase; its final hand-off copies that slot's recurrent state to the
  current decode bank before it joins the ready slots. Target graph ids include the first slot as well as batch size and
  bank phase, and warm-up covers those offset bindings. Contiguous ready slots still share one target forward; separated
  intervals run separately, without waiting for the prompts between them. Auxiliary features are bound directly to persistent
  per-slot slices, so both shared and deferred DFlash proposals consume the right slot's features. Each interval binding owns
  its validation aggregate, and each slot retains the direction of its in-flight target Run so completion errors report that
  Run's status codes even after later intervals execute.
- With several slots the DFlash2 drafter proposes for one slot after the other without waiting for the device in between.
  A step tensor larger than a launch can carry (4032 bytes: the block table of a long context) is uploaded from its host
  mirror when the device reaches the copy, so one shared mirror per tensor would be rewritten by the next slot's proposal
  before the previous upload ran, and the first slot would read another request's block ids. The engine therefore asks the
  drafter for a fresh mirror per upload (`Dflash2Drafter::SetFreshStagingMirrors`, `Dflash2HostStaging`): the pinned-mirror
  pool hands a mirror out again only after the upload that read it has run, and never waits for the device to do so. A
  single-slot engine keeps one mirror per tensor, as before.
- `ModelExecutor::ValidateTurn` and `BackendCapabilities` reject, before any Request mutation, what the device loop cannot do:
  sampling, repetition penalty, no-repeat n-grams, `min_generated_tokens`, seeds and guidance. `MaxDraftTokensPerStep()` is
  zero (the caller cannot attach drafts). `GetCapabilities()` reports the resolved slot count and `max_request_length` =
  KV capacity - reserved tail, clamped to `model.context_length` when that is configured, so it is the smaller of the two.
- `GetSpeculativeStats()` includes the executor's counters (`ModelExecutor::AddSpeculativeStats`): one round per slot and
  cycle, one or more proposed drafts per round (the fold verifies 1..3), accepted drafts, bonus and correction tokens, the
  acceptance-length histogram. Target forwards are counted per physical Run (one per contiguous ready-slot interval), not
  per slot. A shared batched DFlash proposal counts one draft forward across its surviving rows; independent startup and
  deferred-seed proposals still count separately. An explicit zero draft-forward count stays zero. The counters describe
  what the Engine consumed, as `Engine::RecordSpeculativeCommit` does for sampled MTP: a turn that ends inside a cycle (turn or
  session limit, stop string, EOS) only accepts the drafts it delivered,
  `draft_tokens_evaluated` counts the draft positions examined up to the terminal token (every accepted draft consumed, an EOS
  draft included, plus the rejected position only when the turn got as far as the correction token), so `acceptance_rate` is not
  diluted by drafts after the end of the turn. Rounds are classified like `Engine::RecordSpeculativeCommit` does, by the number
  of accepted drafts that were delivered: none is a zero-accept round, all of the proposed ones a full-accept round and anything
  in between a partial-accept round (`partial_accept_rounds`, counted whether or not a correction token was delivered).- The model's device state is leased to one Engine at a time (`Qwen38GraphModel::AcquireDevice`); a second Engine over the same
  model fails with a clear error until the first one is destroyed, and a poisoned context cannot be leased again. The model has
  no Generator API support.
