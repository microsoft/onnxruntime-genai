# Native Nemotron-3.5 Bolt / LiLiCorr

`nemotron_bolt` implements batch-one, single-CUDA-device greedy generation through
the ordinary Model / Generator API. It does **not** use the paged target Engine or
support dynamic batching. The implementation runs the fixed-R8 target and the
LiLiCorr DFlash2 drafter directly in C++ with ORT. No Python inference bridge or
synthetic logits are involved.

## Package configuration

Use `model.type: "nemotron_bolt"`, `model.vocab_size: 131072`,
`model.context_length: 135160`, and `model.decoder.filename: "model.onnx"` for a
fixed-R8 graph with KV capacity 135168. `context_length`
must reserve at least eight scratch positions below the graph's KV capacity.
`search.max_length` includes the prompt and must not exceed `context_length`.
Usual BOS/PAD/EOS fields apply, including an EOS list such as `[2, 11]`.

When deriving configuration from graph metadata, subtract eight from the reported
KV capacity in **both** `model.context_length` and an explicit `search.max_length`.
Omitting `search.max_length`, or setting it to zero in JSON, defaults it to the
model context length. The reserve is conservative fixed-block scratch space:
even a final partially delivered round executes all eight target rows.
It does not reduce the allocated KV tensor or the CPU `total_sequence_length`
input, which both retain the graph's full capacity of 135168.

Configure the LiLiCorr `model.dflash2` section with five layers,
block size 8, seven drafts, top-k 8, mask 990, sliding window 1024, and
`inputs.anchor_row_index: "anchor_row_index"`. Set its filename to the packaged
drafter ONNX path. For speculative decoding, configure **both** sessions with
the CUDA provider and `enable_cuda_graph: "1"`. The legacy plugin provider's
`min_num_runs_before_cuda_graph_capture: "2"` option is not supported by the
built-in CUDA provider; do not carry it into that configuration.
Set `ep.cuda.arena.use_cuda_mempool: "1"` in both session-options objects.
For padded drafter replay, use an ORT API 32+ build implementing
`session.paged_attention.skip_input_sanitizer` and explicitly set this session
config entry to `"1"` in the drafter session. Without that explicit compatible
policy, partial-row proposals remain unpadded and are not captured. Fixed-request
packing requires a windowed drafter; full-attention DSpark keeps its existing
non-fixed execution path.
Legacy custom-op graphs require their matching `custom_ops_library` or
`custom_ops_libraries` registrations. Graphs converted to the corresponding
`com.microsoft` contrib operators require an ORT build containing those kernels
instead. Remove custom-library registration only after converting and validating
both target and drafter graphs.
Provider-specific memory/GEMM settings from the
reference runtime may be retained. Decoder cache name mappings and
layer counts are not used: the native state validates the actual graph metadata.
The standard decoder input keys `past_sequence_length: "seqlens_k"` and
`total_sequence_length: "total_sequence_length"` are accepted, but do not
override these native bindings. No decoder `state_groups`, `sliding_window`, or
Engine settings are needed; decoder `sliding_window` and enabled static/dynamic
batching are rejected. The drafter's internal paged cache and sliding window do
not make the target paged or windowed.
Set `speculative.max_draft_tokens: 7` for speculative decoding; the verify width
is fixed, not adaptive. Ordinary greedy decoding accepts `max_draft_tokens: 0`
and executes fixed-R8 target blocks, committing only the first row. Configuring
the model itself with zero draft tokens skips loading the drafter; requesting
zero draft tokens from a model configured with a drafter leaves its weights
loaded. These are different memory-measurement configurations.
Target-only decoding permits capture to be disabled. Requesting capture requires
a capture-enabled target session; speculative decoding requires both a loaded
drafter and capture.

### Optional larger-chunk prefill

The default package remains fixed-R8, including prompt ingestion. To opt in to
a separate prefill graph, set `model.decoder.prefill_filename` to its packaged
filename. `model.decoder.prefill_sequence_length` may be omitted (zero), or must
match that graph's fixed input width `R`. `R` must be greater than eight,
divisible by eight, and no greater than `model.context_length`. This is a package
choice, not `search.chunk_size`, which remains unsupported.

The runtime consumes full `R`-token chunks with the prefill session, then uses
the existing R8 graph for the remainder. No large chunk is padded. Verification
and ordinary decoding remain R8, and the context-capacity reserve stays eight
positions. Both sessions use the decoder session options and declared shared
initializers. List all common large initializers in
`model.decoder.shared_initializers` to avoid loading a second device copy;
undeclared weights are **not** automatically deduplicated.

The prefill graph is an external exporter/operator dependency with this contract:

- Input and output **names** match the R8 target. Inputs other than
  `input_ids[1,R]` have the exact same types and shapes as that target, including
  the eight-row recurrent banks and full-capacity dense KV.
- `seqlens_k = logical_past + R - 1`; `state_row` selects the incoming recurrent
  row, just as for verification. The graph must produce causal KV for all `R`
  tokens and put its **terminal recurrent state in row zero**. Other recurrent
  rows are scratch. It must support the same in-place `past_*`/`present_*`
  aliases and read the selected incoming state before overwriting it.
- BF16 `logits[1,1,vocab_size]` and INT64 `pred_ids[1,1]` describe only the last
  input token. BF16 `aux_hidden_states[1,R,width]` contains every input row in
  order, with exactly the R8 target's auxiliary-layer selection.
- Prefill runs with `gpu_graph_id=-1`, independently of verification capture.
  Intermediate chunks avoid prediction readback. The final chunk selects the
  anchor, and the runtime retains only the drafter's trailing context window,
  even when its start falls inside a large chunk.

Changing the R8 constant or reshaping the existing model is **not** a compatible
export. In particular, the current packed projection contrib operators accept
at most eight flattened rows, while `Mamba2Verify` accepts at most 16 rows and
emits per-token state snapshots. These are not efficient 2048-token prefill
kernels. Larger graphs require compatible projection, recurrent, MoE and
attention kernels plus a validated exporter implementing the contract above.
Merely adding this configuration to the preserved benchmark packages does not
make them faster. This runtime opt-in makes no 32K/64K/128K/256K-context performance
claim and does not increase a package's KV capacity.

With a plugin CUDA EP, register its library **before** constructing the model:

```python
import onnxruntime_genai as og

og.register_execution_provider_library("CUDAExecutionProvider", "/runtime/libonnxruntime_providers_cuda.so")
model = og.Model("/models/bolt")
params = og.GeneratorParams(model)
params.set_search_options(do_sample=False, max_length=4096)
generator = og.Generator(model, params)
generator.append_tokens(prompt_ids)
while not generator.is_done():
    generator.generate_next_token()
tokens = generator.get_sequence(0)
stats = generator.get_speculative_stats()  # Python dictionary
```

For the built-in CUDA EP, omit the execution-provider-library registration.
Provider-library registration and custom-operator-library registration are
separate mechanisms.

The plugin, ORT runtime, GenAI core/CUDA add-on, and custom operators must be
mutually compatible. ORT API 27 or later is required to release captured graphs.
Point `ORT_LIB_PATH` at the compatible runtime library.
Set `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` before loading the runtime, as in the
reference environment; the cuDNN attention path is not capture-safe
for this fixed-R8 graph.
The build-local bare `onnxruntime_genai` extension can be imported directly by
putting `build/cuda/src/python` on `PYTHONPATH`; make the GenAI libraries and CUDA
dependencies available to the dynamic loader. This does not require installing
or replacing any global Python package. Keep the CUDA plugin registered for the
process lifetime with the supplied ORT build: explicit unregistration can
invalidate ORT's shared allocators and fail during runtime shutdown, even after
all models have been destroyed. Normal Python module cleanup tears down GenAI.

## External operator contract and provenance

The target exporter and `bolt` custom operators are external dependencies.
Tensor names, types, and shapes are necessary checks, not proof of compatibility.
Package labels such as "v9", "v10", or "qualified" and the filename
`libbolt_ops.so` are not immutable version pins.

The exporter and operators must satisfy all of the following:

- **Aliased state:** each `past_*` input and corresponding `present_*` output
  have exactly the same backing allocation. Operators must support this aliasing,
  not assume disjoint input/output storage. Allocations remain alive and at
  stable addresses throughout all executions and captured-graph replays.
- **Recurrent rows:** `state_row` selects the initial convolution and SSM state
  from the previous call, in the range 0 through 7. That selected state must be
  read before an in-place write can overwrite it. Output row `r` is the state
  after consuming input rows 0 through `r`. After accepting `m` drafts, the next
  call selects row `m` and advances the logical KV length by `m + 1`; rejected
  recurrent rows must never influence the committed continuation.
- **Dense KV and padding:** for logical past length `L`, a call processes eight
  tokens starting at `L`, passes device `seqlens_k = L + 7`, and passes the full
  allocation capacity as the CPU scalar `total_sequence_length`. Attention must
  remain causal for each row and ignore stale KV beyond the logical prefix.
  Later calls overwrite rejected positions. A padded final prompt chunk commits
  only its valid rows and selects its last valid recurrent row.
- **Capture and streams:** all operator work must follow the CUDA EP stream and
  be capture/replay safe. Captured work must read mutable input values, including
  `state_row`, on every replay rather than baking in their warmup values.
  The current runtime executes four dummy target runs and four dummy drafter
  proposals with capture delay two, discards their outputs, and synchronously
  resets caches before ingesting a real prompt. Changing adapters is unsupported.
  Graph resources must be releasable when a generator is destroyed without
  invalidating the model-owned sessions or another sequential generator.

Independent package qualification requires a published manifest containing:

- Immutable source URLs and full commit IDs for the exporter, custom operators,
  GenAI, ORT, and CUDA EP; any source patch hashes; and the export/build commands.
- SHA-256 hashes for the target graph and **all** external weight files, drafter
  graph/weights, configuration, tokenizer, custom-op library, runtime libraries,
  and GenAI binaries. Drafter weights and auxiliary-layer selection must match
  the target checkpoint.
- Independently generated prompt/expected-token fixtures, with the reference
  implementation revision and model/checkpoint hashes, rather than tokens
  generated by this runtime or its target-only verification helper.

This PR does not currently publish that manifest or independent token fixtures.
The configuration above describes the required interface, not a certified
exporter/library/model combination. Package compatibility remains unverified
until those artifacts are supplied.

## State and public-token semantics

By default, prompt ingestion uses the same fixed-R8 target, eight tokens per call.
An explicitly configured prefill graph handles full larger chunks as described
above. Intermediate chunks execute asynchronously without host readback; an R8 tail is padded
but only valid rows are committed, and its last valid row selects the recurrent
state and first generated anchor. The drafter then ingests the prompt's trailing
sliding-window suffix (`<= context_window`) in a single proposal pass after the
final prompt chunk. Target `past_*` and `present_*` share the same allocation. Verification consumes `[anchor, seven drafts]`,
advances dense KV length by `accepted + 1`, and selects recurrent row `accepted`.
Only accepted context reaches the drafter. Its packed row maps preserve the real
last-context `anchor_row_index`, even when padded to eight rows for graph replay.

Target and drafter capture/warm on dummy state, then reset before processing the
real prompt. Each generator owns independent buffers and graph annotations.
Drive generators sequentially; concurrent calls on a shared model are unsupported.
Speculative blocks are buffered internally but each `generate_next_token()`
exposes exactly one token. EOS and the total-length budget stop public delivery
without leaking lookahead. `get_speculative_stats()` reports actual forward
passes and acceptance counts; it does not estimate a baseline speedup.
Timing and estimated-speedup fields remain zero; use external wall-clock timing
for throughput rather than interpreting those fields as measurements.

Generator construction includes fixed-capacity allocation, target and drafter
warmup/capture, and cache resets. TTFT measured only after construction excludes
all of those costs. A performance report must record model loading, allocation,
target capture, drafter capture, reset, TTFT, and warm decode separately; capture
subphases require instrumentation inside construction, not subtraction of
unrelated wall-clock measurements. Publish the harness, exact prompt IDs, warmup
and EOS/budget rules, raw results, hardware/software versions, and before/after
runtime and custom-op revisions. Hold custom-op bytes fixed to attribute a gain
to the runtime alone; combined runtime/kernel changes are not an isolated
runtime speedup. No reproducible startup-cost or before/after performance
evidence is currently published with this PR.

Only reset-to-zero rewind is supported: `generator.rewind_to(0)` clears target
state, drafter context, and pending tokens. Continuing with a second
`append_tokens()` requires this reset or a new generator. Sampling, beams,
guidance, nontrivial penalties, external chunking/speculative strategies,
nonzero rewinds, extra inputs, and logits access/replacement are explicitly
unsupported. In particular `get_logits()` fails rather than fabricating scores.

## Focused native tests

`engine_unit_tests --gtest_filter='NemotronBolt.*'` runs CPU-only acceptance,
EOS, budget, search-option, public-token, and reset tests. To additionally check
sequential generators and resets with the real CUDA graphs, set
`ORTGENAI_BOLT_TEST_MODEL` to the package directory and run
`engine_unit_tests --ep_dir /runtime --gtest_filter='NemotronBoltGPU.*'`.
These opt-in GPU tests check repeatability and compare speculative decoding with
the same target implementation; neither comparison is an independent reference
oracle. Qualification additionally needs pinned expected-token checks after
reset and at every output-budget boundary, terminal EOS, every recurrent
acceptance row, and prompt lengths immediately below/at/above an eight-row chunk,
the drafter context window, and the supported capacity (with rejection above
capacity). Those independent real-graph checks remain outstanding.

For a package with the optional prefill graph, set
`ORTGENAI_BOLT_PREFILL_TEST_MODEL` and run
`engine_unit_tests --gtest_filter='NemotronBoltGPU.*Prefill*'`.
For a plugin CUDA EP, also pass `--ep_dir /runtime`; omit it for the built-in EP.
This compares the larger-chunk and original R8 paths **within each mode**,
including committed recurrent/KV bytes, selected logits, first proposals,
generated tokens, reset behavior and actual prefill call counts. It covers R8,
large-chunk and drafter-window boundaries, with and without verification capture
where supported. It is not an independent HF oracle or a throughput benchmark.

The original custom-op package's benchmark scores remain scoped to its pinned
runtime. Migration checks compare custom-op and contrib-op results in the same
decoding mode. A pre-existing ordinary-versus-speculative sequence difference
must be disclosed separately; passing migration checks does not establish
ordinary/speculative equivalence or final release readiness.
