# Native Nemotron-3.5 Bolt / LiLiCorr

`nemotron_bolt` implements batch-one, single-CUDA-device greedy generation through
the ordinary Model / Generator API. It does **not** use the paged target Engine or
support dynamic batching. The implementation runs the fixed-R8 target and the
LiLiCorr DFlash2 drafter directly in C++ with ORT. No Python inference bridge or
synthetic logits are involved.

## Package configuration

Use `model.type: "nemotron_bolt"`, `model.vocab_size: 131072`,
`model.context_length: 135160`, and `model.decoder.filename: "model.onnx"` for the
qualified v9 or compact v10 graph with fixed KV capacity 135168. `context_length`
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

Copy the validated LiLiCorr v3 `model.dflash2` section unchanged (five layers,
block size 8, seven drafts, top-k 8, mask 990, sliding window 1024, and
`inputs.anchor_row_index: "anchor_row_index"`). Set its filename to the packaged
drafter ONNX path. Configure **both** sessions with the CUDA provider and
`enable_cuda_graph: "1"` and `min_num_runs_before_cuda_graph_capture: "2"`.
Set `ep.cuda.arena.use_cuda_mempool: "1"` in both session-options objects.
Set `custom_ops_library: "libbolt_ops.so"` in decoder session options (also valid
in drafter session options). Provider-specific memory/GEMM settings from the
qualified reference runtime may be retained. Decoder cache name mappings and
layer counts are not used: the native state validates the actual graph metadata.
The standard decoder input keys `past_sequence_length: "seqlens_k"` and
`total_sequence_length: "total_sequence_length"` are accepted, but do not
override these native bindings. No decoder `state_groups` or Engine settings
are needed. Omit Engine batching settings; enabled static/dynamic batching is
rejected. The drafter's internal paged cache does not make the target paged.
There are no Bolt-specific JSON fields.
Set `speculative.max_draft_tokens: 7`; the verify width is fixed, not adaptive.

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

The plugin, ORT runtime, GenAI core/CUDA add-on, and custom operators must be
mutually compatible. Point `ORT_LIB_PATH` at the compatible runtime library.
Set `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` before loading the runtime, as in the
qualified reference environment; the cuDNN attention path is not capture-safe
for this fixed-R8 graph.
The build-local bare `onnxruntime_genai` extension can be imported directly by
putting `build/cuda/src/python` on `PYTHONPATH`; make the GenAI libraries and CUDA
dependencies available to the dynamic loader. This does not require installing
or replacing any global Python package. Keep the CUDA plugin registered for the
process lifetime with the supplied ORT build: explicit unregistration can
invalidate ORT's shared allocators and fail during runtime shutdown, even after
all models have been destroyed. Normal Python module cleanup tears down GenAI.

## State and public-token semantics

Prompt ingestion uses the same fixed-R8 target, eight tokens per call. Intermediate
chunks execute asynchronously without host readback; the final chunk is padded
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
