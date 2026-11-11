# Native Nemotron-3.5 Bolt / LiLiCorr

`nemotron_bolt` implements batch-one, single-CUDA-device greedy generation through
the ordinary Model / Generator API. It does **not** use the paged target Engine or
support dynamic batching. The implementation runs a fixed-`R=8` or symbolic-`R`
(`R in {8, 16}`) target and the LiLiCorr DFlash2 drafter directly in C++ with ORT.
No Python inference bridge or synthetic logits are involved.

## Target row contract and `R=16` prefill

The target ONNX graph must declare either a fixed 8-row token contract or a
symbolic-`R` token contract across all four token-indexed tensors:

- Fixed-`R=8` target graph (`supports_r16_prefill_ = false`):
  `input_ids [1,8]`, `pred_ids [1,8]`, `logits [1,8,vocab_size]`, and
  `aux_hidden_states [1,8,width]`.
- Symbolic-`R` target graph (`supports_r16_prefill_ = true`):
  `input_ids [1,R]`, `pred_ids [1,R]`, `logits [1,R,vocab_size]`, and
  `aux_hidden_states [1,R,width]`, where the dynamic dimension (`-1`) carries
  the exact symbolic name `"R"` on all four tensors.

In both contracts, dense attention K/V caches have shape `[1,2,capacity,128]`
(`bf16`) and recurrent state buffers keep their 8-row geometry:
`past_conv.*` / `present_conv.*` have shape `[8,10240,3]` (`bf16`) and
`past_ssm.*` / `present_ssm.*` have shape `[8,128,128,64]` (`float32`).
Symbolic-`R` shape metadata establishes graph-shape eligibility only; it does
**not** certify that an arbitrary `libbolt_ops.so` binary implements `R=16`
prefill semantics.

When `R=16` prefill is active on a symbolic-`R` target:

- Full 16-token prompt chunks execute on captured CUDA graph ID `16` (`R=16`)
  with `seqlens_k = L + 15` and write the terminal recurrent state after all 16
  tokens into **slot 7** (`state_row_out = 7`) of the 8-row recurrent state
  buffers (`past_conv.*` / `present_conv.*` and `past_ssm.*` / `present_ssm.*`).
  Intermediate recurrent rows within an `R=16` prefill chunk are never selected
  by speculative rejection because prompt prefill commits all 16 tokens.
- Any trailing prompt remainder of `1..15` tokens is processed by one or two
  `R=8` chunks (`gpu_graph_id = 8`), with the final chunk padded to 8 rows and
  committing `state_row_out = valid_rows - 1`.
- All post-prefill speculative verification steps continue to execute on the
  8-row profile (`R=8`, `gpu_graph_id = 8`).

### Generator-local `BOLT_DISABLE_R16_PREFILL` switch

Each `Generator` (`NemotronBoltState`) resolves its `r16_enabled_` policy
**exactly once** at construction, before input-buffer layout, `R=16` tensor
allocation, or CUDA graph warmup:

- `BOLT_DISABLE_R16_PREFILL=1` disables `R=16` for that generator. Unset, empty
  `""`, `"0"`, and any other values do not disable `R=16`.
- When disabled (or when the model is fixed-`R=8`), the generator uses an
  `R=8`-only input buffer layout, does not allocate `R=16` tensors, executes
  **zero** `R=16` warmup calls during construction, and plans all prompt chunks
  as `R=8` (`gpu_graph_id = 8`).
- `generator.rewind_to(0)` (`RewindToLength(0)`) preserves that generator's
  frozen policy. Changing `BOLT_DISABLE_R16_PREFILL` after a generator is
  constructed does not affect that existing generator; a newly constructed
  generator snapshots the updated environment.

## Package configuration

Use `model.type: "nemotron_bolt"`, `model.vocab_size: 131072`,
`model.context_length: 135160`, and `model.decoder.filename: "model.onnx"` for a
fixed-R8 or symbolic-R graph with KV capacity 135168. `context_length`
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
drafter ONNX path. Configure **both** sessions with the CUDA provider and
`enable_cuda_graph: "1"` and `min_num_runs_before_cuda_graph_capture: "2"`.
Set `ep.cuda.arena.use_cuda_mempool: "1"` in both session-options objects.
Set `custom_ops_library: "libbolt_ops.so"` in decoder session options (also valid
in drafter session options). Provider-specific memory/GEMM settings from the
reference runtime may be retained. Decoder cache name mappings and
layer counts are not used: the native state validates the actual graph metadata.
The standard decoder input keys `past_sequence_length: "seqlens_k"` and
`total_sequence_length: "total_sequence_length"` are accepted, but do not
override these native bindings. No decoder `state_groups`, `sliding_window`, or
Engine settings are needed; decoder `sliding_window` and enabled static/dynamic
batching are rejected. The drafter's internal paged cache and sliding window do
not make the target paged or windowed.
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
  read before an in-place write can overwrite it. For `R=8` calls, output row
  `r` (`0 <= r < 8`) is the state after consuming input rows 0 through `r`;
  after accepting `m` drafts, the next call selects row `m` and advances the
  logical KV length by `m + 1`, so rejected recurrent rows never influence the
  committed continuation. For `R=16` prompt prefill calls (`gpu_graph_id = 16`),
  all 16 input rows are committed and the custom operators must write the
  terminal recurrent state after input row 15 into **slot 7** of the 8-row
  `present_conv.*` and `present_ssm.*` buffers (`state_row_out = 7`).
- **Dense KV and padding:** for logical past length `L`, an `R=8` call processes
  eight tokens starting at `L` with device `seqlens_k = L + 7`, while an `R=16`
  prefill call processes sixteen tokens starting at `L` with device
  `seqlens_k = L + 15`. Both pass the full allocation capacity as the CPU scalar
  `total_sequence_length`. Attention must remain causal for each row and ignore
  stale KV beyond the logical prefix. Later calls overwrite rejected positions.
  A padded final `R=8` prompt chunk commits only its valid rows and selects its
  last valid recurrent row (`valid_rows - 1`).
- **Capture and streams:** all operator work must follow the CUDA EP stream and
  be capture/replay safe. Captured work must read mutable input values, including
  `state_row` and `seqlens_k`, on every replay rather than baking in their
  warmup values. When `r16_enabled_` is true, the runtime executes four dummy
  `R=8` target runs (`gpu_graph_id = 8`), four dummy `R=16` target runs
  (`gpu_graph_id = 16`), and four dummy drafter proposals with capture delay
  two, discards their outputs, and synchronously resets caches before ingesting
  a real prompt. When `r16_enabled_` is false, only the four `R=8` target runs
  and four drafter proposals are warmed. Changing adapters is unsupported.
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

For the A100-SXM4-80GB custom-op `R=16` qualification across the companion PR stack:

| Component / Artifact | Source / Description | Commit / SHA-256 |
|---|---|---|
| `tianleiwu/nemotron_ops` (`PR #2`) | Custom operators (`_R16` GEMV/MoE dispatch and `R=16` Mamba conv/SSM slot-7 terminal-state writeback, base `e4f63ffdcb830e2faf5ddb149047953d922a6afb`) | Commit `262c685bd83383da79adb09f4000d2f2df91dbaa`<br>`libbolt_ops.so`: `4687f7331d482d7e3cd2aaafe645ca347bd8b9523f254290085c148ffcda292e` |
| `microsoft/onnxruntime` (`PR #33927`) | CUDA plugin EP per-session captured-graph retirement (`ReleaseCapturedGraph`, base `74e4120bbc307ecb5c81ef84a9cf42f1fe19519e`) | Commit `2659a76a9a74a5b9d2e6ebaa97a8391cb677eae9` |
| `model.onnx` (symbolic-`R`) | Target ONNX graph (`input_ids [1,R]`, `pred_ids [1,R]`, `logits [1,R,131072]`, `aux_hidden_states [1,R,24576]`) | `793c20914f6b4ceb1b423a7f1576637c35a48f6ed49c1089e2e433ba42af45c0` |
| `model.onnx` (fixed-`R=8`) | Fixed-`R=8` compatibility target ONNX graph (`input_ids [1,8]`, `pred_ids [1,8]`, `logits [1,8,131072]`, `aux_hidden_states [1,8,24576]`) | `f8f0cd64226c1c575cc0ada277cb6b9f3411708430eaa69d7a75b3a5a1158cc8` |
| `model.onnx.data` | Target external weights (`43,085,691,904` bytes) | `d0b2acfa01b0a622a39e8d3b5d694cd8d09141d67213e9d38126ef5690e68808` |
| `dflash2.onnx` | LiLiCorr DFlash2 drafter ONNX graph | `391d8387c2afc72f9769de3572506f88137c9aed871683e341386fd8303a57a7` |
| `dflash2.onnx.data` | LiLiCorr DFlash2 drafter external weights (`3,937,009,664` bytes) | `a82041f866552f592de7e0f7dcbb930ac60187992826c15f7c27c11c72c8de26` |
| `native-iq2-memory.bin` | Native GGUF `IQ2_XXS` expert/shared-expert payload (`17,506,959,360` bytes) | `6ac1b8a01d3b98969e1d30324918594b59db3a0ed84885aaa0f218b4c7f1b81b` |
| `q8-source-scales.fp16.bin` | Source `Q8_0` FP16 block scales (`72,351,744` bytes) | `70cbb59196977b6ad2079e3f64e1e9b5bb09804946bb97779e9aba5d93374631` |
| `genai_config.json` | Package runtime configuration | `a283ea5606d9414a4ac3353fc5a9e478e4e4ad95abddd846ec7cd747e4959e8d` |
| `tokenizer.json` | Tokenizer vocabulary and merges | `d0432ad8b1c2984a2e2dca82e1933e3fd5628df6e95c51fba4c3ee9950dce299` |
| `ort-token-ids.json` | 32,768-token prompt fixture (`doc-p32768-01-service-incident`) | `654648292d84f4d0696c763ab2118c2c41b8cf7b1018bea9443402d20f9865d2` |

## State and public-token semantics

Prompt ingestion uses `R=16` chunks followed by up to two `R=8` tail chunks when
`r16_enabled_` is true, or `R=8` chunks exclusively when `r16_enabled_` is false.
Intermediate chunks execute asynchronously without host readback; the final chunk
is padded when needed (`R=8` tail) but only valid rows are committed, and its
last valid row selects the recurrent state (`7` for an `R=16` chunk,
`valid_rows - 1` for an `R=8` chunk) and first generated anchor. For multi-chunk
prompts, `Prefill()` retains auxiliary hidden states starting from the 8-row
aligned offset `draft_start = ((prompt_rows - context_window) / 8) * 8` when
`prompt_rows > context_window` (where `context_window = sliding_window + block_size`,
e.g. `1024 + 8 = 1032`), and `Dflash2Drafter::Propose` trims any leading
`draft_rows - context_window` excess rows (`0..7` rows) so the drafter ingests
the exact final `min(prompt_rows, context_window)` suffix. Target `past_*` and `present_*` share the same allocation. Verification consumes `[anchor, seven drafts]`,
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
runtime speedup.

### 32,768-token A/B benchmark summary and reproducible harness

On an idle NVIDIA A100-SXM4-80GB GPU (`ORT_ENABLE_CUDNN_FLASH_ATTENTION=0`,
single model-owning process with fresh generators per request, 32,768 prompt
tokens + 160 generated tokens, 2 untimed warmups `R8, R16` followed by 8
measured interleaved requests `R8, R16, R16, R8, R8, R16, R16, R8`):

| Metric | `R=8` (`BOLT_DISABLE_R16_PREFILL=1`, 4 runs) | `R=16` (`BOLT_DISABLE_R16_PREFILL=0`, 4 runs) | `R=8` Median | `R=16` Median | Median Change |
|---|---|---|---:|---:|---:|
| `first_content_ms` (construction + `append_tokens` + first `generate_next_token`) | `64231.37, 64230.97, 64263.77, 64239.28` | `44840.57, 44808.86, 44865.15, 44854.53` | `64,235.325 ms` | `44,847.548 ms` | `-30.18%` |
| `append_first_content_ms` (`append_tokens` + first `generate_next_token`) | `64103.05, 64105.37, 64109.19, 64102.99` | `44611.03, 44579.47, 44620.67, 44623.53` | `64,104.210 ms` | `44,615.849 ms` | `-30.40%` |
| `request_ms` (total request latency through 160 generated tokens) | `65369.41, 65379.47, 65413.84, 65380.46` | `45985.77, 45956.16, 46004.37, 45998.04` | `65,379.965 ms` | `45,991.905 ms` | `-29.65%` |
| `decode_tokens_per_second` (`159 / post_first_content_decode_s`) | `139.70, 138.44, 138.24, 139.32` | `138.84, 138.84, 139.04, 139.56` | `138.879 tok/s` | `138.937 tok/s` | `+0.04%` |

All 10 requests produced identical 160 generated tokens and 36 speculative
verification rounds (`acceptance_length_histogram: [3, 6, 4, 8, 2, 1, 1, 11]`,
mean accepted draft tokens per round `3.4167`).

To reproduce the interleaved 10-request A/B timing block from Python:

```python
import gc, json, os, statistics, time
import onnxruntime_genai as og

og.register_execution_provider_library("CUDAExecutionProvider", os.environ["ORT_CUDA_PLUGIN_PATH"])
prompt_ids = json.load(open(os.environ["BOLT_32K_PROMPT_JSON"]))["token_ids"]
assert len(prompt_ids) == 32768
gen_budget = 160
model = og.Model(os.environ["BOLT_MODEL_DIR"])

schedule = [("warmup", "R8"), ("warmup", "R16")] + [
    ("measured", m) for m in ("R8", "R16", "R16", "R8", "R8", "R16", "R16", "R8")
]
records = []
for phase, mode in schedule:
    os.environ["BOLT_DISABLE_R16_PREFILL"] = "1" if mode == "R8" else "0"
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(prompt_ids) + gen_budget)
    t0 = time.perf_counter()
    gen = og.Generator(model, params)
    t_ctor = time.perf_counter()
    gen.append_tokens(prompt_ids)
    gen.generate_next_token()
    t_first = time.perf_counter()
    while not gen.is_done():
        gen.generate_next_token()
    t_end = time.perf_counter()
    seq = list(gen.get_sequence(0))[len(prompt_ids):]
    stats = gen.get_speculative_stats()
    del gen, params
    gc.collect()
    if phase == "measured":
        records.append({
            "mode": mode,
            "first_content_ms": (t_first - t0) * 1e3,
            "append_first_content_ms": (t_first - t_ctor) * 1e3,
            "request_ms": (t_end - t0) * 1e3,
            "decode_tokens_per_second": (len(seq) - 1) / (t_end - t_first),
            "generated_tokens": len(seq),
            "rounds": stats["rounds"],
        })

for mode in ("R8", "R16"):
    sub = [r for r in records if r["mode"] == mode]
    print(mode, {k: statistics.median(r[k] for r in sub) for k in (
        "first_content_ms", "append_first_content_ms", "request_ms", "decode_tokens_per_second"
    )})
```

Only reset-to-zero rewind is supported: `generator.rewind_to(0)` clears target
state, drafter context, and pending tokens. Continuing with a second
`append_tokens()` requires this reset or a new generator. Sampling, beams,
guidance, nontrivial penalties, external chunking/speculative strategies,
nonzero rewinds, extra inputs, and logits access/replacement are explicitly
unsupported. In particular `get_logits()` fails rather than fabricating scores.

## Focused native tests

`engine_unit_tests --gtest_filter='NemotronBolt.*'` runs CPU-only acceptance,
EOS, budget, search-option, public-token, reset, disable-switch resolution,
prefill chunk planning, and production auxiliary-window / drafter-trimming tests.
To run the real-package GPU regression with the symbolic-`R` package and
optional fixed-`R=8` compatibility fixture:

```bash
ORTGENAI_BOLT_TEST_MODEL=/path/to/symbolic-r-bolt-package \
ORTGENAI_BOLT_TEST_MODEL_FIXED_R8=/path/to/fixed-r8-bolt-package \
ORTGENAI_BOLT_RUN_32K_TEST=1 \
ORTGENAI_BOLT_REQUIRE_GPU_QUALIFICATION=1 \
engine_unit_tests --ep_dir /runtime --gtest_filter='NemotronBoltGPU.*'
```

`NemotronBoltGPU.SequentialGeneratorsResetAndPublicBoundaries` verifies
disabled construction (`BOLT_DISABLE_R16_PREFILL=1` executes zero `R=16` calls
including construction warmup), policy stability when the environment changes
after construction or across `RewindToLength(0)`, sequential generator
recreation, boundary lengths `1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1031, 1032, 1033`
(asserting identical generated tokens, per-round acceptance traces, and
profile-specific warmup/prefill/verify counts between disabled `R=8` and enabled
`R=16`), the selectable 32,768-token prompt (`4,096` `R=8` vs `2,048` `R=16`
prompt prefill calls), and fixed-`R=8` model compatibility. Same-runtime parity
between `R=8` and `R=16` verifies width equivalence rather than an independent
HuggingFace reference oracle.
