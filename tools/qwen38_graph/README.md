# Qwen3.8-Flash CUDA-graph decode with MTP and DFlash2 speculation through the GenAI Engine

`model.type = "qwen38_flash_graph"` runs the Qwen3.8-Flash-Next static-capture graphs (fixed in-place KV, a recurrent-state bank
ring, a device-side draft -> verify -> commit loop with an MTP head and, in the DFlash2 packages, a DFlash2 drafter) behind the
**Engine API** (`og.Engine` / `OgaEngine`). The Engine hosts it as a static backend that selects its own tokens (see "Static
backend that selects its own tokens" in `docs/paged_attention_engine.md`): the device holds `model.qwen38_graph.max_batch_size`
slots (default 1), and one Engine step is one prompt chunk or one whole cycle that commits up to S greedy tokens (S = 2 for MTP(1),
up to 8 for DFlash2); EOS, limits, stop strings, turns, cancellation and statistics are the Engine's. There is no Generator API
support for this model type.

## Build

```bash
cmake ... -DUSE_CUDA=ON                                  # Linux + CUDA; the plane is compiled into onnxruntime-genai-cuda
make onnxruntime-genai onnxruntime-genai-cuda python
```

The CUDA data plane (`src/qwen38_plane`: per-row input preparation with the host-registered PLE gather, argmax, verify epilogue,
head launch preparation, INT8 embedding rows) is compiled into `onnxruntime-genai-cuda`, which the model loads next to GenAI;
GenAI itself never links the CUDA runtime for it. `-DENABLE_QWEN38_PLANE=ON` additionally builds a standalone
`libqwen38_plane.so`, which `model.qwen38_graph.plane_library` can name (a path inside the model folder; an empty value or a
missing file selects the built-in plane). The model needs the plugin CUDA EP (`og.register_execution_provider_library`) and an ORT
whose CUDA provider supports the S=2 target graphs and `arena_extend_strategy=kSameAsRequested`, plus the custom-op libraries of
the export (all listed in `session_options.custom_ops_libraries`, resolved inside the model folder).

## External artifacts

The Engine backend, the plane library and these scripts are in this repository. What a package needs on top is not:

| artifact | where it comes from |
| --- | --- |
| custom-op libraries of the exported graphs (fused attention, hyper-connection, GEMM, fold and MTP-head ops, guarded DSA and packed QSA indexer; domains `qwen.capture`, `qwen.fused*`, `qwen.static`, `qwen.mtp`) | build them from https://github.com/tianleiwu/qwen38-flash-ops (`sm_90` builds) and list them in `session_options.custom_ops_libraries` of the package (resolved inside the model folder) |
| the exported static-capture graphs (target, MTP head, and for the v12b stack the folded, dense, head-R2 and head-KV graphs plus `weights_pool.txt` and the QSA constants) | exported from the Qwen3.8-Flash-Next NVFP4 checkpoint together with the libraries above; see "Package" for the layout the scripts expect |
| a CUDA execution provider that supports the S=2 target graphs, NVFP4 MoE and `arena_extend_strategy=kSameAsRequested` | the plugin CUDA EP, registered with `og.register_execution_provider_library` (`--provider`, `$QWEN_PROVIDER` or `<package>/provider/`; one provider path per process) |
| `libqwen38_plane.so` (optional) | `make qwen38_plane` of this tree (`-DENABLE_QWEN38_PLANE=ON`); without it the model uses the plane built into `onnxruntime-genai-cuda` |
| a natural-text pool for the benchmark prompts | any natural text of at least ~40k tokens, as `--text-pool` (or `$QWEN38_TEXT_POOL`): a text file or a `.npy` of token ids |

## Package

A package is a model folder: `genai_config.json` with `model.qwen38_graph`, `target/`, `mtp_head/`, `libs/`, `ple/` +
`ple_table.txt` and the tokenizer files; the v12b stack below adds its own graphs. Every script takes the folder as its first
argument and overlays config values for the run: `--no-mtp` is `mtp.enabled=false` (the plain S=1 graph decode) and
`--prefill steps|fast` is `prefill.enabled`. Config overlays append to `eos_token_id`, so a fixed-length benchmark ("ignore EOS")
needs its own package whose stop id is an ordinary id that is never generated (`248319`).

## v12b stack: folded cycle + fast prefill + shared weights

With `model.qwen38_graph.fold.enabled` the decoder session is the **fold** model (verify S=2 body + MTP head + glue): one CUDA-graph
launch per cycle, the cycle record is published mid-graph into a pinned ring that `WaitCycle` spins on. A standalone R=2 head
session (`fold.head_r2`) serves the prompt context and the first draft. With `prefill.enabled` the prompt goes through the dense
chunked prefill session (`prefill.chunk` tokens, default 8192; the Engine delivers the prompt in chunks of that size and the dense
state continues across the chunks), is handed over to the packed decode state in `BeginDecode` (KV rows, GDN state, conv / PLE
state, pooled compressed index keys; the `qwen38_plane` kernels `ple_rows`, `pool_keys`, `transpose_f32`) and the head KV is
prefilled in batches by the KV-only `prefill.head_kv` model. The initializers that several sessions share (`prefill.weights_pool`,
a text manifest) are loaded once to the device and handed to every session with `AddInitializer` (one copy of the weights).
`prefill.enabled=false` keeps the bit-exact muted-fold S=2-step prompt path (the token gate reference); the fast prefill is
tolerance level and not run-to-run reproducible. `fold.force_reject` (default off) is a test hook: every draft is replaced, so the
stream is the no-MTP greedy stream.

### DFlash2 S=8 packages (rc4): `SpeculativePrefixCommit` contract

The `qwen38_flash_graph` fold model of the DFlash2 packages commits each verified cycle with one
`com.microsoft::SpeculativePrefixCommit` node (ORT PR #33474; 3 <= `head_rows` <= 8). The runtime wires it as follows:

| Graph I/O | Binding |
|---|---|
| in `next_tokens`, `input_ids`, `validation.aggregate` | regular row inputs / target outputs |
| in `fold.ctl` (16 int64 per slot), `fold.ctx_commit` (2 per slot), `fold.ctx_a` (`2 * head_rows` per slot, slot stride 16), `fold.ptrs` | persistent device control block (`ctl_`, `ctx_commit_`, `ctx_a_`) and `{result ring address, ring size}` |
| out `fold.tail_done` (`head_c`, int32 `[batch]`) | allocated by the engine |
| out `fold.ctl_out` (`updated_ctl`), `fold.ctx_commit_out` (`updated_ctx_commit`) | **required**; bound in place over the same `ctl_` / `ctx_commit_` buffers as `fold.ctl` / `fold.ctx_commit` (the op is alias safe) so the state update is never lost |

`fold.ctx_b`, the rotary caches and the `head.capture.*` / `fold.sel` outputs of the rc3 node are gone; rc3 packages (10-input /
7-output node) are not loadable by this runtime. The same bindings apply to the target decode graphs (single slot, batched
`B = 2..max_batch`) and the unified prefill graphs (muted slot: the op rewrites the unchanged state in place). A package is
upgraded from rc3 with `scripts/qwen_flash_next/derive_rc4_from_rc3.py` of the packaging repo (no re-quantization).

## Engine usage

```python
og.register_execution_provider_library("CUDAExecutionProvider", PROVIDER_SO)
model = og.Model(og.Config(PKG))
engine = og.Engine(model)
buffer = engine.create_event_buffer(16)
request = engine.create_request()
turn = og.TurnOptions(request)
turn.set_do_sample(False)                  # greedy only
turn.set_max_generated_tokens(256)
request.begin_turn(prompt_tokens, turn)
while engine.has_pending_requests():
    for event in engine.run(buffer):       # one Run = one prompt chunk or one cycle (1-2 tokens)
        ...
```

## Scripts

All need a package; the ones that build prompts need a text pool (`--text-pool`, see above).

| script | purpose |
| --- | --- |
| `qwen38_engine_util.py` | helpers shared by the scripts: CUDA EP registration and model loading with config overlays (`$QWEN38_OVERLAY_JSON`), the natural-text prompt pools, and `run_turn` (one greedy turn through `og.Engine`); registers the DLL directories on Windows |
| `calibrate_qwen38_kv.py` | calibrate per-channel INT8 and FP8 KV-cache scales for the 12 target sparse-attention layers and 1 MTP-head sparse-attention layer |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token) |
| `bench_ttft_engine.py` | time to first token at several prompt lengths (+ optional MMLU-Pro prompt ids from `--mmlu-prompts`): begin_turn to the Run that delivers the token |
| `run_with_ram_budget.py` | runs a command (for example `bench_engine_mtp.py`) with a bounded host expert cache for a QMoE-offload package: `$ORT_QMOE_HOST_CACHE_MIB` / `$ORT_QMOE_HOST_CACHE_LAYERS` from `--ram-budget-gib` minus `--reserve-gib`, and with `--enforce` a `systemd-run --user` cap on the whole command; `test_ram_budget.py` is its unit test (`python -m unittest test_ram_budget`) |

The provider-side settings the numbers in the PR descriptions were taken with are
`CUDA_MODULE_LOADING=EAGER ORT_ENABLE_FP4_GEMV=1 ORT_NVFP4_FC1_EXPERT_REUSE=0 ORT_FORCE_DETERMINISTIC_MOE=1`
(provider-branch switches; they are not read by GenAI).

The logic that does not need a GPU is covered by `test/cpp/engine/qwen38_graph_*_tests.cpp` (scripted device, real Engine).
`engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'` measures the host cost of one Engine step with
an instantly answering device.

## Limitations (each is a clear error)

* Greedy only (the device loop verifies drafts by argmax): sampling, repetition penalty, no-repeat n-grams,
  `min_generated_tokens`, seeds and guidance are rejected by `BeginTurn`.
* One resident Request per slot (`model.qwen38_graph.max_batch_size`, default 1); further Requests queue behind the resident ones.
  One Engine per model at a time.
* No prefix cache and no caller-supplied draft tokens.
* A continued turn and the replay after `RewindToStartOfTurn` re-ingest the whole history (a fast prefill with `prefill.enabled`,
  S=2 step speed otherwise).
* Sessions up to `kv_capacity - 4` tokens (33,024 with the default capacity).
* CUDA only (Linux and Windows); other platforms compile the sources but creating the model fails with a clear message.
