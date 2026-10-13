# Qwen3.8-Flash CUDA-graph decode + MTP(1) through the GenAI Engine

`model.type = "qwen38_flash_graph"` runs the Qwen3.8-Flash-Next NVFP4 static-capture graphs (fixed in-place KV, three-bank
recurrent ring, device-side draft -> verify -> commit loop, MTP head) behind the **Engine API** (`og.Engine` / `OgaEngine`). The
Engine hosts it as a single-slot static backend (see "Static single-slot backend" in `docs/paged_attention_engine.md`): one
Engine step is one prompt chunk or one whole cycle that commits one or two greedy tokens; EOS, limits, stop strings, turns,
cancellation and statistics are the Engine's. There is no Generator API support for this model type.

## Build

```bash
cmake ... -DUSE_CUDA=ON -DENABLE_QWEN38_PLANE=ON        # Linux + CUDA; builds libqwen38_plane.so next to GenAI
make onnxruntime-genai python qwen38_plane
```

`libqwen38_plane.so` (the CUDA data plane: per-row input preparation with the host-registered FP8 PLE gather, argmax, verify
epilogue, head launch preparation) is `dlopen`-ed from a path inside the model folder; GenAI itself never links CUDA runtime for
it. The model needs the plugin CUDA EP (`og.register_execution_provider_library`) and an ORT whose CUDA provider supports the
S=2 target graphs and `arena_extend_strategy=kSameAsRequested`, plus the custom-op libraries of the export (all listed in
`session_options.custom_ops_libraries`, resolved inside the model folder).

## External artifacts

The Engine backend, the plane library and these scripts are in this repository. What a package needs on top is not:

| artifact | where it comes from |
| --- | --- |
| custom-op libraries of the exported graphs (fused attention, hyper-connection, GEMM, fold and MTP-head ops, guarded DSA and packed QSA indexer; domains `qwen.capture`, `qwen.fused*`, `qwen.static`, `qwen.mtp`) | build them from https://github.com/tianleiwu/qwen38-flash-ops (`sm_90` builds) and list them in `session_options.custom_ops_libraries` of the package (resolved inside the model folder) |
| the exported static-capture graphs (target, MTP head, and for the v12b stack the folded, dense, head-R2 and head-KV graphs plus `weights_pool.txt` and the QSA constants) | exported from the Qwen3.8-Flash-Next NVFP4 checkpoint together with the libraries above; the package layout is the one `build_package.py` writes |
| a CUDA execution provider that supports the S=2 target graphs, NVFP4 MoE and `arena_extend_strategy=kSameAsRequested` | the plugin CUDA EP, registered with `og.register_execution_provider_library` (`--provider` or `$QWEN_PROVIDER`; one provider path per process) |
| `libqwen38_plane.so` | `make qwen38_plane` of this tree (`-DENABLE_QWEN38_PLANE=ON`) |
| natural-text pools for the benchmark and gate prompts | any natural text of at least ~40k tokens, as `--text-pool` (and `--prose-pool` for the gate's `prose` case): a text file or a `.npy` of token ids |
| reference streams for the identity gate | a recording of the same prompts from another runner, or of this backend with `--no-mtp` / `--force-reject` (see `engine_gate.py`) |

## Package

`build_package.py --out <dir> --target-bundle <dir> --head-bundle <dir> --plane-lib <libqwen38_plane.so> --checkpoint <dir> --lib-*
<file> ...` hard-links a package from those parts (`genai_config.json` with `model.qwen38_graph`, `target/`, `mtp_head/`, `libs/`,
`ple/` + `ple_table.txt`, tokenizer). The PLE table is written with the `mobius` package; pass `--skip-ple` for a bundle that has one.
Use `--stop-token-ids 248319` for a fixed-length benchmark package (config overlays append to `eos_token_id`, so "ignore EOS" needs
its own package). `mtp.enabled=false` (config overlay) gives the plain S=1 graph decode.

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
stream is the no-MTP greedy stream. A v12b package is built from an existing v12b package with `build_package_engine_v12b.py`
(hard links, the plane library of this build, optional unreachable stop id).

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
| `build_package.py` | v11 package (S=1 / S=2 chain verify) from its parts, with the PLE table |
| `build_package_engine_v12b.py` | v12b package (folded cycle + fast prefill) from an existing v12b package: hard links, the plane library of this build, optional unreachable stop id, `package_manifest.json` with the SHA256 of every library |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token-identity gate: greedy streams of the gate prompts against a `--reference` (a recording of `--no-mtp` or `--force-reject`, i.e. MTP == no-MTP, or another runner's streams); `--prefill steps` must be identical, `--prefill fast` is reported as agreement statistics; EOS mode and `--ignore-eos` fixed length |
| `engine_behaviours.py` | (`--prefill steps\|fast`) stop tokens, limits, stop strings, cancel during prefill/decode, second request queued, continuation, rewind, capability errors, statistics |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token) |
| `bench_ttft_engine.py` | time to first token at several prompt lengths (+ optional MMLU-Pro prompt ids from `--mmlu-prompts`): begin_turn to the Run that delivers the token |
| `engine_prompt_lengths.py` | `run` the same greedy turns for 1 .. 20000-token prompts (dense chunk boundaries 8191/8192/8193, ...) with `--prefill steps` and `--prefill fast`, then `compare` (first token equal, shared prefix) |

A qualification pass of a package: `engine_gate.py` with `--force-reject --prefill steps` (record), then without `--force-reject`
with `--reference` of that recording (MTP stream == no-MTP stream); `engine_behaviours.py --prefill steps`; `bench_engine_mtp.py` on
the fixed-length package; `bench_ttft_engine.py`. The provider-side settings the numbers in the PR descriptions were taken with are
`CUDA_MODULE_LOADING=EAGER ORT_ENABLE_FP4_GEMV=1 ORT_NVFP4_FC1_EXPERT_REUSE=0 ORT_FORCE_DETERMINISTIC_MOE=1`
(provider-branch switches; they are not read by GenAI).

The logic that does not need a GPU is covered by `test/cpp/engine/qwen38_graph_*_tests.cpp` (scripted device, real Engine).
`engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'` measures the host cost of one Engine step with
an instantly answering device.

## Limitations (each is a clear error)

* Greedy only (the device loop verifies drafts by argmax): sampling, repetition penalty, no-repeat n-grams,
  `min_generated_tokens`, seeds and guidance are rejected by `BeginTurn`.
* One resident Request; further Requests queue behind it (static batching semantics). One Engine per model at a time.
* No prefix cache, no continuous batching, no caller-supplied draft tokens.
* A continued turn and the replay after `RewindToStartOfTurn` re-ingest the whole history (a fast prefill with `prefill.enabled`,
  S=2 step speed otherwise).
* Sessions up to `kv_capacity - 4` tokens (33,024 with the default capacity).
* Linux + CUDA only (`dlopen`, `mmap`, `cudaHostRegister`); other platforms compile the sources but creating the model fails with
  a clear message.
