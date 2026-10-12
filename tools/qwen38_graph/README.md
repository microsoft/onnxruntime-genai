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

## Package

`build_package.py --out <dir> --plane-lib <libqwen38_plane.so>` hard-links a package from the research bundles
(`genai_config.json` with `model.qwen38_graph`, `target/`, `mtp_head/`, `libs/`, `ple/` + `ple_table.txt`, tokenizer). Use
`--stop-token-ids 248319` for a fixed-length benchmark package (config overlays append to `eos_token_id`, so "ignore EOS" needs
its own package). `mtp.enabled=false` (config overlay) gives the plain S=1 graph decode.

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

## Scripts (all need the real package and the research runner for prompts and reference hashes: `QWEN38_RUNNER_DIR`)

| script | purpose |
| --- | --- |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token identity against the research no-MTP greedy hashes (EOS mode and `--ignore-eos` fixed length, `--no-mtp`) |
| `engine_behaviours.py` | stop tokens, limits, stop strings, cancel during prefill/decode, second request queued, continuation, rewind, capability errors, statistics |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token) |

The logic that does not need a GPU is covered by `test/cpp/engine/qwen38_graph_*_tests.cpp` (scripted device, real Engine).
`engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'` measures the host cost of one Engine step with
an instantly answering device.

## Limitations (each is a clear error)

* Greedy only (the device loop verifies drafts by argmax): sampling, repetition penalty, no-repeat n-grams,
  `min_generated_tokens`, seeds and guidance are rejected by `BeginTurn`.
* One resident Request; further Requests queue behind it (static batching semantics). One Engine per model at a time.
* No prefix cache, no continuous batching, no caller-supplied draft tokens.
* A continued turn and the replay after `RewindToStartOfTurn` re-ingest the whole history (S=2 prefill speed).
* Sessions up to `kv_capacity - 4` tokens (33,024 with the default capacity).
* Linux + CUDA only (`dlopen`, `mmap`, `cudaHostRegister`); other platforms compile the sources but creating the model fails with
  a clear message.
