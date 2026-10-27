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

### v12c memory-reduced quantized package (`pkg_v12b_engine_int8kv_fp8w_fp4moe`)

`build_package_engine_v12b.py` also builds the **v12c** quantized package from a v12b package, reducing peak GPU memory by
**12.47 GiB** (`113.40 -> 100.93 GiB` on H200) and improving decode speed (`261.2 -> 265.3` MMLU decode tok/s) while achieving
`667/800` (`83.375%`) on MMLU-Pro 800:

1. **Per-channel INT8 (or FP8) KV cache** (`--kv-quant {int8,fp8} --kv-scale-file <json>`): quantizes the 12 target sparse-attention
   KV caches and 1 MTP-head sparse-attention KV cache in `GuardedDSA` (`qwen.static::GuardedDSA` with `k_scale` / `v_scale`
   attributes) and during the dense-to-packed hand-off (`qwen38_plane_quantize_kv`). Scales are calibrated with
   `calibrate_qwen38_kv.py` (partial-rotary `rotary_dim=64` pair envelope, 99.99th percentile on the calibration corpus).
2. **16-bit weight quantization + dense-prefill weight sharing** (`--weight-quant fp8 --mtp-moe-quant nvfp4 --share-dense-weights`):
   quantizes the 48-layer + head attention and shared-expert projections (`merged_proj`, `out_proj`, `o_proj`,
   `shared_expert.down_proj`) from BF16 to per-column FP8 E4M3FN (or block-16 NVFP4) in `qwen.fused_gemm::{SkinnyGemm,SharedExpertDown}`,
   quantizes the 512 MTP-head routed experts (`4.69 GiB` BF16 -> `1.32 GiB` NVFP4 or `2.36 GiB` FP8) in `qwen.mtp::MtpMoe`, and
   rewrites `dense/model.onnx` with `qwen.fused_gemm::{DequantWeightBf16,UntileHcBf16}` so the dense prefill graph shares all
   `merged_proj`, `out_proj`/`o_proj`/`shared_expert.down_proj`, and `hc_tiled.data` weights with `target/model.onnx` in
   `weights_pool.txt` (`100.90 GiB` -> `86.26 GiB` pool).

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
| `build_package.py` | v11 package (S=1 / S=2 chain verify) from its parts, with the PLE table |
| `build_package_engine_v12b.py` | v12b / v12c package (folded cycle + fast prefill, optional `--kv-quant`, `--weight-quant`, `--mtp-moe-quant`, `--share-dense-weights`) from an existing v12b package |
| `calibrate_qwen38_kv.py` | calibrate per-channel INT8 and FP8 KV-cache scales for the 12 target sparse-attention layers and 1 MTP-head sparse-attention layer |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token identity against the research no-MTP greedy hashes (EOS mode and `--ignore-eos` fixed length, `--no-mtp`); v12b: `--python-ref` (python runner streams) `--prefill steps\|fast` `--no-research-hash` |
| `engine_behaviours.py` | (`--prefill steps\|fast`) stop tokens, limits, stop strings, cancel during prefill/decode, second request queued, continuation, rewind, capability errors, statistics |
| `bench_ttft_engine.py` | time to first token at several prompt lengths (+ MMLU-Pro prompt ids): begin_turn to the Run that delivers the token |
| `mmlu_engine_eval.py` | MMLU-Pro 800 through the Engine (shards, resumable) and the MTP == no-MTP stream check (`--check-items`, `--force-reject`) |
| `engine_prompt_lengths.py` | `run` the same greedy turns for 1 .. 20000-token prompts (dense chunk boundaries 8191/8192/8193, ...) with `--prefill steps` and `--prefill fast`, then `compare` (first token equal, shared prefix) |
| `test_plane_kernels.py` | kernel tests of the plane's fast prefill kernels against the research implementations |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token) |

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
