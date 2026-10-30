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
| the exported static-capture graphs (target, MTP head, and for the v12b stack the folded, dense, head-R2 and head-KV graphs plus `weights_pool.txt` and the QSA constants) | exported from the Qwen3.8-Flash-Next NVFP4 checkpoint together with the libraries above; the package layout is the one `build_package.py` writes |
| a CUDA execution provider that supports the S=2 target graphs, NVFP4 MoE and `arena_extend_strategy=kSameAsRequested` | the plugin CUDA EP, registered with `og.register_execution_provider_library` (`--provider`, `$QWEN_PROVIDER` or `<package>/provider/`; one provider path per process) |
| `libqwen38_plane.so` (optional) | `make qwen38_plane` of this tree (`-DENABLE_QWEN38_PLANE=ON`); without it the model uses the plane built into `onnxruntime-genai-cuda` |
| natural-text pools for the benchmark and gate prompts | any natural text of at least ~40k tokens, as `--text-pool` (and `--prose-pool` for the gate's `prose` case; or `$QWEN38_TEXT_POOL` / `$QWEN38_PROSE_POOL`): a text file or a `.npy` of token ids |
| reference streams for the identity gate | a recording of the same prompts from another runner, or of this backend with `--no-mtp` / `--force-reject` (see `engine_gate.py`) |
| research inputs of `mmlu_engine_eval.py`, `lcb_engine_eval.py` and `test_plane_kernels.py` | the research helper modules (`eval_lib`, `eval_mcq_genai`; `$QWEN38_EVAL_PATHS` or `PYTHONPATH`), the MMLU-Pro and LiveCodeBench prompts, samples and tests, an HF checkpoint folder, and for the kernel tests `mobius`, `torch` and the research runner; the other scripts need none of them |

## Package

A package is a model folder: `genai_config.json` with `model.qwen38_graph`, `target/`, `mtp_head/`, `libs/`, `ple/` +
`ple_table.txt` and the tokenizer files; the v12b stack below adds its own graphs. Every script takes the folder as its first
argument and overlays config values for the run: `--no-mtp` is `mtp.enabled=false` (the plain S=1 graph decode) and
`--prefill steps|fast` is `prefill.enabled`. Config overlays append to `eos_token_id`, so a fixed-length benchmark ("ignore EOS")
needs its own package whose stop id is an ordinary id that is never generated (`248319`, set with `--stop-token-ids` of
`build_package.py` or `--stop-token-id` of `build_package_engine_v12b.py`).

`build_package.py --out <dir> --target-bundle <dir> --head-bundle <dir> --plane-lib <libqwen38_plane.so> --checkpoint <dir> --lib-*
<file> ...` hard-links a package of that layout from its parts: the exported target and MTP-head graphs, the custom-op libraries
(one `--lib-*` per library, all required), and the config, tokenizer files and PLE safetensors of the checkpoint (`--tokenizer-from`
takes precedence for the tokenizer files, `--copy` copies instead of linking). `--out` must be new or empty and must not be equal to
or nested in an input folder (or contain one), and two libraries must not share a file name. The PLE table is written with the
`mobius` package; pass `--skip-ple` for a bundle that has one. `--no-mtp` makes the plain S=1 graph decode the package default
(`mtp.enabled=false`; the head stays packaged).

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
tolerance level and not run-to-run reproducible. The test hook `ORT_QWEN38_FORCE_REJECT=1` (an environment variable of the process,
read once when the Engine is constructed; it is not a `genai_config.json` key) replaces every draft, so the stream is the no-MTP greedy
stream; `engine_gate.py --force-reject` and `mmlu_engine_eval.py --force-reject` set it for their Engine and clear an inherited value
otherwise. A v12b package is built from an existing v12b package with `build_package_engine_v12b.py` (hard links, the plane library of
this build, optional unreachable stop id).

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

### v12c and v12d memory-reduced quantized packages (`pkg_v12b_engine_int8kv_fp8w_fp4moe`, `pkg_v12d_engine`)

`build_package_engine_v12b.py` also builds the **v12c** and **v12d** quantized packages from a v12b package, reducing peak GPU
memory by up to **28.95 GiB** (`113.40 -> 100.79 GiB` in v12c, **`84.45 GiB`** in v12d on H200) and improving decode speed
(`260.0 -> 286.99` MMLU decode tok/s in v12c, **`290.26` tok/s** in v12d) while maintaining MMLU-Pro 800 accuracy (`668/800 =
83.50%` in v12c, `664/800 = 83.00%` in v12d BF16 LM-Head, `663/800 = 82.88%` in v12d FP8 LM-Head):

1. **Per-channel INT8 (or FP8) KV cache** (`--kv-quant {int8,fp8} --kv-scale-file <json>`): quantizes the 12 target sparse-attention
   KV caches and 1 MTP-head sparse-attention KV cache in `GuardedDSA` (`qwen.static::GuardedDSA` with `k_scale` / `v_scale`
   attributes) and during the dense-to-packed hand-off (`qwen38_plane_quantize_kv`). Scales are calibrated with
   `calibrate_qwen38_kv.py` (partial-rotary `rotary_dim=64` pair envelope, 99.99th percentile on the calibration corpus).
2. **16-bit weight quantization + dense/head-KV weight sharing** (`--weight-quant fp8 --mtp-moe-quant nvfp4 --lm-head-quant fp8 --share-dense-weights`):
   quantizes the 48-layer + head attention and shared-expert projections (`merged_proj`, `out_proj`, `o_proj`,
   `shared_expert.down_proj`) and `lm_head.weight_t` from BF16 to per-column FP8 E4M3FN in `qwen.fused_gemm::{SkinnyGemm,SharedExpertDown}`,
   quantizes the 512 MTP-head routed experts (`4.69 GiB` BF16 -> `1.32 GiB` NVFP4) in `qwen.mtp::MtpMoe`, merges the MTP head's
   hyper-connection blocks into `HcBlock`/`HcPostBlock`, and rewrites `dense/model.onnx` and `head_kv/model.onnx` with
   `qwen.fused_gemm::{DequantWeightBf16,UntileHcBf16}` (`col_offset` slice dequantization in `head_kv`) plus rotary-cache and
   `attn_consts.data` deduplication across all 4 sessions (`177.8 MiB -> 4.21 MiB` session-private initializers).
3. **v12d FP8 hyper-connection weights, host-mapped embeddings, and bounded prefill chunk** (`--hc-quant fp8 --host-map-embed --prefill-chunk 2048`):
   quantizes all 196 `[160, 20480]` hyper-connection `w_down`/`w_up` tensors in `hc_tiled.data` from BF16 (`1,225 MiB`) to FP8
   E4M3FN (`612.5 MiB`), maps `model.embed_tokens.weight` (`1.184 GiB`) read-only on the host via `host_pool` (`cudaHostRegisterMapped`),
   and caps `prefill.chunk` at `2048` (or `4096`) to bound the `dense` BFC arena activation peak.

### Clean root-layout packages (`--clean-package`)

With `--clean-package` (implied by `--explicit-cpu-embed`) `build_package_engine_v12b.py` writes the `v12d-engine-rc1` stack in a flat
root layout instead of the `target/`, `head_r2/`, `dense/` and `head_kv/` folders: one `model.onnx` that serves prefill and decode,
`mtp.onnx`, the shared weight files in the package root with the remaining BF16 initializers packed into `norm_and_rope.data`
(4096-byte aligned) and the expert scales compacted, and the token embedding split off into a standard CPU `embedding.onnx` +
`embedding.onnx.data` (`--embed-quant int8`, the default, is per-row INT8; `none` keeps BF16). Unreferenced safetensors and the
`dense.data`, `merged.data` and `generated.data` side files are dropped. `--mtp-kv-quant bf16` (the default) keeps the MTP-head KV
cache in BF16 while the target KV cache is INT8; `--keep-bf16-router-shared` and `--keep-bf16-shared-down` keep the MoE router and
the shared-expert projections in BF16.

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

The builders produce a package; every other script needs one, and the ones that build prompts need a text pool (`--text-pool`, see above).

| script | purpose |
| --- | --- |
| `build_package.py` | v11 package (S=1 / S=2 chain verify) from its parts, with the PLE table |
| `build_package_engine_v12b.py` | v12b / v12c / v12d package (folded cycle + fast prefill, optional `--kv-quant`, `--weight-quant`, `--mtp-moe-quant`, `--hc-quant`, `--share-dense-weights`, and the flat `--clean-package` layout) from an existing v12b package |
| `qwen38_engine_util.py` | helpers shared by the scripts: CUDA EP registration and model loading with config overlays (`$QWEN38_OVERLAY_JSON`), the natural-text prompt pools, and `run_turn` (one greedy turn through `og.Engine`); registers the DLL directories on Windows |
| `calibrate_qwen38_kv.py` | calibrate per-channel INT8 and FP8 KV-cache scales for the 12 target sparse-attention layers and 1 MTP-head sparse-attention layer (the JSON of the wanted quantization is the `--kv-scale-file` of `build_package_engine_v12b.py`); `--out` receives the `--quant-type` scales and the other quantization is written beside it under the name with the quantization token swapped (`kv_scales_int8_per_channel.json` -> `kv_scales_fp8_per_channel.json`, `scales.json` -> `scales.fp8_per_channel.json`), so the two files never collide (a collision is an error before any work) |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token-identity gate: greedy streams of the gate prompts against a `--reference` (a recording of `--no-mtp` or `--force-reject`, i.e. MTP == no-MTP, or another runner's streams); `--prefill steps` must be identical, `--prefill fast` is checked for an equal first token, equal prompt length and an identical prefix of at least `--min-fast-prefix` tokens and reported as agreement statistics; EOS mode and `--ignore-eos` fixed length |
| `engine_behaviours.py` | (`--prefill steps\|fast`) stop tokens, limits, stop strings, cancel during prefill/decode, second request queued, continuation, rewind, capability errors, statistics |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token); the acceptance statistics are the difference of the two full acceptance-length histograms taken around the run (rounds of different draft widths keep their own bins), and an inconsistent snapshot is an error |
| `bench_ttft_engine.py` | time to first token at several prompt lengths (+ optional MMLU-Pro prompt ids from `--mmlu-prompts`): begin_turn to the Run that delivers the token |
| `engine_prompt_lengths.py` | `run` the same greedy turns for 1 .. 20000-token prompts (dense chunk boundaries 8191/8192/8193, ...) with `--prefill steps` and `--prefill fast`, then `compare` (first token equal, shared prefix) |
| `mmlu_engine_eval.py` | MMLU-Pro 800 through the Engine (shards, resumable) and the MTP == no-MTP stream check (`--check-items`, `--force-reject`, `--compare`); a run stops before generating when an Engine prompt differs from the canonical prompt ids/text, and `--compare A B` (A an MTP run, B a `--force-reject` run) exits non-zero unless both cover the same items and every stream is identical |
| `lcb_engine_eval.py` | LiveCodeBench v6 (175 problems) through the Engine, generated in resumable `--shard K --shards N` runs; with `--grade --arms name=glob ...` the completed shards are graded in the `lcb-grader` Docker sandbox at each cap of `--caps` (default 16K and 32K) and the arms are compared with a paired exact McNemar test |
| `test_plane_kernels.py` | kernel tests of the plane's fast prefill kernels (`ple_rows`, `pool_keys`, `transpose_f32`) against the research implementations |
| `run_with_ram_budget.py` | runs a command (for example `bench_engine_mtp.py`) with a bounded host expert cache for a QMoE-offload package: `$ORT_QMOE_HOST_CACHE_MIB` / `$ORT_QMOE_HOST_CACHE_LAYERS` from `--ram-budget-gib` minus `--reserve-gib`, and with `--enforce` a `systemd-run --user` cap on the whole command; `test_ram_budget.py` is its unit test (`python -m unittest test_ram_budget`) |

A qualification pass of a package: `engine_gate.py` with `--force-reject --prefill steps` (record), then without `--force-reject`
with `--reference` of that recording (MTP stream == no-MTP stream); `engine_behaviours.py --prefill steps`; `bench_engine_mtp.py` on
the fixed-length package; `bench_ttft_engine.py`; `engine_prompt_lengths.py` with both prefill modes; and, for accuracy,
`mmlu_engine_eval.py` and `lcb_engine_eval.py`.

The provider-side settings the numbers in the PR descriptions were taken with are
`CUDA_MODULE_LOADING=EAGER ORT_ENABLE_FP4_GEMV=1 ORT_NVFP4_FC1_EXPERT_REUSE=0 ORT_FORCE_DETERMINISTIC_MOE=1`
(provider-branch switches; they are not read by GenAI).

The logic that does not need a GPU is covered by `test/cpp/engine/qwen38_graph_*_tests.cpp` (scripted device, real Engine); the pure logic of
the qualification scripts (acceptance histogram, calibration output names, the MMLU prompt check and `--compare`) by
`test/python/builder/test_qwen38_qualification_tools.py` (`cd test/python/builder && python -m pytest -q test_qwen38_qualification_tools.py`).
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
