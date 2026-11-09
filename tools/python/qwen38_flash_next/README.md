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
| natural-text pools for the benchmark and gate prompts | any natural text of at least ~40k tokens, as `--text-pool` (and `--prose-pool` for the gate's `prose` case; or `$QWEN38_TEXT_POOL` / `$QWEN38_PROSE_POOL`): a text file or a `.npy` of token ids |
| reference streams for the identity gate | a recording of the same prompts from another runner, or of this backend with `--no-mtp` / `--force-reject` (see `engine_gate.py`) |
| research inputs of `mmlu_engine_eval.py`, `lcb_engine_eval.py` and `test_plane_kernels.py` | the research helper modules (`eval_lib`, `eval_mcq_genai`; `$QWEN38_EVAL_PATHS` or `PYTHONPATH`), the MMLU-Pro and LiveCodeBench prompts, samples and tests, an HF checkpoint folder, and for the kernel tests `mobius`, `torch` and the research runner; the other scripts need none of them |

## Package

A package is a model folder: `genai_config.json` with `model.qwen38_graph`, `target/`, `mtp_head/`, `libs/`, `ple/` +
`ple_table.txt` and the tokenizer files; the v12b stack below adds its own graphs. Every script takes the folder as its first
argument and overlays config values for the run: `--no-mtp` is `mtp.enabled=false` (the plain S=1 graph decode) and
`--prefill steps|fast` is `prefill.enabled`. Config overlays append to `eos_token_id`, so a fixed-length benchmark ("ignore EOS")
needs its own package whose stop id is an ordinary id that is never generated (`248319`).

`--no-mtp` requires a non-folded package with compatible S=1 graphs. The tools reject this mode when the effective
`fold.enabled` is true (including v12b/rc4/rc5 folded packages). For a folded package's greedy identity reference, record
`engine_gate.py --force-reject`; forced rejection keeps the folded execution path and does not measure S=1 performance.

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
otherwise.

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

Calibration registers each dense/head-KV session's configured `custom_ops_library` and `custom_ops_libraries` before loading
its graph, resolving relative paths from the package and falling back to the decoder's libraries if neither field is set.
Graphs that take `inputs_embeds` run the package's CPU `model.embedding` graph with its configured input/output names:
dense prefill receives the prompt embeddings, and head-KV receives the shifted embeddings for tokens 1 through N-1.
The embedding graph is not loaded when both calibration graphs take `input_ids`.
Both output quantizations use `--method minmax` or `percentile`; MSE is rejected because its integer rounding objective does
not model FP8. Incomplete K/V layer pairs and non-finite captures, scales or statistics are rejected before either scale file
is written.

The gate requires an explicit `--prefill steps|fast` with `--reference`, and matching prompt hashes whenever the reference
provides them, in addition to checking prompt lengths. Prompt-length comparisons require equal, non-empty length sets.
MMLU and LiveCodeBench resume protocols record the parsed `$QWEN38_OVERLAY_JSON` and any launcher host-expert byte budget;
MMLU also records `--force-reject`, while LiveCodeBench records the inherited `$ORT_QWEN38_FORCE_REJECT` hook. Changing these
settings requires a fresh output. Older protocol files without these fields also require a fresh output.
LiveCodeBench rejects prompts that leave no generation budget and requires disjoint, complete grading shards with matching
`.protocol.json` metadata (apart from the shard index). Evaluation wall time ends at turn completion, including EOS-first
turns; when the stop token is appended to the recorded count, decode timing includes the terminal EOS run.
Grading still saves accuracy when no decode interval, acceptance rounds or delivered first token can be measured:
unavailable aggregate performance metrics are JSON null.

`engine_behaviours.py` requires the single-slot baseline (`configured_max_batch_size=1`, `max_request_length=33024`); it rejects
other configurations before running probes. Its 20K prefill, 40K capacity and sequential queueing probes assume that baseline.
The decode benchmark requires `--new` to be at least 2 and reports cycle timing without speculative acceptance statistics
when no drafts were proposed. Stop-string probes use fresh tokenizer streams so split UTF-8 tokens match the Engine's text,
and lost-residency probes require the specific "no longer resident" diagnostic rather than accepting arbitrary exceptions.
Completed turn usage counters are owned snapshots, not references to the reused Engine event buffer.
`bench_ttft_engine.py` raises an explicit benchmark error if EOS arrives before any token is delivered; `smoke.py` reports the
completed zero-token turn with unavailable latency/throughput and raises on failed or capacity-blocked Engine events.

| script | purpose |
| --- | --- |
| `qwen38_engine_util.py` | helpers shared by the scripts: CUDA EP registration and model loading with config overlays (`$QWEN38_OVERLAY_JSON`), the natural-text prompt pools, and `run_turn` (one greedy turn through `og.Engine`); registers the DLL directories on Windows |
| `calibrate_qwen38_kv.py` | calibrate per-channel INT8 and FP8 KV-cache scales for the 12 target sparse-attention layers and 1 MTP-head sparse-attention layer; `--out` receives the `--quant-type` scales and the other quantization is written beside it under the name with the quantization token swapped (`kv_scales_int8_per_channel.json` -> `kv_scales_fp8_per_channel.json`, `scales.json` -> `scales.fp8_per_channel.json`), so the two files never collide (a collision is an error before any work) |
| `smoke.py` | minimal example, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token-identity gate: greedy streams of the gate prompts against a `--reference` (a recording of `--no-mtp` or `--force-reject`, i.e. MTP == no-MTP, or another runner's streams); `--prefill steps` must be identical, `--prefill fast` is checked for an equal first token, equal prompt length and an identical prefix of at least `--min-fast-prefix` tokens and reported as agreement statistics; EOS mode and `--ignore-eos` fixed length |
| `engine_behaviours.py` | (`--prefill steps\|fast`) stop tokens, limits, stop strings, cancel during prefill/decode, second request queued, continuation, rewind, capability errors, statistics |
| `bench_engine_mtp.py` | vLLM-protocol decode benchmark (exact prompt length, 256 new tokens, 1 warm-up + 3 runs, TPS = 255 tokens / time between first and last token); the acceptance statistics are the difference of the two full acceptance-length histograms taken around the run (rounds of different draft widths keep their own bins), and an inconsistent snapshot is an error |
| `bench_ttft_engine.py` | time to first token at several prompt lengths (+ optional MMLU-Pro prompt ids from `--mmlu-prompts`): begin_turn to the Run that delivers the token |
| `engine_prompt_lengths.py` | `run` the same greedy turns for 1 .. 20000-token prompts (dense chunk boundaries 8191/8192/8193, ...) with `--prefill steps` and `--prefill fast`, then `compare` (first token equal, shared prefix) |
| `mmlu_engine_eval.py` | MMLU-Pro 800 through the Engine (shards, resumable) and the MTP == no-MTP stream check (`--check-items`, `--force-reject`, `--compare`); a run stops before generating when an Engine prompt differs from the canonical prompt ids/text, and `--compare A B` (A an MTP run, B a `--force-reject` run) exits non-zero unless both cover the same items and every stream is identical |
| `lcb_engine_eval.py` | LiveCodeBench v6 (175 problems) through the Engine, generated in resumable `--shard K --shards N` runs; with `--grade --arms name=glob ...` the completed shards are graded in the `lcb-grader` Docker sandbox at each cap of `--caps` (default 16K and 32K) and the arms are compared with a paired exact McNemar test |
| `test_plane_kernels.py` | kernel tests of the plane's fast prefill kernels (`ple_rows`, `pool_keys`, `transpose_f32`) against the research implementations; uses ABI 3 and checks the PLE CUDA return status before inspecting outputs |
| `run_with_ram_budget.py` | runs a command (for example `bench_engine_mtp.py`) with a bounded host expert cache for a QMoE-offload package: the legacy `$ORT_QMOE_HOST_CACHE_MIB` / `$ORT_QMOE_HOST_CACHE_LAYERS` and `$QWEN38_HOST_EXPERT_BYTES` from `--ram-budget-gib` minus `--reserve-gib`; the shared model loader applies the byte budget to decoder session options and all runtime-profile overrides. With `--enforce`, a `systemd-run --user` cap bounds the whole command; `test_ram_budget.py` is its unit test (`python -m unittest test_ram_budget`) |

A qualification pass of a package: `engine_gate.py` with `--force-reject --prefill steps` (record), then without `--force-reject`
with `--prefill steps --reference` of that recording (MTP stream == no-MTP stream); `engine_behaviours.py --prefill steps`; `bench_engine_mtp.py` on
the fixed-length package; `bench_ttft_engine.py`; `engine_prompt_lengths.py` with both prefill modes; and, for accuracy,
`mmlu_engine_eval.py` and `lcb_engine_eval.py`.

The RAM launcher's session-option budget applies to commands that use `qwen38_engine_util.load_model`. It takes precedence
over package, environment-overlay and CLI-overlay budgets, including runtime profiles. The loader uses a temporary config
with links to the original assets and keeps it alive for the process; it never edits the original package.
Creating the config view requires symlink support (on Windows, Developer Mode or the usual symlink privilege).
Other commands must apply `session.moe.offload.host_expert_bytes` themselves or use `--enforce` for a total process limit.
`--ram-budget-gib 0` clears the launcher override and leaves the package's session options unchanged.

The provider-side settings the numbers in the PR descriptions were taken with are
`CUDA_MODULE_LOADING=EAGER ORT_ENABLE_FP4_GEMV=1 ORT_NVFP4_FC1_EXPERT_REUSE=0 ORT_FORCE_DETERMINISTIC_MOE=1`
(provider-branch switches; they are not read by GenAI).

The logic that does not need a GPU is covered by `test/cpp/engine/qwen38_graph_*_tests.cpp` (scripted device, real Engine); the pure logic of
the qualification scripts (acceptance histogram, calibration output names, the MMLU prompt check and `--compare`) by
`test/python/builder/test_qwen38_qualification_tools.py` (`cd test/python/builder && python -m pytest -q test_qwen38_qualification_tools.py`).
`test/python/builder/test_qwen38_calibration_embeddings.py` additionally runs tiny CPU ONNX embedding graphs (including BF16)
and checks dense/head-KV token and embedding bindings without a GPU or research checkpoint.
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

---

# Qwen3.8-Flash-Next (`qwen38_flash_graph`) Package Builder

`tools/python/qwen38_flash_next/build_package.py` procedurally constructs self-contained ONNX Runtime GenAI model packages for **Qwen3.8-Flash-Next** (`model.type: "qwen38_flash_graph"`) + **DFlash2** (`S=8`, 7-draft speculative decoding) directly from the two Hugging Face checkpoints (`Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-DFlash2`), without depending on any pre-exported `.onnx` model.

Every generated package supports:
- **Single unified `model.onnx` across all context lengths (`32K` / `128K` / `256K`)**: cache capacity dimensions (`kv_capacity` and `state_capacity`) are symbolic and `contrib_const_dsa_total_seq` is set to `[262152]`, so a single `model.onnx` serves `32K` (`num_blocks: 142`), `128K` (`num_blocks: 525`), and `256K` (`num_blocks: 1037`) without duplicate `model_128k.onnx` / `model_256k.onnx` files.
- **Pure GPU execution** up to **256K (`262,144`) context length** on H200 / high-VRAM GPUs (`gpu-128k`, `gpu-256k` -> `model.onnx`).
- **QMoE CPU/GPU hybrid offload** on 20–36.5 GB consumer GPUs (`consumer-gpu-24g`, `consumer-gpu-24g-host-64g`, `consumer-gpu-30g-ram-48g-32k`, `consumer-gpu-32g` -> `model.onnx` + `expert-profile.bin`).
- **Authoritative protocol boundary token IDs** (`bot_token_id: 248058`, `eot_token_id: 248059`, `bor_token_id: 248068`, `eor_token_id: 248069`) and **Engine dynamic batching** (`max_batch_size: 1`, `max_scheduled_tokens: 8192`).

---

## 1. Building Predefined RC5 Packages from the Hugging Face Checkpoints

```bash
# Build all 4 core RC5 packages directly from /data/tlwu/Qwen3.8-Flash-Next and /data/tlwu/Qwen3.8-Flash-Next-DFlash2:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --models-root /home/tianlei/models \
  --link-root /data/tlwu \
  --presets rc5_all

# Or build a single predefined preset (e.g., 2b375_x02_y14_z32_dp23_b64):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --out /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64
```

| Preset Name | Alias | `(x INT4, y INT3, z INT2)` | KL Mode | Avg Bits | QMoE Size | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | Min VRAM (`gpu-128k` / `gpu-256k`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2` | `all_int4` | `(48, 0, 0)` | `rank` | `4.000` | `63.28 GiB` | `75.42 GiB` (`80.99 GB`) | `77.90 GiB` (`83.65 GB`) | `83 GB` / `87 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64` | `top16` | `(16, 32, 0)` | `rank` | `3.333` | `51.56 GiB` | `64.79 GiB` (`69.57 GB`) | `67.63 GiB` (`72.62 GB`) | `72 GB` / `76 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64` | `top08` | `(8, 40, 0)` | `rank` | `3.167` | `48.63 GiB` | `61.87 GiB` (`66.43 GB`) | `64.71 GiB` (`69.48 GB`) | `69 GB` / `73 GB` |
| `qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64` | `2b375` | `(2, 14, 32)` | `dp23` | `2.375` | `37.06 GiB` | `50.30 GiB` (`54.01 GB`) | `52.00 GiB` (`55.84 GB`) | `57 GB` / `61 GB` |

---

## 2. Building Custom `(x INT4 + y INT3 + z INT2)` QMoE Partitions Sorted by KL Similarity

```bash
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  -x 2 -y 14 -z 32 --kl-mode dp23 \
  --out /home/tianlei/models/qwen_flash_next_custom_2_14_32
```

---

## 3. Exporting Zero-Copy GGUF Model Packages (`build_package_gguf.py`)

`tools/python/qwen38_flash_next/build_package_gguf.py` exports `unsloth/Qwen3.8-Flash-Next-GGUF` quantization variants (`UD-IQ1_S`, `UD-IQ2_XS`, `UD-IQ3_XXS`) and the `IQ4_NL` n-gram (`engram`) GGUF directly to `qwen38_flash_graph` packages using `GatherPacked`, `MatMulPacked`, and `MoEPacked`, sharing dynamic cache surgery (`ensure_unified_dynamic_cache_model`), protocol boundary tokens, and config migration helpers with `build_package.py`:

```bash
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64 \
  --out-root /data/tlwu \
  --variants UD-IQ1_S,UD-IQ2_XS,UD-IQ3_XXS
```

| GGUF Variant | Total GPU Weights | 32K Peak VRAM (`num_blocks=142`, min `141`) | 64K Peak VRAM (`num_blocks=270`, min `269`) | 128K Peak VRAM (`num_blocks=525`) | 256K Peak VRAM (`num_blocks=1037`) | `runtime_profiles` Min VRAM (`gpu-32k` / `gpu-64k` / `gpu-128k` / `gpu-256k`) |
|---|---:|---:|---:|---:|---:|---:|
| **`UD-IQ1_S`** | `35.66 GiB` (`38.29 GB`) | `41.39 GiB` (`44.45 GB`) | `41.74 GiB` (`44.82 GB`) | `42.85 GiB` (`46.00 GB`) | `45.54 GiB` (`48.90 GB`) | `47 GB` / `48 GB` / `49 GB` / `52 GB` |
| **`UD-IQ2_XS`** | `44.46 GiB` (`47.74 GB`) | `50.14 GiB` (`53.84 GB`) | `50.49 GiB` (`54.21 GB`) | `51.59 GiB` (`55.39 GB`) | `54.28 GiB` (`58.28 GB`) | `56 GB` / `57 GB` / `58 GB` / `61 GB` |
| **`UD-IQ3_XXS`** | `51.73 GiB` (`55.55 GB`) | `57.44 GiB` (`61.68 GB`) | `57.79 GiB` (`62.05 GB`) | `58.89 GiB` (`63.24 GB`) | `61.59 GiB` (`66.13 GB`) | `64 GB` / `65 GB` / `66 GB` / `69 GB` |

---

## 4. Component Precision Options (`precision_options.py`)

Both `build_package.py` and `build_package_gguf.py` accept `--precision-profile {default,fp32,fp32_fp16kv,fp32_fp16kv_v2,fp32_int8kv_v2}` as well as individual per-component flags (`--kv-cache-dtype`, `--rotary-cache-dtype`, `--qsa-qkv-dtype`, `--indexer-dtype`, `--gdn-gate-dtype`, `--gdn-precision`, `--ple-engram`, `--ple-norm-dtype`, `--small-weights-source`, `--emit-int8kv-variant`, `--consolidate-data-files`):

- **Dual-ONNX (`model.onnx` + `model_int8kv.onnx`) & Consolidated External Data (`model.onnx.data`)**:
  - With `--precision-profile fp32_fp16kv_v2` (or `--emit-int8kv-variant`), the builder emits **both** `model.onnx` (`FP16` KV cache) and `model_int8kv.onnx` (`INT8` KV cache with inline `node_GuardedDSA_*_{k,v}_scale` initializers) in the same package directory, sharing the exact same external weight files.
  - `genai_config.json` `runtime_profiles` automatically select `"filename": "model_int8kv.onnx"` for VRAM-constrained offload profiles (`consumer-gpu-24g`, `consumer-gpu-32g`, etc., where `session.moe.offload.mode == "required"`) and inherit `"model.onnx"` (`FP16` KV cache) for full-VRAM profiles (`gpu-128k`, `gpu-256k`).
  - With `--consolidate-data-files` (`consolidate_package_external_data`), all target decoder external `.data` files are merged into a single `4096`-byte page-aligned `model.onnx.data` file (compatible with `PosixEnv::MapFileIntoMemory` and `PromoteFileMappingToSharedReadOnly` for QMoE offload), leaving only 3 `.data` files in the package (`embedding.onnx.data`, `dflash2.onnx.data`, and `model.onnx.data`).

```bash
# Build rc5_top16_fp32_fp16kv_v2, rc5_all_int4_fp32_fp16kv_v2, and rc5_2b375_fp32_fp16kv_v2
# (emits both model.onnx [FP16 KV] and model_int8kv.onnx [INT8 KV] + consolidated model.onnx.data):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset top16_44_33_b64,qwen_flash_next_int4_int8_rc5_contrib_dflash2,2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32_fp16kv_v2 \
  --consolidate-data-files \
  --out /data/tlwu/qwen_flash_next_rc5_v2

# Build rc5_all_int4_fp32_fp16kv (FP16 KV cache + FP32 rotary cache + FP32 GDN/indexer/PLE norms + Q8_0 engram):
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset qwen_flash_next_int4_int8_rc5_contrib_dflash2 \
  --precision-profile fp32_fp16kv \
  --out /data/tlwu/qwen_flash_next_rc5_all_int4_fp32_fp16kv

# Build rc5_2b375 with BF16 KV cache + FP32 GDN/indexer/PLE norms:
python tools/python/qwen38_flash_next/build_package.py \
  --hf-dir /data/tlwu/Qwen3.8-Flash-Next \
  --dflash-dir /data/tlwu/Qwen3.8-Flash-Next-DFlash2 \
  --preset 2b375_x02_y14_z32_dp23_b64 \
  --precision-profile fp32 \
  --out /data/tlwu/qwen_flash_next_rc5_2b375

# Build GGUF UD-IQ3_XXS and UD-IQ2_XS with Q8_0 engram, FP32 long-context path, and FP16 KV cache:
python tools/python/qwen38_flash_next/build_package_gguf.py \
  --gguf-root /data/tlwu/qwen_flash_next_ggufs \
  --base-pkg /data/tlwu/qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64_unified \
  --out-root /data/tlwu \
  --variants UD-IQ3_XXS,UD-IQ2_XS \
  --precision-profile fp32_fp16kv \
  --pkg-suffix _q8_engram_fp32_fp16kv
```

