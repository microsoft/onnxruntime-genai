# Qwen3.8-Flash-Next NVFP4: CUDA-graph decode + MTP(1) in ONNX Runtime GenAI

Two stacks run through the same model type:

* **v12 (current)**: fold decode (the MTP head is inside the verify graph: one graph launch per cycle), composed v12 kernels
  (attn r3, HC r3, gemm r4, DSA r2 + memset kernel), merged provider `b6a99bf4...`, **fast prefill** (dense chunked prefill session,
  dense -> packed state hand-off, batched KV-only head prefill, one device copy of the shared weights). Package: `build_package_v12.py`,
  section 6 below.
* v11 (previous): S=2 verify graphs + separate S=1/S=2 head graphs, prompt through S=2 graph steps (3.9 ms/token). Package:
  `build_package.py`, sections 1-5. Unchanged and still supported.

`model.type = "qwen38_flash_graph"` loads the research CUDA-graph stack (target S=1/S=2 graphs, MTP head graphs, CUDA data plane,
host-registered PLE table) as one GenAI model. Users keep the normal API: `og.Model`, `og.Tokenizer`, `og.Generator`
(`append_tokens`, `generate_next_token`, `get_next_tokens`, `is_done`, `get_sequence`). Decode is greedy and lossless:
the tokens are identical to no-MTP greedy decode of the same graphs. One token is returned per `generate_next_token()`;
an accepted MTP draft produces two tokens from one verify cycle, the second is returned by the next call.

## 1. Build

Same conventions as the other GenAI worktrees (Unix Makefiles, isolated ORT home, CUDA 13, sm_90):

```bash
export PATH=/home/tianlei/cuda13.0/bin:/home/tianlei/git/venv/bin:$PATH
# ort_home/include = ORT headers, ort_home/lib = libonnxruntime.so(.1.31.0) of the provider runtime you will run with
cmake -S . -B build/cuda130/Release -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90-real -DCMAKE_CUDA_COMPILER=/home/tianlei/cuda13.0/bin/nvcc \
  -DORT_HOME=<ort_home> -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/home/tianlei/git/venv/bin/python \
  -DENABLE_TESTS=OFF -DENABLE_TELEMETRY=OFF -DENABLE_MODEL_BENCHMARK=OFF -DUSE_GUIDANCE=OFF
make -j90 onnxruntime-genai python qwen38_plane
```
Outputs: `build/cuda130/Release/wheel/onnxruntime_genai` (Python package: put it on `PYTHONPATH`) and
`build/cuda130/Release/libqwen38_plane.so` (the data-plane library the package builder links into the model folder).
The Python package needs an `onnxruntime/capi/libonnxruntime.so*` on `PYTHONPATH` (GenAI's loader finds the ORT shared library through
the `onnxruntime` package); `eval_runs/qwen38_genai_mtp_20261012_01/env.sh` shows the isolated setup (ORT 1.31 + the S=2 CUDA plugin provider,
SHA256 `53cb2dab...`, loaded through `og.register_execution_provider_library("CUDAExecutionProvider", <provider.so>)`).

## 2. Package

```bash
python tools/qwen38_graph/build_package.py --out <pkg>            # needs the research Python env (mobius) for ple_table.txt
python tools/qwen38_graph/build_package.py --out <pkg_fixed> --stop-token-ids 248319   # benchmark package: EOS is an ordinary token
```
Everything is hard-linked from the research bundles (`--target-bundle`, `--head-bundle`, `--lib-*`, `--plane-lib`, `--checkpoint` select
other inputs; `--copy` copies instead of linking). Layout: `genai_config.json`, `target/` (v11 S=2 bundle), `mtp_head/`, `libs/*.so`,
`ple/` (safetensors holding the FP8 PLE shards) + `ple_table.txt`, tokenizer files. Custom-op libraries must live under the model folder
(GenAI's allow-list), which is why they are linked into `libs/`; to try new kernels build new `.so` files and rebuild the package.

`genai_config.json` additions (see `src/config.h`):

| key | meaning |
| --- | --- |
| `model.decoder.session_options.custom_ops_libraries` | list of custom-op libraries registered for a session (besides `custom_ops_library`) |
| `model.qwen38_graph.head.{filename_s1,filename_s2,session_options}` | MTP head sessions (S=1 draft, S=2 refeed); head options inherit the decoder's providers/entries |
| `model.qwen38_graph.plane_library`, `ple_table` | data-plane library (dlopen, must be inside the folder) and PLE table |
| `model.qwen38_graph.kv_capacity` | fixed KV capacity of the graphs (33028); `context_length` is capacity-4 |
| `model.qwen38_graph.{target,head}_s{1,2}_graph_id` | first CUDA-graph annotation ids (3 target phases, 2 head phases per row count) |
| `model.qwen38_graph.mtp.{enabled,prefill_head,skip_refeed}` | MTP on/off (off = plain S=1 graph decode), prefill the head with the prompt, skip-refeed experiment |
| `model.qwen38_graph.warmup` | capture all graphs when the model is created (a few seconds) |

## 3. Use

```python
import json, onnxruntime_genai as og
og.register_execution_provider_library("CUDAExecutionProvider", PROVIDER_SO)   # plugin CUDA EP (required)
model = og.Model(og.Config(PKG))                                                  # ~30 s: 74 GB weights, PLE pin, warm-up
tok = og.Tokenizer(model)
ids = tok.encode(tok.apply_chat_template(json.dumps([{"role": "user", "content": "Hello"}]), add_generation_prompt=True))
params = og.GeneratorParams(model); params.set_search_options(max_length=len(ids) + 256, do_sample=False)
gen = og.Generator(model, params)
gen.append_tokens(ids)                     # chunked appends are fine before the first generate_next_token
while not gen.is_done():
    gen.generate_next_token()
    print(tok.decode(gen.get_next_tokens()), end="")
print(gen.get_speculative_stats())         # rounds, draft_tokens_accepted, ...
```
Environment of the runtime (set by `env.sh`): `ORT_ENABLE_FP4_GEMV=1 ORT_NVFP4_FC1_EXPERT_REUSE=0 ORT_FORCE_DETERMINISTIC_MOE=1`,
`CUDA_MODULE_LOADING=EAGER`, one provider path per process.

## 4. Limitations (all raise a clear `RuntimeError`)

* greedy only (`do_sample=false`, no repetition penalty / min_length / beams / guidance / `set_logits`), batch size 1.
* one active `Generator` per model (the CUDA-graph state, 74 GB of weights and the captured graphs are owned by the model);
  creating a new generator after deleting the previous one is cheap and keeps the graphs.
* no continuous decoding: `append_tokens` after `generate_next_token`, and `rewind_to(n>0)`, are rejected (recurrent GDN/conv/PLE/indexer state
  cannot be cropped to an arbitrary token; the loop also looks one cycle ahead). `rewind_to(0)` resets. Chunked prefill before the first
  `generate_next_token` is supported.
* `max_length <= 33024`; prompt prefill uses S=2 graph steps (about 3.9 ms/token) through `Qwen38GraphEngine::PrefillChunk`, the single
  entry point to replace with a faster prefill.
* the stop tokens come from `genai_config.json` (`<|im_end|>`, `<|endoftext|>`); the stop token is returned by `get_next_tokens()` but not
  appended to `get_sequence()` (standard GenAI behaviour). A fixed-length benchmark needs a package with an unreachable stop id
  (config overlays append to the stop list, they cannot remove entries).

## 5. Tests

`smoke.py` (usage example), `test_api.py` (API behaviour: sequence/EOS/rewind/lease/unsupported options), `gate_genai.py` (token identity vs
the research no-MTP reference hashes, chat template), `bench_genai_mtp.py` (vLLM protocol TPS 2k/8k/32k). Results:
`eval_runs/qwen38_genai_mtp_20261012_01/REPORT.md`.

## 6. The v12 stack (fold decode + fast prefill)

### 6.1 Package

```bash
source eval_runs/qwen38_genai_mtp_20261012_01/env_v12.sh              # GPU=4 default; merged provider b6a99bf4... (one provider path per process)
$PYTHON tools/qwen38_graph/build_package_v12.py --out <pkg>            # hard links, ~7 s, no extra disk; --stop-token-ids 248319 for benchmarks
$PYTHON tools/qwen38_graph/build_package_v12.py --out <pkg> --no-prefill   # fold decode only (prompt through the muted fold graph, S=2 steps)
```
Every input is an argument: `--fold-bundle` (fold model, `mtp_runner/fold/bundle/fold_v12all/target`), `--head-r2-bundle` (standalone R=2 head),
`--dense-bundle` (dense prefill model), `--head-kv-bundle` (KV-only head), `--lib-qsa|fused-ops|fused-attn|fused-hc|fused-gemm|dsa|mtp-head|fold`,
`--plane-lib`, `--chunk`. Swapping a kernel library or bundle = rebuilding the package (links only; the custom-op libraries must live inside the model
folder). Layout: `genai_config.json`, `target/` (fold model), `head_r2/`, `dense/`, `head_kv/`, `libs/*.so`, `ple/` + `ple_table.txt`,
`weights_pool.txt` (shared initializers: 625 tensors, 72.2 GiB), `qsa_consts.{txt,bin}` (hand-off constants), tokenizer files.

`genai_config.json` additions (see `src/config.h`):

| key | meaning |
| --- | --- |
| `model.qwen38_graph.fold.{enabled,target_graph_id}` | fold stack: `model.decoder` is the fold model (verify S=2 + head + glue), graph ids `target_graph_id + ring phase` (71-73) |
| `model.qwen38_graph.fold.head_r2.{filename,graph_id,session_options}` | standalone R=2 head (graph ids 81-83): prompt context by pairs (when no fast prefill) and the first draft |
| `model.qwen38_graph.prefill.enabled` | fast prefill on/off (overlay `{"model":{"qwen38_graph":{"prefill":{"enabled":false}}}}` = prompt through the muted fold graph) |
| `model.qwen38_graph.prefill.{dense,head_kv}.{filename,session_options}` | dense prefill model (contrib ops, no graph capture) and KV-only head model |
| `model.qwen38_graph.prefill.chunk` | dense prefill chunk (8192); `warmup_lengths`: prompt lengths run once at model load (arena growth, first-use costs) |
| `model.qwen38_graph.prefill.weights_pool` | text manifest of the initializers that several sessions share: loaded once into CUDA memory (8 threads, pinned staging) and passed to every session with `AddInitializer` (float8 handled by the C API) |
| `model.qwen38_graph.prefill.constants` | key-norm weights and rotary tables of the PackedQSAIndexer state conversion |

### 6.2 What runs where

* **Decode.** One fold graph (ids 71-73) per cycle: verify S=2 body, accept test, head R=2 chain, next draft. The host enqueues the two `k_prep` rows
  and one graph launch, then spins on the pinned result record that the graph publishes mid-cycle (`Qwen38GraphEngine::EnqueueFold/WaitCycle/Advance`).
  `Qwen38GraphStrategy` is unchanged (1-2 tokens per `generate_next_token`).
* **Prefill** (`Qwen38GraphEngine::FastAppend`, called from `AppendTokens`): PLE rows of the chunk by a device kernel (`mtp_ple_rows`), dense session over
  chunks of 8192 (`logits_indices` = last row, hidden rows written straight into the head-context buffer), host argmax of the last logits row.
  **Chunked `append_tokens` before the first `generate_next_token` is supported** (the dense state continues across calls). At the first
  `generate_next_token`: hand-off (`FastHandoff`: KV rows -> fixed store, GDN state transposed into ring bank 0, conv/PLE state copied, raw index keys
  -> pooled compressed keys + pending rows + lengths by `mtp_pool_keys`), batched head KV prefill (`FastHeadKvPrefill`), first draft (`FoldFirstDraft`).
* **Warm-up** at model load (`model.qwen38_graph.warmup`): muted fold steps (capture of graph ids 71-73), head R=2 graphs 81-83, then one dummy prompt per
  `prefill.warmup_lengths` through dense + hand-off + head KV. About 2-3 s.

### 6.3 Properties and limits

* Decode is lossless: with `prefill.enabled=false` the tokens are identical to the research no-MTP greedy hashes (12/12) and to the Python fold runner (12/12 benchmark runs: tokens and cycle counts). Measured (H200, merged provider): 244 / 240 / 231 TPS at 2k / 8k / 32k natural text (7.74 / 7.92 / 8.11 ms per cycle, equal to the Python fold runner), TTFT 0.30 / 0.76 / 3.2 s at 2k / 8k / 32k, 0.13 s for 250-token prompts.
* The fast prefill is **tolerance level** (dense vs packed numerics; the dense grouped GEMM is not bit-reproducible between runs, so two runs of the same
  prompt - in this port and in the Python `PrefillEngine` - can diverge after tens of tokens). First tokens agree; see REPORT_V12.md.
* Greedy only, batch 1, one active generator per model, no `append_tokens` after `generate_next_token`, `rewind_to(0)` only; `max_length <= 33024`.
* GPU memory: 113.4 GiB (116 GB) from the end of model load (the warm-up of the 8192-token chunk reserves the dense arena) through 32k-context decode (141 GB card). The dense session arena is not released between requests.
* Tests: `test_plane_kernels.py` (kernel level vs the python/torch implementations), `gate_v12.py` (token gates), `bench_ttft_genai.py`,
  `bench_genai_mtp.py`, `test_api.py`.


## 7. The v12b stack (accurate bf16 Sigmoid)

v12b = the v12 package with the CUDA EP provider `a5ce286d...` (accurate bf16 `Sigmoid`, branch `qwen38-prefill-fix` 453ac97a68) and the fused libraries
`libqwen_fused_{attn,hc,gemm}_r5.so` (research report `qwen38_cuda_graph_20261010_01/REPORT_V12B.md`). Bundles, head / fold / DSA / indexer libraries and the GenAI
code are unchanged; the v12 libraries and provider emulated the buggy sigmoid at two sites, v12b is accurate at all six.

```bash
source eval_runs/qwen38_genai_mtp_20261012_01/env_v12b.sh                 # GPU=4 default; QWEN_PROVIDER = <pkg>/provider/libonnxruntime_providers_cuda.so
$PYTHON tools/qwen38_graph/build_package_v12b.py --out <pkg>               # = build_package_v12.py with the r5 libraries + the fixed provider linked into <pkg>/provider/
$PYTHON tools/qwen38_graph/build_package_v12b.py --out <pkg_fixed> --stop-token-ids 248319     # benchmark package (EOS ignored)
```
`build_package_v12b.py` takes every argument of `build_package_v12.py` plus `--v12b-dir` (research work dir with `lib/*_r5.so`, `runtime_fixed/`) and `--provider`.
The provider is a plugin library, so one process = one provider path: `og.register_execution_provider_library("CUDAExecutionProvider", "<pkg>/provider/libonnxruntime_providers_cuda.so")`
(`smoke.py`, `test_api.py`, ... read it from `$QWEN_PROVIDER`). The GenAI Python / C++ package and `libqwen38_plane.so` are the v12 ones (the same `libonnxruntime.so` 1.31 sits next to the provider).

Test hook: `model.qwen38_graph.fold.force_reject` (overlay `{"model":{"qwen38_graph":{"fold":{"force_reject":true}}}}`, default false) replaces the draft of every cycle by a different token
before the verify (the python `draft_hook`), so every cycle commits exactly the greedy token of verify row 0 = no-MTP greedy decode of the same graph (about 3.8 ms per token slower, a host sync per cycle).
It is what `mmlu_genai_eval.py --check-items ... --force-reject` uses to check that the MTP(1) stream equals the plain greedy stream (use `prefill.enabled=false` for a bit-exact common start state).
`gate_v12.py --no-research-hash` gates against the python v12b runner only (the stored research hashes predate the v12b numerics).

## 8. Experimental generic-port performance package

`optimize_ported_package.py` builds a separate package from the generic-QSA/DSA port.
The source and original runnable packages are not modified. It:

* precomputes constant hyper-connection untiling and reuses `HyperConnectionLowRankMix`,
  including its optional identity post-mix prologue and residual-stream output;
* replaces eligible constant FP8 dequantize/cast/matmul chains with `MatMulBlockQuantizedFp8Weight`;
* sends INT8 BNSH caches and per-channel scales directly to `DynamicSparseAttention`, sharing
  dequantized KV tiles across grouped query heads for one- and two-token decode;
* retains floating-point dense-prefill outputs for subsequent chunks and adds separate quantized handoff outputs;
* folds eligible projection channel slices, layout changes, and final/first-token carry updates into
  the existing `CausalConvWithState` CUDA operator;
* refreshes live shared-weight bindings and copies the matching ORT/CUDA-provider/GenAI binaries into the new package.

Build the changed ORT and GenAI sources first, then run from the GenAI checkout:

```bash
python tools/qwen38_graph/optimize_ported_package.py <generic-ported-package> <new-package> \
  --provider <ORT-build>/libonnxruntime_providers_cuda.so \
  --runtime <ORT-build>/libonnxruntime.so \
  --genai-library <GenAI-build>/libonnxruntime-genai.so
```

Use the rebuilt GenAI Python extension. Set `ORT_LIB_PATH=<new-package>/runtime/libonnxruntime.so`
and put `<new-package>/runtime` and the GenAI build directory on `LD_LIBRARY_PATH`
(alongside the CUDA libraries). Register `<new-package>/provider/libonnxruntime_providers_cuda.so`.
On systems requiring CUDA forward compatibility, retain the compatible driver-library directory
ahead of the other CUDA libraries.

Set `ORT_QMOE_NVFP4_PACKED_PREFILL=1` for the existing packed NVFP4 prefill path.
Otherwise this runtime can fall back to dequantizing all MoE expert weights during prefill.
The package enables CUDA graphs **and model warmup** together. Do not disable warmup with capture enabled:
capture setup repeats graph runs, and the folded state updates must be muted during those runs.
Provider-option overlays append entries; clear and rebuild the provider list when overriding capture.
Dense prefill uses 512-token chunks to bound scratch memory.

### Measured partial recovery

H200, batch 1, greedy MTP(1), 512 generated tokens, one warmup and three measured requests
per prompt length. Ported results use the public `Generator` API; the historical pre-port
baseline used the original `Engine` API. They are not a same-API kernel-only comparison.
The v5 persisted package and copied runtime binaries were used on an otherwise idle GPU.
It retains native E4M3 scale decoding for NVFP4, the four-branch/rank-320 HC up-kernel
specialization, increased INT8 sparse-decode split parallelism, and column-tiled MoE finalization.
Native K-packed NVFP4 FC2 additionally fuses the weighted expert reduction for one or two
rows and up to 16 selected experts under single-EP/single-TP execution. Expert outputs
are rounded to the activation type before reduction, retaining the unfused rounding boundary.
Other shapes retain their generic paths. V5 also fuses 36 target-graph convolution-preparation chains,
removing 324 graph nodes without changing their outputs. It reuses `CausalConvWithState` through
generic channel selection (`input_channels`, `input_channel_offset`), independent carry layout
(`state_channels_last`), extended history (`state_length`), and optional `first_state` output.
These extensions require rank-3 channels-last CUDA input and `state_window=0`; CPU/WebGPU reject
them explicitly. Existing default execution is unchanged. The original and v1-v4 packages are unchanged.

| Prompt tokens | Pre-port decode tok/s | Uncaptured port tok/s | v2 port tok/s | v4 port tok/s | v5 port tok/s | v5 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 234.32 | 39.73 | 141.30 | 155.43 | 166.31 | 3.868 s |
| 16,384 | 245.40 | 39.99 | 146.79 | 166.70 | 178.34 | 8.167 s |
| 32,512 | 249.01 | 44.37 | 151.57 | 165.54 | 177.00 | 17.582 s |

MTP acceptance was 85.14%, 100%, and 100%; sampled peak VRAM was 86.52 GiB.
All nine measured requests produced exactly 512 tokens with no MTP failures or fallback steps.
The benchmark-only alternate EOS ID was supplied through configuration; the package's normal EOS IDs are unchanged.
The package's `performance_benchmark.json` contains individual runs, generated tokens, flags, and ORT/provider hashes.
All 4,608 measured output token IDs match v4 exactly across the nine requests; the package's
`performance_v4_token_parity.json` records these checks and artifact hashes. This establishes sampled
token parity with v4, not numerical parity or parity with the pre-port model. Packed-QKV reuse was
not retained after a small throughput regression. A gated-SiLU fusion was also rejected after it
changed MTP acceptance; the existing standard-op arithmetic is retained.
`performance_default_smoke.json` records chat generation using the persisted configuration without overlays.
`performance_quality_smoke.json` compares three short greedy prompts against the original Engine:
both produce the same arithmetic, capital-city, and sorting answers. Two decoded outputs match
exactly; the capital-city prompt has different reasoning text. Generator also exposes the terminal
EOS token that the original Engine's token events omit. These checks do not establish long-context
quality or full-model numerical parity.

This is **partial recovery**, not restoration of pre-port throughput. The FP8 fused operator can
round differently from dequantize-to-BF16 followed by matmul; direct INT8 attention also reads the
quantized current rows. Operator tests and successful generation do not establish full-model numerical parity.
Some Qwen-domain MTP/head nodes remain; the manifest records their current counts.
Warmed decode profiling still identifies NVFP4 MoE, small-row FP8 GEMV, HC up/down projections,
and sparse attention as remaining decode costs. A wider NVFP4 gate-up CTA did not improve throughput
and was removed. The v5/v6 packages retain their qualified provider binary; subsequent source rebuilds
are not interchangeable with that binary when reproducing these measurements.

Validation: 140 selected CPU/CUDA operator tests passed, including 53 causal-convolution tests,
the complete 47-test DSA suite, fused tensor operations,
HC low-rank mixing (including specialized FP16/BF16 paths), direct NVFP4 gate-up arithmetic
and positive finite E4M3 scale coverage, MoE top-ten finalization with partial column tiles,
and fused NVFP4 FC2 routing with FP16/BF16 outputs, bias, invalid routes, and rounding checks.
Extended convolution tests cover float/FP16/BF16, one/two/nine positions, two batches, packed offsets,
both carry layouts, dilation, missing past/bias, longer histories, optional first-state output,
invalid dimensions/attributes, and explicit CPU rejection.
The plugin test wiring now checks the plugin FP4 target; the earlier 66-test selection
inadvertently excluded the guarded direct NVFP4 tests. All eight package-transform tests passed,
including conservative convolution-fusion guards, idempotence, and hard-linked source preservation:

```bash
python -m unittest discover -s tools/qwen38_graph -p test_optimize_ported_package.py -v
```

The five previously reported DSA failures are fixed: symbolic packed-QKV widths and unshaped cache
capacities remain unknown, head counts respect kernel integer bounds, finite lowest logits remain valid,
and zero-probability valid entries preserve non-finite value arithmetic. Full-model quality and numerical
parity are still not established. The WebGPU-specific cache guards have not been built or tested.

### Standard-op fusion recovery (v6/v7)

V6 reuses `com.microsoft::RotaryEmbedding` for 24 eligible target attention rotary chains.
Normalization stays unchanged: its BF16 rounding boundary is retained before FP32 rotary arithmetic,
with BF16 conversion again afterward. Token-aligned caches are flattened into a position-indexed
cache, including explicit expansion for batch/sequence broadcasting. Conservative shape and external-user
guards leave unsupported patterns unchanged. A broader RMS-normalization fusion was rejected because
it changed sampled generation.

V7 additionally reuses `SigmoidGateMul` for 48 shared-expert `Sigmoid -> Mul -> Add` chains,
removing two kernels per chain. The generic CUDA extension accepts a scalar gate per feature row
(`broadcast_gate=1`) and optional same-shaped residual. It rounds the sigmoid and gated product
to BF16 before residual addition. `sigmoid_in_fp32=0` preserves CUDA `Sigmoid`'s input-type
intermediates, including negative-input subtraction; using FP32 sigmoid instead changed sampled tokens.
The default packed attention layout and FP32 sigmoid behavior remain unchanged.

Same H200/Generator setup as above: one warmup and three measured 512-token requests per context.
Original packages and the qualified v5/v6 packages remain untouched.

| Prompt tokens | v5 decode tok/s | v6 rotary tok/s | v7 rotary + shared expert tok/s | v7 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 166.31 | 169.52 | 171.43 | 3.881 s |
| 16,384 | 178.34 | 181.69 | 183.77 | 8.193 s |
| 32,512 | 177.00 | 180.32 | 182.39 | 17.641 s |

Both versions match all 4,608 corresponding v5 output token IDs across their nine measured requests,
with zero MTP failures/fallbacks. MTP acceptance remains 85.14%, 100%, and 100%.
Each package persists its benchmark, runtime hashes, `performance_v5_token_parity.json`, and default-config
arithmetic/Paris/sorting smoke evidence. This is sampled token parity, not full numerical or long-context
quality parity with the original model. Historical 234-249 tok/s throughput has **not** been recovered.

Validation: 173 selected CPU/CUDA operator tests passed; three WebGPU-only rotary tests were skipped.
New gate tests use zero numerical tolerance for both sigmoid precision modes, residual/no-residual paths,
partial column tiles, and widths 1/257/2560, plus empty inputs, invalid attributes, and shape errors.
All ten package-transform tests pass, including exact reference rotary outputs for multiple batches,
one/nine positions, broadcast caches, fusion guards, idempotence, and hard-linked source preservation.
An exploratory wider indexer selection also reproduced three existing indexer failures on the unchanged
v5 runtime; those unrelated failures are not part of this passing targeted selection.

V8 additionally fuses 48 shared-expert `Sigmoid -> Mul -> Mul` chains using the same generic
`SigmoidGateMul` operator (`silu_gate=1`, `sigmoid_in_fp32=0`). It rounds sigmoid, SiLU, and the
final gated-up product separately to BF16, matching the standard CUDA chain rather than substituting
an FP32 SiLU. Fusion requires provable BF16 types and equal-width outputs from a shared Split;
external users of intermediate sigmoid/SiLU outputs block the rewrite. Existing packed and
broadcast gate modes remain unchanged, and conflicting layout modes fail explicitly.

| Prompt tokens | v7 decode tok/s | v8 decode tok/s | v8 TTFT |
| ---: | ---: | ---: | ---: |
| 8,192 | 171.43 | 174.01 | 3.880 s |
| 16,384 | 183.77 | 186.50 | 8.192 s |
| 32,512 | 182.39 | 185.07 | 17.640 s |

The persisted v8 package again uses one warmup and three measured 512-token requests per context.
All 4,608 output tokens match v5 exactly, with unchanged MTP acceptance and no failures/fallbacks.
Default-config arithmetic, Paris, and sorting token sequences also match v5. V8 retains its
qualified runtime binaries and hash-bound benchmark/quality evidence. All eleven optimizer tests
pass; the expanded operator selection passes 175 CPU/CUDA tests with three WebGPU-only skips.

A one-CTA routing/maps fusion was rejected after slower end-to-end decode. A direct packed-QKV
recurrent experiment preserved sampled tokens but barely improved decode and regressed prefill,
so it was also removed. Its wider GatedDeltaNet test selection reproduced a pre-existing
`RecurrentOutputsIncludeFirstRowState` numerical failure on the untouched v8 provider. The current
source GenAI Engine rejects this model type as unsupported, so v8 measurements continue to use
Generator; the original Engine baseline is not a same-API comparison. Full throughput recovery
and full-model numerical parity remain outstanding.

V9 removes another 50 target projection Split nodes by extending the existing generic
`MatMulBlockQuantizedFp8Weight` operator with optional `split_sizes` and variadic outputs. It
supports one to eight positive last-dimension partitions whose widths sum to N. Eligible
tensor-core GEMV launches store directly into the partition buffers; other paths retain their
original arithmetic and partition the completed result with strided copies. Single-output
graphs retain a separately specialized unsplit kernel. FP16/BF16, activation quantization,
optional bias, rank-one/batched inputs, empty rows, zero K, ragged partitions, and fallback
paths are covered by the operator tests.

| Prompt tokens | v8 decode tok/s | v9 decode tok/s | v9 TTFT |
| ---: | ---: | ---: | ---: |
| 8,192 | 174.01 | 175.24 | 3.881 s |
| 16,384 | 186.50 | 187.85 | 8.192 s |
| 32,512 | 185.07 | 186.39 | 17.634 s |

The v9 table uses the same one-warmup/three-request protocol and 512 generated tokens. A
separate repeated experiment compared against a fresh v8 control found approximately 0.5%
decode improvement at all three context lengths; the final table is not that paired control.
All 4,608 measured output IDs and the three default-config prompt token sequences match v5
exactly, with zero MTP failures/fallbacks. The isolated v9 package persists hash-bound benchmark,
parity, and quality evidence. The rebuilt provider/core hashes match the qualified v9 artifacts.
The combined operator selection passes 193 tests with three WebGPU-only skips, and all twelve
optimizer tests pass. This remains partial recovery, not a restoration of the historical
234--249 tok/s Engine results.

### Profile-guided Slice and gate Cast removal (v10/v11)

Matched v9/original decode profiles identified standard elementwise/layout/reduction
operations as a major regression. These changes target measured launch overhead rather
than changing normalization arithmetic:

* V10 reuses FP8 matmul's partitioned outputs for 48 target projections: 36 convolution
  projections and 12 attention projections. This removes 156 final-axis `Slice` nodes.
  The rewrite proves complete, contiguous, non-overlapping partitions and rebases the
  convolution channel offset or packed `SigmoidGateMul` offset/stride. Unknown layouts,
  incompatible consumers, gaps, overlaps, and full-projection external users block fusion.
  One attention projection remains conservatively unfused.
* V11 additionally extends the existing generic `GatedDeltaNet` CUDA operator to load
  FP16/BF16 gates directly, removing 72 lossless gate-to-FP32 `Cast` nodes. Decay and beta
  share a type: FP32 or the query activation type. Gate loads widen to FP32 before the
  unchanged arithmetic; recurrent state, `a_log`, `dt_bias`, and state updates remain FP32.
  Float-gate and activation-type-gate kernel specializations avoid a prefill regression
  observed with dynamic gate-type loading. Existing FP32-gate models remain supported.
  WebGPU registration remains FP32-gate-only; this CUDA build does not validate WebGPU.

Reprofiling uses the same H200 GPU, Generator, prompt, 128 generated tokens, one warmup
request, and capture after 16 tokens. Each capture contains 55 MTP replays, verified by
two LM-head calls per replay. The table reports **summed GPU kernel durations**, not
critical-path latency; overlap and Nsight node-tracing overhead prevent interpreting
these numbers as a speedup prediction.

| Profile metric, per MTP replay | v9 | v10 | v11 |
| --- | ---: | ---: | ---: |
| Total kernel launches, 8K | 2,028 | 1,872 | 1,800 |
| Standard-op kernel launches, 8K | 760 | 604 | 532 |
| Standard-op GPU time, 8K | 1.037 ms | 0.823 ms | 0.748 ms |
| Standard-op GPU time, 32K | 1.038 ms | not profiled | 0.748 ms |

V11 removes 228 launches per replay, reduces standard-op launches by 30%, and reduces
their summed GPU time by approximately 28% at both lengths. GDN kernel time itself
increases slightly, from 0.344 ms to 0.353 ms per replay; the Cast removal still produces
a net improvement, confirmed by uninstrumented benchmarks.

The following comparison uses a fresh unchanged v9 control and the final qualified v11
benchmark from this optimization campaign. Both use GPU3, batch one, greedy MTP(1), fast
prefill, warmed CUDA graphs, 512 generated tokens, one warmup, and three measured requests
per context. These are Generator results, not the historical original-model Engine baseline.

| Prompt tokens | Fresh v9 decode tok/s | v11 decode tok/s | Decode gain | v11 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 175.62 | 180.73 | 2.91% | 3.871 s |
| 16,384 | 188.21 | 193.70 | 2.91% | 8.168 s |
| 32,512 | 186.73 | 192.13 | 2.89% | 17.589 s |

Every v11 measured decode rate exceeds the corresponding fresh v9 control's measured
range. TTFT does not regress. The independently benchmarked Slice-only v10 reaches
179.08/191.90/190.37 tok/s at these lengths.

Both persistent packages (`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v10`
and `pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v11`) retain their own benchmark,
runtime/graph hashes, parity evidence, and default-config smoke evidence. Each matches all
4,608 measured v5 output token IDs and the arithmetic/Paris/sorting prompt token sequences,
with zero MTP failures/fallbacks. Fresh v9 control tokens also match v11. V11 core/provider/
GenAI hashes match the current build; the v9 qualified artifacts remain unchanged.

Validation: the combined operator selection passes 233 tests with three WebGPU-only skips;
all 16 optimizer tests pass. New tests cover bit-exact FP16/BF16 gates versus already-rounded
FP32 gates across recurrence rules and sequence lengths, invalid gate/query types, exact
partition reference values, rank/layout/type guards, rebased offsets/strides, external
users, idempotence, and hard-linked source preservation. The previously reproduced
`GatedDeltaNetTest.RecurrentOutputsIncludeFirstRowState` baseline failure remains excluded.

This is a measured reduction, not elimination, of the standard-op bottleneck: normalization
chains, remaining casts, and layout operations still account for approximately 0.748 ms
of summed GPU time per replay. Full numerical/model-quality parity and historical
234--249 tok/s throughput recovery remain outstanding.

### Exact-expression sharing (performance v12)

The remaining v11 profile contains repeated BF16-to-FP32 conversions of the same rotary
cosine/sine caches. A conservative standard-expression reuse pass shares exact deterministic
cast/layout results with identical inputs and attributes. It does not fuse arithmetic or
change conversion boundaries. Graph outputs and values captured by nested subgraphs are
protected; custom operators and random/stateful operations are not commoned.

Applied to qualified v11, this removes 48 target `Cast` nodes and two `Unsqueeze` nodes.
The two layout nodes do not launch GPU kernels. No new operator or runtime change is needed;
dense prefill and standalone head graphs, initializers, configuration, and runtime binaries
remain byte-for-byte unchanged.

Matched profiles use the preceding 55-replay methodology:

| Profile metric, per MTP replay | v11 | performance v12 |
| --- | ---: | ---: |
| Total kernel launches | 1,800 | 1,752 |
| Standard-op kernel launches | 532 | 484 |
| Standard-op GPU time, 8K | 0.748 ms | 0.693 ms |
| Standard-op GPU time, 32K | 0.748 ms | 0.693 ms |

Standard-op summed GPU time falls another 7.4%. This is not a critical-path latency
measurement. Uninstrumented decode improves approximately 0.56--0.66% across two repeated
v12 benchmark campaigns against a fresh unchanged v11 control. The final repeated campaign,
persisted in the package, uses the same batch-one Generator/MTP(1), H200 GPU3, fast-prefill,
warmed-CUDA-graph, one-warmup/three-measurement, 512-output-token protocol:

| Prompt tokens | Fresh v11 decode tok/s | performance v12 decode tok/s | Decode gain | v12 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 180.71 | 181.90 | 0.66% | 3.868 s |
| 16,384 | 193.74 | 194.95 | 0.63% | 8.164 s |
| 32,512 | 192.13 | 193.31 | 0.61% | 17.582 s |

The first v12 campaign measured TTFT 0.17--0.34% above the control; this did not reproduce
in the repeated campaign, where TTFT was 0.03--0.08% lower. Both campaigns' measured decode
ranges exceed the control's ranges at every context. Each campaign matches all 4,608 v5
benchmark output IDs, with zero MTP failures/fallbacks; three default-config prompt token
sequences also match exactly.

The persistent `pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v12` package retains
the final benchmark, parity/quality evidence, artifact hashes, and
`performance_standard_expression_comparison.json` with both benchmark campaigns and
hash-bound 8K/32K profile comparisons. V11 and earlier qualified packages remain unchanged.
All 19 optimizer tests pass, including exact rounding/value checks, atomic hard-link-safe
writes, idempotence, attribute/input/domain guards, graph outputs, nested-subgraph captures,
and metadata persistence. No C++ runtime code changed in this step.

Normalization chains and GDN splits remain significant standard-op costs. This step leaves
their arithmetic and layout unchanged; it does not restore historical 234--249 tok/s
throughput or establish full-model numerical/quality parity.

### Normalization chains, GDN splits, and follow-up casts (performance v13--v15)

The next profile-guided stages preserve the qualified v12 package and use existing generic
contrib operators:

* **V13: standard-arithmetic RMS chains.** `BranchwiseRMSNorm` gains
  `use_onnx_arithmetic=1`. It combines input widening/squaring, reuses ORT's existing
  fast FP32 reduction implementation, and combines mean/epsilon/sqrt/divide/scale/output
  conversion in an epilogue. The scalar-result case uses ORT's row reduction; other
  rows use its column reduction. Explicit rounding boundaries and division are retained,
  rather than changing the reduction tree or substituting reciprocal square root.
  Twenty-four target attention normalization chains become three-launch contrib calls.
  CPU supports the arithmetic mode; WebGPU explicitly rejects it and retains its default
  implementation. The CUDA-only build does not validate WebGPU.
* **V14: packed GDN decode.** Thirty-six convolution Q/K/V Split nodes and their 108
  private reshape views are replaced by the existing packed `GatedDeltaNet` input contract.
  Short warp-specialized recurrence (at most two tokens per batch on average) reads the
  packed rows directly, with a separately specialized packed-input kernel. Output/state
  layout and arithmetic remain unchanged. Longer inputs and other engines retain the
  existing unpack path; dense fast-prefill graphs remain byte-for-byte unchanged.
  An initial larger variant-pack layout regressed TTFT by approximately 0.6--0.9%;
  keeping the packed flag in existing padding removed that regression.
* **V15: follow-up standard-op reduction.** Reprofiling identified normalization outputs
  immediately widened again for FP32 rotary. Optional `BranchwiseRMSNorm.Y_fp32` stores
  the **already FP16/BF16-rounded output** widened to float, removing another 24 Cast
  launches. It does not expose the unrounded normalization result or bypass the rounding
  boundary. This optional output requires `use_onnx_arithmetic=1`.

Rewrites prove normalization axes, scale/input types, epsilon, reshape widths, and private
intermediates. Packed GDN requires complete Q/K/V partitions, compatible head geometry,
known recurrent-state dimensions, and channels-last one-dimensional convolution input.
Graph outputs, external users, and nested-subgraph captures block removal when necessary.
Unsupported patterns remain untouched.

All stages below use the final rebuilt core/provider, H200 GPU3, batch-one Generator,
greedy MTP(1), fast prefill, warmed CUDA graphs, one warmup and three measured 512-output-token
requests per context. The v12 comparison is a fresh unchanged control from this campaign.

| Prompt tokens | Fresh v12 tok/s | V13 RMS tok/s | V14 + packed GDN tok/s | V15 + cast removal tok/s | V15 gain over v12 | V15 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 181.89 | 185.58 | 187.77 | 188.39 | 3.57% | 3.869 s |
| 16,384 | 194.91 | 198.90 | 201.22 | 201.86 | 3.57% | 8.173 s |
| 32,512 | 193.35 | 197.22 | 199.57 | 200.17 | 3.53% | 17.594 s |

V15 TTFT is slightly lower than the fresh v12 control at all three lengths; measured
decode ranges do not overlap the v12 control ranges. Each of v13/v14/v15 matches all
4,608 corresponding v5 benchmark output IDs and the three default-config prompt sequences,
with zero MTP failures/fallbacks. These are sampled token checks, not full numerical or
model-quality parity.

Final matched 8K/32K profiles retain the preceding 55-replay protocol:

| Profile metric, per MTP replay | V12 | V15 |
| --- | ---: | ---: |
| Total GPU kernel launches | 1,752 | 1,548 |
| Standard-op GPU kernel launches | 484 | 232 |
| Standard-op summed GPU time, 8K | 0.693 ms | 0.318 ms |
| Standard-op summed GPU time, 32K | 0.693 ms | 0.318 ms |
| New fused normalization square/epilogue time, 8K | none | 0.067 ms |
| New fused normalization square/epilogue time, 32K | none | 0.067 ms |

The standard-op category alone drops approximately 54%, but part of its work moves into
48 new contrib square/epilogue launches. **Including those new kernels**, summed time for
the affected standard/normalization work drops approximately **44.4--44.5%**, to
0.384--0.385 ms per replay. The unchanged reduction launches remain in the standard
category. Summed GPU kernel durations include overlap and Nsight overhead; uninstrumented
benchmarks, not these sums, establish the throughput gain.

All three persistent packages retain separate benchmarks, runtime/graph hashes, token-parity
and default-quality evidence. The final
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v15` also contains
`performance_normalization_gdn_comparison.json` with staged benchmarks, profile database
hashes, and explicit accounting for new contrib normalization kernels. Its binaries match
the current build. Dense/head graphs, weights, configuration, and the GenAI library remain
unchanged from v12; the final optimizer reproduces the qualified target graph byte-for-byte.

Validation: 240 selected operator tests pass with three WebGPU-only skips; all 22 optimizer
tests pass. New CUDA tests compare both normalized and widened output bytes to the actual
unfused standard CUDA chain for FP32/FP16/BF16, widths 1/31/128/256/513/1024/4096, and
1/2/9 rows. Packed/separate GDN outputs and recurrent state match byte-for-byte for FP16/BF16,
64/128/256 key widths, short decode and longer fallback sequences, and both gate types.
CPU arithmetic/invalid-mode tests and graph-layout/type/external-user/capture guards also
pass. The documented pre-existing GDN first-row-state test remains excluded.

Remaining standard costs now include 49 FP32 Expand launches (about 48 us/replay), 46
FP32/BF16 conversion launches (about 70 us), and 28 FP32 reductions (about 44 us, including
the retained RMS sums), plus smaller slices/gathers/elementwise chains. Two head attention
normalization chains remain conservatively unfused because their activation type is not
proven through the current head graph. Further optimization should follow these measured
costs and preserve reduction and rounding behavior. Historical 234--249 tok/s recovery
remains unfinished.

### Remaining Expand, conversion, and reduction work (performance v16--v18)

The next three measured stages start from qualified v15 and keep its package unchanged:

* **V16: share rotary cache expansions.** The target's 48 rotary cache expansions
  repeatedly materialize the same two caches. A conservative batch/sequence shape proof
  follows standard layouts and the contracts of generic hyper-connection, FP8 projection,
  normalization, and channels-last convolution operators. Equal leading dimensions let
  23 redundant shape expressions share a result; existing exact-expression deduplication
  then removes 46 Expand nodes. Different/unknown dimensions, graph outputs, and
  nested-subgraph captures block unsafe sharing. Arithmetic and runtime binaries remain
  unchanged from v15.
* **V17: fuse squares into the existing reduction.** The CUDA reduction helper gains an
  FP32-rounded square transform. `BranchwiseRMSNorm`'s explicit-arithmetic mode uses it
  for multiple rows, retaining exactly the previous thread/warp/block summation order.
  `__fmul_rn` prevents contraction of square-plus-sum into FMA. This eliminates 24
  separate square launches per target replay and their full-sized FP32 scratch buffers.
  The scalar-result case retains the previous square-plus-row-reduction path. Required
  sums are not removed or replaced by a different reduction tree.
* **V18: keep rotary storage in the activation type.** Version 2 of generic
  `com.microsoft::RotaryEmbedding` adds `use_fp32_arithmetic=1`: load FP16/BF16 input
  and cache values, widen inside the kernel, perform separately rounded FP32 products
  and addition, then round once to the input type. This is byte-equivalent to the
  previous widen -> FP32 rotary -> output Cast chain. Twenty-four rotary chains
  remove 26 standalone Cast nodes and no longer need the 24 optional widened RMS
  outputs. The shared caches are expanded in BF16 rather than FP32. Version 1 and
  legacy arithmetic stay unchanged. Version 2 is CUDA-only; versioning prevents other
  providers from silently applying their legacy low-precision arithmetic to this mode.
  Dense/head graph files are unchanged.

Same-campaign H200 GPU3 results: batch-one Generator, greedy MTP(1), fast prefill, warmed
CUDA graphs, one warmup and three measured 512-output-token requests per context.

| Prompt tokens | Fresh v15 tok/s | V16 shared Expand tok/s | V17 + square reduction tok/s | V18 + rotary casts tok/s | V18 gain over v15 | V18 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 188.60 | 191.29 | 191.83 | 192.62 | 2.13% | 3.870 s |
| 16,384 | 201.99 | 204.88 | 205.39 | 206.25 | 2.11% | 8.168 s |
| 32,512 | 200.33 | 203.19 | 203.66 | 204.50 | 2.08% | 17.587 s |

Cache sharing contributes approximately 1.43%; the square/reduction stage adds another
0.23--0.28%, and rotary cast fusion adds approximately 0.41--0.42%. V17 has a small
0.11--0.22% TTFT increase in this campaign; final v18 TTFT is slightly lower than the fresh
v15 control at every length. All staged decode ranges are above the v15 control range.

Matched 8K and 32K profiles use the preceding 55-replay protocol:

| Profile metric, per MTP replay | V15 | V18 |
| --- | ---: | ---: |
| Total GPU launches | 1,548 | 1,452 |
| Standard-op GPU launches | 232 | 160 |
| All Expand launches | 51 | 5 |
| Float/BF16 Cast launches | 46 | 20 |
| Separate RMS square/epilogue launches | 48 | 24 |
| Standard-op summed GPU time, 8K / 32K | 0.318 / 0.318 ms | 0.248 / 0.248 ms |
| RMS square/epilogue summed time, 8K / 32K | 0.067 / 0.067 ms | 0.042 / 0.042 ms |
| Rotary summed time, 8K / 32K | 0.039 / 0.039 ms | 0.045 / 0.045 ms |

That is 96 fewer launches per replay. The standard-op category drops approximately 22%;
including RMS and rotary kernels, summed time for the affected work drops **21.0--21.1%**
(approximately 0.423--0.424 ms to 0.334 ms). Rotary kernels and transformed reductions
individually take slightly longer because they now include work previously done by Cast
and square kernels; the inclusive totals account for this. These are summed GPU durations,
not critical-path latency or a substitute for uninstrumented throughput measurements.

Validation: **244 selected operator tests pass**, with three WebGPU-only skips and the
same documented pre-existing GDN test excluded. RMS tests remain byte-exact against the
actual standard CUDA chain for float/FP16/BF16, including scalar and multi-block cases.
New rotary tests compare FP16/BF16 bytes against the actual widen -> float rotary -> round
chain for full/partial rotary widths, 1/2/9 sequence lengths, batch two, both layouts,
both rotation formats, and scalar/per-token positions. Legacy-default, invalid-mode,
and unsupported-CPU-fallback tests pass. All **24 optimizer tests pass**, including
numerical cache/cast checks, external/captured values, explicit legacy attributes,
generated-node name collisions, idempotency, and atomic hard-link safety.

V16/v17/v18 each match all 4,608 v5 benchmark token IDs and three default-config prompt
sequences, with zero MTP failures/fallbacks. All artifact hashes are verified; the final
optimizer reproduces both qualified v16 and v18 target graph bytes. V16 retains v15's
runtime; v17/v18 match the final runtime build. Configurations, weights, dense/head graphs,
and the GenAI library remain unchanged. The final
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v18` contains
`performance_remaining_standard_comparison.json` with staged benchmarks, hashes, and
inclusive profile accounting.

Remaining standard work is smaller and no longer Expand-dominated: all five Expand
launches total about 5 us/replay; 20 conversions total about 35 us; 24 exact FP32
square-sums take about 44 us, and five other reductions take about 12 us. Slices and
head/PLE elementwise work remain. The required normalization sums intentionally remain
to preserve their numerical contract. Sampled exact tokens do not establish full
numerical/model-quality parity, and historical 234--249 tok/s recovery remains unfinished.

### Head normalization and rotary chains (performance v19)

V19 starts from qualified v18 and reuses its runtime binaries unchanged. Two remaining
head normalization/rotary chains can use the existing exact-arithmetic contrib operators:

* `SigmoidGateMul` constrains every floating input/output to BF16. This proves the
  shared head projection's type even when Python ONNX inference cannot infer through
  its opaque hyper-connection producer; no model-specific producer assumption is used.
* `FoldMid2`'s fixed CUDA contract provides BF16 rotary caches of shape `[1, 2, 32]`.
  Optimizer-local inference hints recover this missing metadata without changing the
  graph's declared shapes. Contradictory declared types/shapes fail explicitly.
* Rotary matching now permits shared cache Unsqueeze views, removing them only when
  unused after all matches. Graph outputs and nested-subgraph captures remain protected.
  The existing FP32 rotary fusion runs before exact RMS fusion and activation-storage
  rotary fusion, preserving every normalization and BF16 rounding boundary.

The target adds two `BranchwiseRMSNorm` chains and two version-2 `RotaryEmbedding`
chains. Ten standalone Cast nodes and six Slice nodes are removed. Other package graphs,
all runtime libraries, configurations, and the 45 hard-linked external weight files stay
unchanged. The separately qualified v18 package is preserved.

Fresh same-campaign control and candidate results use H200 GPU3, batch-one Generator,
greedy MTP(1), fast prefill, warmed CUDA graphs, one warmup and three measured
512-output-token requests per context:

| Prompt tokens | Fresh v18 tok/s | V19 tok/s | Decode gain | Fresh v18 TTFT | V19 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 192.55 | 193.35 | 0.42% | 3.887 s | 3.879 s |
| 16,384 | 206.39 | 207.11 | 0.35% | 8.190 s | 8.185 s |
| 32,512 | 204.52 | 205.30 | 0.38% | 17.633 s | 17.625 s |

All three candidate decode ranges exceed their fresh control ranges. The gain is
incremental, not recovery of the historical pre-port throughput.

Matched 8K/32K profiles retain the 55-replay, two-LM-head-calls-per-replay protocol:

| Profile metric, per MTP replay | V18 | V19 |
| --- | ---: | ---: |
| Total GPU launches | 1,452 | 1,418 |
| Standard-op GPU launches | 160 | 122 |
| Float/BF16 Cast launches | 20 | 10 |
| Slice launches | 24 | 18 |
| Expand launches | 5 | 9 |
| Standard-op summed GPU time, 8K / 32K | 0.248 / 0.248 ms | 0.196 / 0.196 ms |
| RMS epilogue summed GPU time, 8K / 32K | 0.042 / 0.042 ms | 0.045 / 0.045 ms |
| Rotary summed GPU time, 8K / 32K | 0.045 / 0.045 ms | 0.048 / 0.048 ms |

The affected work, including added RMS/rotary work rather than just the shrinking
standard-op category, drops **13.5--13.6%** in summed GPU duration. Cache broadcasting
adds four Expand launches; they are included rather than hidden. These are summed
kernel durations, not critical-path latency.

Validation: **25 optimizer tests pass**, including opaque-producer type proofs,
contradictory metadata, shared-cache numerical parity, nested captures, idempotency,
and atomic hard-link safety. **51 selected operator tests pass**, with three WebGPU-only
skips; these include exact float/FP16/BF16 RMS, exact rotary rounding, gate contracts,
and folded-state behavior. V19 matches all **4,608 v5 benchmark token IDs** and all three
default-configuration prompt sequences, with zero MTP failures/fallbacks. Artifact
hashes, unchanged v18 files, topology, external-weight inodes, and byte-exact graph
reproduction are verified. Evidence is persisted in
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v19/performance_head_standard_comparison.json`.

The remaining standard costs include 26 required exact square-sum reductions
(about 47 us/replay), 13 four-dimensional BF16 Slice launches (about 20 us), and eight
BF16-to-FP32 conversions (about 20 us). Those are candidates for further profile-guided
work, not permission to change the reduction tree or discard rounding boundaries.
Sampled-token agreement still does not establish full numerical/model-quality parity.

### Packed query Slice -> RMS reads (performance v20)

V20 removes the next repeated layout cost: thirteen private four-dimensional BF16
query Slice nodes feeding exact RMS normalization. Version 2 of generic
`com.microsoft::BranchwiseRMSNorm` accepts `input_start` and `input_width` to read a
last-axis interval directly from packed rows. The output retains the leading dimensions
with the selected last-axis width. This mode requires `use_onnx_arithmetic=1` and grouped
input (`num_branches=0`). Version 1 and default arithmetic are retained; version 2 is
CUDA-only so other providers cannot silently ignore the packed layout.

The CUDA implementation passes the original row stride to the existing reduction
kernel. The selected width still controls the block/grid configuration and every
accumulation step. The scalar-result path still materializes FP32-rounded squares and
uses the original row reduction. Separate compile-time packed/contiguous epilogues
retain the existing contiguous path while reading packed activations directly when
needed. No summation tree or rounding boundary changes.

The optimizer fuses only private, constant, in-bounds, unit-step last-axis slices with
proven source rank/width. Shared outputs, graph outputs, nested-subgraph captures,
unknown widths, other axes, negative offsets, and unsupported RMS modes remain intact.
The rewrite is atomic and idempotent and is wired into normal package optimization.
The separately qualified v19 package remains unchanged.

H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA graphs,
one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v19 control tok/s | V20 tok/s | Decode gain | V19 TTFT | V20 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 193.61 | 193.89 | 0.14% | 3.871 s | 3.876 s |
| 16,384 | 207.36 | 207.66 | 0.15% | 8.164 s | 8.185 s |
| 32,512 | 205.56 | 205.89 | 0.16% | 17.587 s | 17.620 s |

This is a small decode gain, with a 0.12--0.25% TTFT increase in the repeat campaign.
The first campaign was almost flat at 8K/16K and showed an unexpectedly large 3.85%
32K gain. Both unchanged packages were rerun; that large gain did not reproduce and
is not attributed to this optimization. Initial and repeated results are preserved
in the verification evidence. The repeated candidate decode ranges are above the
corresponding repeated control ranges, but the small gain should not be overstated.

Matched 8K/32K profiles use the preceding 55-replay protocol:

| Profile metric, per MTP replay | V19 | V20 |
| --- | ---: | ---: |
| Total GPU launches | 1,418 | 1,405 |
| Standard-op GPU launches | 122 | 109 |
| Slice launches | 18 | 5 |
| Standard-op summed GPU time, 8K / 32K | 0.196 / 0.196 ms | 0.182 / 0.181 ms |
| RMS epilogue summed GPU time, 8K / 32K | 0.045 / 0.045 ms | 0.043 / 0.043 ms |
| Exact square-sum time (included in standard-op row), 8K / 32K | 0.047 / 0.047 ms | 0.053 / 0.053 ms |

The packed reduction reads take slightly longer than contiguous reads, but no longer
need the thirteen Slice launches. Standard work plus normalization epilogues drops
**6.5--7.1%** in summed GPU duration. The table includes moved work rather than
claiming the full removed Slice duration as a net saving. Summed kernel time is not
critical-path latency or an end-to-end throughput prediction.

Validation: **23 selected operator/reduction tests pass** and **26 optimizer tests
pass**. Packed RMS outputs and optional widened outputs match the actual standard
CUDA Slice/cast/square/mean/sqrt/divide/scale/round chain byte-for-byte for
float/FP16/BF16, widths 1/31/128/256/513/1024/4096, rows 1/2/9/48, zero/nonzero
offsets, and rank-three/rank-four inputs. Tests also cover empty inputs, absent scale,
invalid attributes/bounds, CPU fallback rejection, unchanged legacy arithmetic,
shared-reduction buffers, and repeated multi-block reductions.

Both benchmark campaigns match all 4,608 v5 output token IDs; the qualified final
package also matches all three default-configuration prompt sequences, with zero
MTP failures/fallbacks. The final optimizer reproduces the qualified target graph
bytes. Configurations, dense/head graph files, the GenAI library, and all 45
hard-linked external weight files remain unchanged. The runtime/provider are updated
only in the isolated v20 package. Verification evidence is persisted in
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v20/performance_slice_rms_comparison.json`.

Eight head BF16-to-FP32 conversions and remaining gate/layout chains are still
candidates for later work. Required exact reductions remain; historical pre-port
throughput and full numerical/model-quality parity are still not established.

### Profile-guided MoE and hyper-connection tuning (performance v21 HC L1)

The next pass targets the dominant decode projections rather than further standard-op
removal. The v20 8K profile attributes 2.129 ms per MTP replay to MoE expert compute
and finalization and 2.006 ms to hyper-connections, versus 0.182 ms to standard ops.
These are summed kernel durations, including overlap and dependent-launch waits,
not critical-path latency.

Only a small HC cache-policy change is retained in the isolated
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v21_hc_l1` package.
The existing up kernel now loads its repeatedly reused down-projection partial sums
with coherent `__ldca` caching instead of bypassing L1 with `__ldcg`. Loads remain
after `GridWait()`. Kernel launches, split-sum order, ScaledSiLU rounding, up-projection
arithmetic, graph files, and weights are unchanged. An exploratory `__ldg` variant
was superseded by the coherent load rather than relying on non-coherent reads with
overlapping dependent grids.

Two larger changes were tested and **removed from the final source/runtime**:

| Rejected experiment | 8K decode change | 32K decode change |
| --- | ---: | ---: |
| Separate HC hidden-preparation kernel | -2.63% | -2.57% |
| MoE gate-up: 256 threads / 16 columns instead of 128 / 8 | -0.20% | -0.13% |

These screening runs use one warmup and one measured 512-token request per context,
against a fresh three-request v20 control. The extra HC kernel adds 96 launches per
replay; its inclusive HC summed time increases from 2.006 to 3.286 ms. Extracting
repeated work is therefore not an end-to-end win once launch/dependency costs are
included. MoE retains the original tile and fixed K-lane reduction order.

The retained coherent-cache variant was measured twice. First-campaign decode changes
were -0.01% / +0.06% / +0.11% at 8K / 16K / 32K. Both unchanged packages were then
rerun using H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA
graphs, one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v20 tok/s | V21 HC L1 tok/s | Decode change | V20 TTFT | V21 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 193.87 | 194.12 | +0.13% | 3.871 s | 3.870 s |
| 16,384 | 207.55 | 207.84 | +0.14% | 8.173 s | 8.173 s |
| 32,512 | 205.80 | 206.09 | +0.14% | 17.609 s | 17.602 s |

The repeated decode ranges are separated, but this is a **very small tuning result**,
not substantial throughput recovery. The first 8K campaign was flat. Matched 55-replay
profiles retain all 1,405 launches and 109 standard-op launches; inclusive HC summed
time is nearly unchanged: 2.006 -> 2.005 ms at 8K and 2.010 -> 2.000 ms at 32K.
MoE remains approximately 2.13 ms. Do not attribute a larger speedup to removed or
moved work, or infer critical-path savings from these summed durations.

Validation: all **16 selected HC/NVFP4 operator tests pass**, including added coverage
for the deployed four-branch, hidden-2560, rank-320 HC shape with one/two rows,
FP16/BF16, and both prologue modes. Sixteen deterministic standalone HC/QMoE cases
also match preserved v20 output bytes exactly for every experimental runtime:
FP16/BF16, one/two rows, HC prologue on/off, and MoE bias on/off with top-ten routing.
Both full coherent-cache benchmark campaigns match all 4,608 control token IDs.
The final package additionally matches all 4,608 v5 benchmark token IDs and all three
default-configuration prompt sequences, with zero MTP failures or standard fallbacks.

The qualified v20 package is unchanged. All graph/configuration files, the GenAI
library, and 45 hard-linked external weight files are preserved in the candidate.
Final runtime/provider hashes match the current build. Benchmark campaigns, rejected
experiments, operator byte-parity results, package-preservation checks, and both
context profiles are persisted in the candidate's `performance_moe_hc_comparison.json`.
Further meaningful recovery still requires improving MoE/HC projection efficiency;
neither the rejected tile nor the extra preparation launch should be reintroduced
without new evidence.

### Exact-order NVFP4 gate-up scheduling (performance v22)

A fresh v21 decode profile still identifies MoE gate-up as the largest individual
kernel family: approximately 1.36 ms across 48 calls per MTP replay. Nsight Compute
is installed, but collecting hardware counters fails with `ERR_NVGPUCTRPERM`.
No shared-system permissions were changed. Memory bandwidth, stall causes, and
achieved occupancy therefore remain unmeasured; static disassembly is not a
hardware-counter diagnosis.

The retained v22 experiment unrolls two successive K-loop iterations in the raw
K-packed NVFP4 fused gate-up kernel, exposing more work to compiler scheduling.
It does not change the 128-thread/eight-column tile, K-lane count, per-lane
accumulation sequence, shuffle tree, dequantization rounding, or activation
epilogue. Non-gate-up paths use unroll factor one. BF16 gate-up register usage
increases from 34 to 48 registers per thread, with no local-memory spill reported
by the resource dump. This tradeoff is qualified by end-to-end measurements,
not an assumption that more unrolling or higher occupancy must be better.

The isolated package is
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v22_moe_unroll2`.
It retains v21's coherent HC partial-sum loads and all v20 graph rewrites.
H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA graphs,
one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v21 tok/s | V22 tok/s | Decode change | V21 TTFT | V22 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 194.06 | 194.44 | +0.20% | 3.876 s | 3.876 s |
| 16,384 | 207.95 | 208.41 | +0.22% | 8.170 s | 8.162 s |
| 32,512 | 206.18 | 206.63 | +0.22% | 17.596 s | 17.580 s |

The first paired campaign showed +0.25% / +0.16% / +0.15% at 8K / 16K / 32K.
The table is the repeat campaign, with candidate measured before control to
reverse package order. Candidate/control decode ranges are separated in both
campaigns, but the improvement remains **small**, not substantial recovery of
historical pre-port throughput.

Matched 55-replay profiles retain 1,405 GPU launches and 109 standard-op launches:

| Summed GPU duration per MTP replay | V21 8K / 32K | V22 8K / 32K |
| --- | ---: | ---: |
| NVFP4 gate-up | 1.360 / 1.364 ms | 1.349 / 1.347 ms |
| FC2 and finalization | 0.770 / 0.771 ms | 0.771 / 0.769 ms |
| Inclusive MoE expert compute and finalization | 2.129 / 2.135 ms | 2.120 / 2.116 ms |

The inclusive MoE category improves approximately 0.45--0.90%; the full decode
gain is smaller. Summed durations include overlap, dependent-launch waits, and
profiling overhead and must not be treated as critical-path savings.

Validation: **17 selected HC/NVFP4 tests pass**. New gate-up tests cover partial
two/ten-column tiles, K=16/272 short and tail loops, FP16/BF16, and bias on/off.
Sixteen deterministic deployed-shape HC/QMoE cases match preserved v20 and v21
outputs byte-for-byte. Both full benchmark campaigns match all 4,608 v5 output
token IDs per package. The final candidate also matches all three v5
default-configuration prompt sequences, with zero MTP failures/fallbacks.

Qualified v20/v21 artifacts remain unchanged. Graphs, configuration, the GenAI
library, and all 45 hard-linked external weight files are preserved. Final
runtime/provider hashes match the current build. Comparison evidence, including
both campaign orders, hardware-counter denial, register tradeoff, operator parity,
and profiles, is persisted in the candidate's `performance_moe_unroll_comparison.json`.
FP8 GEMV remains the next major projection target; its hardware limiting factor
has not been established by this experiment.

### Exact-order FP8 projection scheduling (performance v23)

V23 applies a bounded scheduling experiment to the next major decode projection
cost, tensor-core FP8 GEMV. The existing K-window loop is unrolled twice without
changing KSplit selection, window visitation order within each warp, per-block
weight-scale boundaries, accumulator reduction, activation Q/DQ, or split-output
layout. Earlier KSplit changes are not reintroduced. This retains v22's MoE
scheduling and v21's coherent HC cache loads.

Hardware performance counters remain unavailable (`ERR_NVGPUCTRPERM`); no system
permissions were changed. The result is therefore a measured scheduling improvement,
not proof of a bandwidth, occupancy, or stall diagnosis. Static resource inspection
of the dominant BF16 KSplit8/single-row-tile/split-output entry point shows register
usage increasing from 48 to 64 per thread, unchanged 5,120-byte shared allocation,
and no reported local-memory or stack allocation.

The isolated package is
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v23_fp8_unroll2`.
H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA graphs,
one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v22 tok/s | V23 tok/s | Decode change | V22 TTFT | V23 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 194.24 | 194.92 | +0.35% | 3.882 s | 3.892 s |
| 16,384 | 208.31 | 209.09 | +0.38% | 8.168 s | 8.187 s |
| 32,512 | 206.54 | 207.30 | +0.37% | 17.587 s | 17.621 s |

First-campaign decode changes were +0.40% / +0.35% / +0.35%. The table shows
the repeated campaign with candidate measured before control, reversing package
order. Candidate/control decode ranges are separated in both campaigns.
Repeat TTFT increases approximately 0.20--0.24%; the first campaign had slightly
lower candidate TTFT. This is a small decode gain, not substantial historical
throughput recovery or an established TTFT improvement.

Matched 55-replay profiles retain all 1,405 GPU launches and 109 standard launches:

| Summed GPU duration per MTP replay | V22 8K / 32K | V23 8K / 32K |
| --- | ---: | ---: |
| FP8 projection GEMV | 1.771 / 1.768 ms | 1.730 / 1.731 ms |
| MoE expert compute and finalization | 2.120 / 2.116 ms | 2.116 / 2.117 ms |
| Hyper-connections | 2.007 / 2.004 ms | 2.038 / 2.039 ms |

FP8 summed time decreases approximately 2.1--2.3%, but HC summed durations
increase by approximately 31--36 microseconds per replay. Include that offset
rather than attributing the entire FP8 reduction to end-to-end savings. These
durations include overlap, dependent-launch waits, and profiler overhead, not
critical-path latency. End-to-end retention is based on the paired benchmarks.

Validation: **18 FP8 operator tests pass**, including extended exact split-output
coverage for K=320/1088 short and odd-window tails. Twenty-four deterministic
standalone FP8 cases match v22 output bytes exactly: FP16/BF16; one/two/nine
activation rows; split/contiguous outputs; partial output tiles; activation
scale and bias present/absent; K=256/320/1088/2560; and block widths 64/128/192.
Both full benchmark campaigns match all 4,608 v5 output token IDs per package.
The final candidate also matches all three v5 default-configuration prompt
sequences, with zero MTP failures or standard fallbacks.

Qualified v20/v22 artifacts remain unchanged. Graphs, configurations, the GenAI
library, and 45 hard-linked external weight files are preserved; final candidate
runtime/provider hashes match the current build. Both campaign orders, resource
tradeoffs, byte-parity checks, preserved artifacts, default prompts, and inclusive
profiles are persisted in the candidate's `performance_fp8_scheduling_comparison.json`.

### Stable tiny-decode MoE expert maps (performance v24)

V24 specializes only the FP4 GEMV expert-map builder for one/two activation
rows, at most 512 local experts, and at most 16 routes per token. A single
128-thread CTA stable-ranks the selected expert IDs, writes both permutation
maps and sorted expert IDs, and counts all expert offsets directly. At most
32 selected routes fit in shared memory. Equal expert IDs retain token-major
selection order; source-row maps retain the existing route-major convention.
Unsupported shapes keep the previous CUB map builder and three-step fallback.
Softmax/top-k arithmetic, routing snapshots, expert arithmetic, and existing
v21--v23 optimizations are unchanged. No graph or schema changes are required.

The isolated qualified package is
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v24_small_maps_final`.
H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA graphs,
one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v23 tok/s | V24 tok/s | Decode change | V23 TTFT | V24 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 195.01 | 196.16 | +0.59% | 3.884 s | 3.894 s |
| 16,384 | 209.18 | 210.41 | +0.59% | 8.168 s | 8.189 s |
| 32,512 | 207.37 | 208.58 | +0.59% | 17.584 s | 17.627 s |

The first campaign measured control before candidate and improved decode by
+0.62% / +0.69% / +0.63%. The table reverses package order; candidate/control
decode ranges are separated in both campaigns. Repeat TTFT increases
approximately 0.25--0.26%; first-campaign TTFT changes range from -0.10% to
+0.22%. This is a small decode gain with a small measured TTFT cost, not
substantial historical throughput recovery. Source/header rebuilds changed
provider hashes between preliminary and final trials, so both final campaign
orders and the profiles use the same frozen final provider binary. Preliminary
results are recorded separately rather than attributed to that binary.

Matched 55-replay profiles retain all 1,405 GPU launches, including the same
48 routing and 48 map launches per replay. This optimization reduces map work,
not launch count:

| Summed GPU duration per MTP replay | V23 8K / 32K | V24 8K / 32K |
| --- | ---: | ---: |
| Routing softmax/top-k | 0.292 / 0.292 ms | 0.292 / 0.293 ms |
| Expert maps | 0.187 / 0.187 ms | 0.123 / 0.122 ms |
| Routing plus expert maps | 0.479 / 0.479 ms | 0.416 / 0.415 ms |
| MoE expert compute and finalization | 2.116 / 2.117 ms | 2.114 / 2.119 ms |
| Hyper-connections | 2.038 / 2.039 ms | 2.042 / 2.038 ms |

Map summed time decreases approximately 34%; routing plus maps decreases
approximately 13%. Include the surrounding expert/HC costs rather than treating
all removed map time as end-to-end savings. These durations include overlap,
dependent-launch waits, and profiler overhead, not critical-path latency.

A combined routing/map fusion was also tested and **rejected**: despite
removing a map launch, sequential row sorts regressed full-model decode by
approximately 1.60--1.61%. All fusion changes were removed. An initial standalone
probe with hidden/intermediate dimensions 64/32 was outside FP4 GEMV support;
repeating unchanged v23 itself changed low output bits on that fallback path.
Its mismatch therefore does not establish a fusion correctness defect. It was
replaced by GEMV-eligible 512/512 probes with repeated-baseline checks, rather
than relaxing exact-output requirements.

Validation: **45 focused operator tests pass**, including a CPU stable-sort map
reference for FP16/BF16 routing inputs, one/two rows, 16/257/512 experts,
1/10/16 routes, normalized/unnormalized routing, tied/varied logits, and
unsupported-shape guards. Sixteen GEMV-eligible 512-expert standalone cases
match repeated v23 output bytes exactly across FP16/BF16, one/two rows,
normalization on/off, and bias on/off. Sixteen additional deployed-shape
MoE/HC cases match v20 and v23 bytes exactly. Both final benchmark campaigns
match all 4,608 v5 output token IDs per package; the candidate also matches all
three default-configuration v5 prompt sequences. Qualification and both
128-output-token profiles report zero MTP failures or standard fallbacks.

Qualified v20/v23 artifacts remain unchanged. Graphs, configurations, the GenAI
library, and all 45 hard-linked external weight files are preserved. Final
runtime/provider hashes match the current build and package manifest. Both
campaign orders, rejected-fusion evidence, exact-output checks, preservation
checks, default prompts, and inclusive profiles are persisted in the candidate's
`performance_small_maps_comparison.json`.

The final v24 traces were also analyzed for interval overlap. MoE expert compute
has approximately 2.110/2.116 ms of exclusive kernel intervals per 8K/32K replay;
FP8 projections have 1.719/1.719 ms. HC has only 0.803/0.806 ms exclusive despite
approximately 2.04 ms summed duration, so its summed rank alone is misleading.
No graph kernel is active in approximately 3.06%/2.98% of the traced interval;
this does not establish host overhead, since other GPU work may occupy gaps.
Exclusive intervals likewise do not prove critical-path contributions.

Within sparse indexing, tile-top-k merge costs 0.407/0.457 ms across 117 launches
per replay; score-tile top-k costs 0.079/0.183 ms across 13 launches. Scoring
accounts for approximately two-thirds of the indexer's 8K-to-32K duration growth,
while merging dominates its absolute cost. The next focused investigation is
therefore sparse indexer merge/scoring, with exact selection and tie-order
preservation required; MoE gate-up remains the largest individual compute
target at 1.344/1.348 ms. No new kernel optimization is claimed from this
analysis. Interval breakdowns and static resources are persisted in the
candidate's `performance_bottleneck_refresh_analysis.json`.

### Shared two-level sparse indexer merges (performance v25)

V25 targets the packed QSA indexer's hierarchical top-k merge, not its scoring
arithmetic. A CUDA block stages four sorted input lists in shared memory,
merges each pair into shared intermediate lists, then merges those lists into
the output buffer. Both levels retain the existing integer merge-rank search,
packed score/index ordering, padding keys, and top-512 truncation. A remaining
unpaired level uses the original kernel. The existing hierarchical eligibility
guards, radix-top-k option, nonhierarchical paths, and score/update/emit kernels
are unchanged. Graphs and schemas are unchanged.

The isolated qualified package is
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v25_sparse_merge2_shared`.
It retains v21--v24's qualified runtime improvements. H200 GPU3, batch-one
Generator, greedy MTP(1), fast prefill, warmed CUDA graphs, one warmup and three
measured 512-output-token requests per context:

| Prompt tokens | Repeated v24 tok/s | V25 tok/s | Decode change | V24 TTFT | V25 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 196.12 | 196.97 | +0.43% | 3.893 s | 3.892 s |
| 16,384 | 210.45 | 211.54 | +0.52% | 8.189 s | 8.187 s |
| 32,512 | 208.65 | 209.04 | +0.19% | 17.625 s | 17.621 s |

The first campaign measured control before candidate and improved decode by
+0.39% / +0.60% / +0.22%. The table reverses package order. Decode ranges are
separated at all three contexts in both campaigns. Repeat TTFT changes are
approximately -0.03% / -0.02% / -0.02%; first-campaign changes range from
-0.24% to -0.01%. Those small differences do not establish a TTFT improvement.

Matched 55-replay profiles reduce total launches from 1,405 to 1,353 per replay:
the indexer's nine merge launches become four two-level launches plus one
original launch per indexer invocation. There are 13 invocations per replay,
so merge launches decrease from 117 to 65. Scoring remains 13 launches and
standard operators remain 109 launches per replay.

| Summed GPU duration per MTP replay | V24 8K / 32K | V25 8K / 32K |
| --- | ---: | ---: |
| Sparse indexer, all kernels | 0.667 / 0.823 ms | 0.651 / 0.823 ms |
| Hierarchical merges, all levels | 0.407 / 0.457 ms | 0.389 / 0.456 ms |
| Tile scoring | 0.079 / 0.183 ms | 0.079 / 0.183 ms |
| MoE expert compute and finalization | 2.114 / 2.119 ms | 2.112 / 2.119 ms |
| Hyper-connections | 2.042 / 2.038 ms | 2.047 / 2.037 ms |
| FP8 projection GEMV | 1.730 / 1.730 ms | 1.731 / 1.730 ms |

The inclusive indexer duration improves approximately 2.4% at 8K but is
essentially flat at 32K; scoring's context-dependent growth is not addressed.
The new kernel reports 24 registers/thread, 24,576 bytes of static shared
memory, and zero local bytes/thread in the captured launch metadata. The
original merge uses 30 registers/thread and no shared memory. These are
resource observations, not proof of occupancy or stall causes. Summed durations
include overlap, dependent-launch waits, and profiler overhead, not critical-path
latency. Retention is based on both uninstrumented benchmark orders, not launch
counts alone.

An initial two-level implementation staged only intermediate lists. It removed
the same 52 launches, but 8K merge time increased from approximately 0.407 to
0.414 ms and end-to-end results were mixed (+0.12% / +0.35% / -0.11%).
That package was not promoted. Staging the input lists as well avoids repeated
global-memory binary-search loads and is the implementation qualified above.

Validation: **47 focused packed-indexer/operator-shape tests pass**, including
stable ties, partial final merge groups, top-512 boundaries, large padded
capacity, and packed multiple-row cases. Three unrelated tests also fail with
the original merge dispatch (`RejectsWrongOutputCount`,
`QsaMalformedStateLengthsAreClamped`, and `CsaMalformedStateLengthsAreClamped`);
they are recorded and excluded from the passing selection, not fixed here.
An initial randomized CPU-reference partial-group probe also failed with the
original dispatch; it was replaced by separated-score reference coverage,
with randomized exact cross-package probes retained independently.

Twelve standalone cases match repeated v24 bytes exactly for all five outputs:
selected indices, counts, key state, raw KV buffer, and state lengths.
Coverage includes FP16/FP16-state and BF16/FP32-state, one/two/three packed
rows, capacities 2,051/4,097/16,384, partial live lists, tied/random scores,
and no visible blocks. Both full benchmark campaigns match all 4,608 v5
output token IDs per package. Three default-configuration prompt sequences
also match v5. Qualification and both 128-output-token profiles report zero
MTP failures or standard fallbacks.

Qualified v20/v24 artifacts remain unchanged. Graphs, configurations, the GenAI
library, and 45 hard-linked external weight files are preserved. Final
runtime/provider hashes match the build and package manifest. Both benchmark
orders, rejected-trial evidence, exact-output checks, baseline test failures,
preservation checks, and inclusive profiles are persisted in the candidate's
`performance_sparse_merge_comparison.json`.

### Exact-order sparse indexer key reuse (performance v26)

V26 targets hierarchical QSA tile scoring. Each warp now scores one compressed
key against all four query heads, reusing its loaded key vector instead of
having four separate warps load the same key. Query vectors are prepared once
per warp. The final launch uses 16 warps (512 threads) rather than 32 warps
(1,024 threads), processing 16 keys concurrently. This follows the existing
warp-head key-reuse pattern in ragged prefill without changing that path.

The per-head four-element dot-product expression, warp reduction tree,
nonnegative clamp, float shared head scores, sequential head addition, scale,
packed stable score/index keys, tile sorting, and v25 merge implementation are
unchanged. Existing hierarchical eligibility guards are unchanged: at most
64 packed tokens, state capacity at least 2,048, head size 128, four heads,
compression ratio four, and supported bounded top-k. Models outside that path
retain their previous implementation; other models inside it use the new
kernel. No model-name checks, environment flags, schemas, or graph changes
were added. This is operator-level validation, not a claim that a broad
cross-model regression suite has been run.

The isolated qualified package is
`pkg_v12b_engine_int8kv_fp8w_fp4moe_ported_performance_v26_sparse_score_reuse16`.
H200 GPU3, batch-one Generator, greedy MTP(1), fast prefill, warmed CUDA graphs,
one warmup and three measured 512-output-token requests per context:

| Prompt tokens | Repeated v25 tok/s | V26 tok/s | Decode change | V25 TTFT | V26 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 196.93 | 197.70 | +0.39% | 3.891 s | 3.881 s |
| 16,384 | 211.52 | 212.22 | +0.33% | 8.189 s | 8.171 s |
| 32,512 | 209.02 | 210.76 | +0.83% | 17.621 s | 17.590 s |

The first campaign measured control before candidate and improved decode by
+0.42% / +0.42% / +0.81%. The table reverses package order. Decode ranges
are separated in both campaigns. An additional candidate-before-control 16K
pair improves 0.21%, also with separated ranges. Repeat TTFT decreases
approximately 0.25% / 0.21% / 0.18%; first-campaign changes range from
-0.32% to approximately zero. These small TTFT differences should not be
treated as substantial prefill recovery.

Matched 55-replay profiles retain all 1,353 launches, including 13 tile-scoring
launches and 109 standard-operator launches per replay:

| Summed GPU duration per MTP replay | V25 8K / 32K | V26 8K / 32K |
| --- | ---: | ---: |
| Hierarchical tile scoring | 0.079 / 0.183 ms | 0.061 / 0.099 ms |
| Sparse indexer, all kernels | 0.651 / 0.823 ms | 0.635 / 0.740 ms |
| MoE expert compute and finalization | 2.112 / 2.119 ms | 2.114 / 2.113 ms |
| Hyper-connections | 2.047 / 2.037 ms | 2.042 / 2.047 ms |
| FP8 projection GEMV | 1.731 / 1.730 ms | 1.729 / 1.729 ms |

Scoring summed time decreases approximately 23% at 8K and 46% at 32K;
inclusive indexer time decreases approximately 2.4% and 10.1%. Include the
surrounding costs rather than treating the scoring reduction as direct
end-to-end latency savings. Captured BF16/FP32-state scoring resources change
from 52 to 60 registers/thread, retain 768 static shared bytes, and report zero
local bytes/thread. No hardware performance counters were collected; the
memory-reuse design and resource observations do not establish a bandwidth
or occupancy diagnosis. Summed durations include overlap, dependent-launch
waits, and tracing overhead, not critical-path latency.

An eight-warp key-reuse trial also passed exact-output validation and improved
8K/32K throughput, but a supplementary 16K pair regressed 0.11%. That launch
configuration was not promoted. Increasing to 16 warps retains more inter-key
parallelism and is the binary qualified above; rejected-trial measurements are
retained separately. Final benchmarks, profiles, and parity checks all use the
same frozen 16-warp binary.

Validation: **48 focused packed-indexer/operator-shape tests pass**, including
new 4,095/4,096/4,097/8,191-live-key boundary coverage with packed multiple
requests and all supported input/state dtype pairs: FP32/FP32, FP16/FP16,
BF16/BF16, and BF16/FP32. The three pre-existing failures documented for v25
remain outside the passing selection; they were not silently fixed or counted
as passing.

Forty-eight standalone cases match repeated v25 bytes exactly for selected
indices, counts, key state, raw KV buffer, and state lengths. They cover all
four dtype pairs, one/two/three/64 packed rows, capacities up to 65,536,
partial and empty live lists, loop boundaries, and randomized/tied scores.
Both full benchmark campaigns match all 4,608 v5 token IDs per package; the
additional 16K pair matches all 1,536 reference tokens per package. Three
default-configuration v5 prompt sequences also match. Qualification and both
128-output-token profiles report zero MTP failures or standard fallbacks.

Qualified v20/v25 artifacts remain unchanged. Graphs, configurations, the GenAI
library, and 45 hard-linked external weight files are preserved. Final
runtime/provider hashes match the build and package manifest. Both campaign
orders, supplementary 16K checks, rejected-trial evidence, exact-output cases,
compatibility limits, preservation checks, and inclusive profiles are persisted
in the candidate's `performance_sparse_scoring_comparison.json`.
