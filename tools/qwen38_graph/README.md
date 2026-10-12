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
