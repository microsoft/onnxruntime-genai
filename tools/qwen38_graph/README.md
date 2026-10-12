# Qwen3.8-Flash-Next NVFP4: CUDA-graph decode + MTP(1) in ONNX Runtime GenAI

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
