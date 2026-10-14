# Other-VM Copilot handoff: reproduce optimized Qwen context results

## Objective and boundaries

Reproduce the measured **uncached, batch-one Qwen3.8 Flash NVFP4**
source-code-review/chat context sweep on an otherwise idle H200.
Run the original graphs first. Do not substitute the short-context memory
candidate or treat workload-specific selected-logits validation as universal.
Do not overwrite another user's environment, model, results, or GPU workload.
Keep all samples; never filter for acceptance or throughput targets.

All published changes use **`asonawane/qwen-38-flash-100-e2e` in both repositories**.

| Repository | Pinned source commit | Purpose |
|---|---|---|
| microsoft/onnxruntime | `16289abceb0eb1d2330bca6e8aa1de708d227f86` | Optimized CUDA provider and memory kernels |
| microsoft/onnxruntime-genai | `2a69912474e29275ad2cccb01bcd96593ea29c7f` | Matching runtime, graph preparation, portable benchmark scripts |

The measured GenAI native runtime was built at
`37e2b4773655427afa493098c7f44cb593cff0fa`. The pinned later GenAI commit
adds Python scripts/documentation without changing that native runtime.
Selected-logits preparation was added in separate commit `25ac6ecb`.
These pins reproduce the historical table. The experimental prefix-cache and
warp-heads follow-ups below require newer source and freshly rebuilt libraries.
Do **not** mix these with newer `kvaishnavi/qwen-38-flash` HEADs or stock wheels.
The API32 compatibility patch used for the kvaishnavi comparison is not
needed for this pinned optimized runtime.

## Prerequisites

- Linux x86-64, Python 3.12, a dedicated environment.
- H200 / SM90 with enough free VRAM: this long-context sweep sampled about
  92-98 GiB, so it is **not an 80 GiB workload**.
- Original model export with decoder NVFP4 weights, FP8 MTP, Engram,
  tokenizer, chat template, and all external weights.
  Model files are not in Git and this procedure does not download them.
- CUDA/cuDNN/compiler/CMake/Ninja and each repository's build dependencies.
  Reference machine: CUDA 13.3.73, driver 580.105.08, GCC 13.2.
  A different toolchain may change results or require compatibility work.
- Enough disk for source/builds and the model. Selected-logits preparation
  additionally copies the external weights and hashes them; allow substantial
  disk space and time.

Use real absolute paths instead of the placeholders below.

## 1. Get the pinned sources

Use new clone directories, not an unrelated dirty checkout:

```bash
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime.git onnxruntime-qwen
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime-genai.git onnxruntime-genai-qwen

export ORT_SRC="$PWD/onnxruntime-qwen"
export GENAI_SRC="$PWD/onnxruntime-genai-qwen"
git -C "$ORT_SRC" switch --detach 16289abceb0eb1d2330bca6e8aa1de708d227f86
git -C "$GENAI_SRC" switch --detach 2a69912474e29275ad2cccb01bcd96593ea29c7f
git -C "$ORT_SRC" submodule update --init --recursive
git -C "$GENAI_SRC" submodule update --init --recursive
```

## 2. Build and install matching wheels

Activate/create a dedicated Python 3.12 environment first. Follow repository
dependency instructions; install missing dependencies into this environment
only. Do not install a stock ORT GPU wheel to resolve a local-runtime problem.

```bash
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/absolute/path/to/cudnn
unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd "$ORT_SRC"
./build.sh --config Release --update --build --parallel 8 \
  --build_shared_lib --build_wheel --use_cuda \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --skip_tests --cmake_generator Ninja \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
    onnxruntime_USE_FP4_QMOE=ON onnxruntime_USE_FP8_QMOE=ON
python -m pip install --no-deps build/Linux/Release/dist/*.whl

export ORT_HOME="$ORT_SRC/ort-sdk-qwen38"
mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$ORT_HOME/include/"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.inc "$ORT_HOME/include/"
cp -a "$ORT_SRC"/build/Linux/Release/libonnxruntime*.so* "$ORT_HOME/lib/"

cd "$GENAI_SRC"
python build.py --config Release --update --build --parallel \
  --use_cuda --cuda_home "$CUDA_HOME" --ort_home "$ORT_HOME" \
  --skip_tests --cmake_generator Ninja \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90
find build/Linux/Release -name '*.whl'
```

Install the GenAI CUDA wheel printed by `find`:

```bash
python -m pip install --no-deps /actual/path/to/onnxruntime_genai_cuda.whl
export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import sys, onnxruntime as ort, onnxruntime_genai as og; print(sys.executable); print(ort.__version__, ort.__file__); print(og.__file__)'
```

Keep the SDK directory in place. If cuDNN uses `lib64` or another layout,
adjust the library path accordingly. Configure failures must be resolved
explicitly, not by mixing SDK headers/providers from another branch.
`--skip_tests` is not a statement that runtime tests passed.

## 3. Confirm the original model and idle GPU

```bash
export MODEL=/absolute/path/to/qwen_38_flash_nvfp4_engine
sha256sum "$MODEL/genai_config.json"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

Reference original configuration SHA256:
`bc7316907c666870c2065729f746d498306c69854f54d2ffcd367ee700b170fa`.
Reference original external weight SHA256:

| File | SHA256 |
|---|---|
| model.onnx.data | `493cba9faace381022c749fc210a87c9d5553795ffed303f74aa7dbd13376928` |
| mtp.onnx.data | `11f178b98e15cd889a9fbc5e570e76b0f71637d27e13a63cee97a1363af65f47` |
| engram.onnx.data | `b910f6c322efea60a5d641c42d5c518e8a57c19b336b78d9606c921320a0b993` |

A changed export/config is not an exact reproduction. Hashing these large
weight files takes time. Never kill another user's process to free a GPU.

## 4. Run the measured long-context workload

```bash
python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model "$MODEL" \
  --output-dir /absolute/path/to/new-original-context-results \
  --gpu 0
```

`--gpu` is the **physical nvidia-smi index**. The output directory must not
exist and must be outside the model directory.

The runner sets these variables before model loading:

```bash
ORT_QMOE_NVFP4_PACKED_PREFILL=1
ORT_QMOE_NVFP4_FUSED_SWIGLU=1
ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH=1
ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT=1
ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1
ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS=0
```

No manual flag exports are needed for this runner. It preserves explicit
`PYTHONPATH`/`LD_LIBRARY_PATH` for deliberate runtime discovery, but does not
forward unrelated inherited ORT debug/calibration overrides.

Fixed protocol:

- Contexts 8192, 16384, 32768, 65536, 131072; MTP widths 1, 3, 5, 7.
- Source-code-review **chat** prompt built from repeated Python worker modules.
  It is not the older plain distributed-inference-service prompt.
- Batch one, greedy, 512 output tokens.
- Fresh process per configuration, fresh Engine/request per run.
- One excluded warmup plus three measured runs; no reused prompt cache.
- CUDA graphs and decoder/MTP GEMM tuning enabled.
- Scheduler 8192; KV-cache budget 0.05 for 8K-64K and 0.12 for 128K.
- Nominal 20 ms device-wide VRAM sampling, including load/warmup and timing.
  This is sampled peak, not a hard cap; occupancy is checked before each row,
  not continuously.
- Mean TTFT; pooled decode TPS = total 511-token counts / total decode time;
  pooled E2E TPS = total 512-token counts / total request time.
- Acceptance = accepted/evaluated; first-draft acceptance =
  (partial + full acceptance rounds)/completed rounds; proposal yield =
  accepted/proposed. Target 75-85% is descriptive, not a filter.

Outputs: `TABLE.md`, `manifest.json`, per-row JSON, logs, overlays and memory
samples. Failures remain explicit and produce a nonzero exit.

## 5. Expected reference results

MTP width 3 produced the highest measured E2E TPS at each context:

| Requested context | Actual prompt | TTFT (ms) | Decode TPS | E2E TPS | Sampled peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8190 | 1226.31 | 129.52 | 99.00 | 94340 |
| 16384 | 16384 | 2517.40 | 134.54 | 81.07 | 94336 |
| 32768 | 32768 | 5329.79 | 132.43 | 55.72 | 94342 |
| 65536 | 65536 | 11748.26 | 131.11 | 32.72 | 94332 |
| 131072 | 131072 | 27779.95 | 121.41 | 16.01 | 100490 |

E2E counts generated tokens only and includes prompt processing. Long
contexts therefore have lower E2E TPS even when decoding stays fast.
These are measured aggregates, not strict thresholds or performance guarantees.
Fresh hardware/builds may differ. Matching prompts do not establish numerical
or answer-quality equivalence; output hashes differed from the kvaishnavi baseline.

## 6. Selected-logits experiment: separate from reproduction

This newly published option projects only requested hidden-state rows through
the LM head. It preserves the original 262144 configured context, indexer
capacity and other settings. Preparation/tests passed. A subsequent fresh
64K/128K MTP-width-three comparison measured the following results, with one
excluded warmup and three measured 512-token requests per configuration:

| Context | Original TTFT (s) | Selected-logits TTFT (s) | Original E2E TPS | Selected-logits E2E TPS | Sampled VRAM saved |
|---:|---:|---:|---:|---:|---:|
| 65536 | 11.747 | 11.630 | 32.42 | 32.68 | 4 GiB |
| 131072 | 27.733 | 27.510 | 16.13 | 16.24 | 4 GiB |

All six measured candidate token streams exactly matched their paired
controls. This single control-then-candidate comparison shows a useful
memory reduction and a small observed latency improvement, not a robust
speedup guarantee or broad answer-quality qualification. Other context/MTP
configurations remain unverified for the transformed graphs.

A separate CUDA-profiler capture of 128K candidate prefill attributed 27.3%
of summed GPU kernel time to sparse indexer scores, 18.5% to sparse paged
attention, and 26.4% to packed NVFP4 expert GEMMs. These are kernel-time
shares, not wall-clock speedup predictions. Profiling timings were excluded
from the paired throughput results.

```bash
python "$GENAI_SRC/benchmark/python/qwen38/prepare_memory_candidate.py" \
  --source "$MODEL" \
  --destination /absolute/path/to/new-long-context-selected-logits-model \
  --preserve-context

python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model /absolute/path/to/new-long-context-selected-logits-model \
  --output-dir /absolute/path/to/new-selected-logits-context-results \
  --gpu 0
```

Preparation requires `onnx`, copies external weights independently, checks
SHA256, and refuses an existing destination. Use a separate model/output path.
Compare prompt hashes, overlays, full `token_ids` streams, TTFT/decode/E2E,
acceptance and memory against fresh original-graph controls. Report mismatches;
do not claim a speedup or quality preservation from preparation tests alone.
An existing matching runtime does not need a kernel rebuild for this option.

Do not omit `--preserve-context`: default preparation creates the bounded
8K memory candidate, configured at 10240 with a previously measured 9217-total-
token Engine ceiling. That package cannot reproduce the long-context table.
The historical `run_benchmark.sh` and `benchmark_qwen_mtp.py` are 8K/512 tools,
not the context-sweep entry point.

## 7. Experimental prefix-cache and warp-heads follow-ups

### Source selection for the follow-up results

Both repositories publish these follow-ups on
`asonawane/qwen-38-flash-100-e2e`. Unlike historical-table reproduction in
section 1, select the following source before rebuilding both wheels/SDK
using section 2:

```bash
git -C "$ORT_SRC" fetch origin asonawane/qwen-38-flash-100-e2e
git -C "$GENAI_SRC" fetch origin asonawane/qwen-38-flash-100-e2e
git -C "$ORT_SRC" switch --detach bfd45538f44fe0e72dbb08350ae98d84bed67f53
git -C "$GENAI_SRC" switch --detach origin/asonawane/qwen-38-flash-100-e2e
git -C "$ORT_SRC" submodule update --init --recursive
git -C "$GENAI_SRC" submodule update --init --recursive
git -C "$ORT_SRC" rev-parse HEAD
git -C "$GENAI_SRC" rev-parse HEAD
```

Record the resolved GenAI commit: the remote branch may move after publication.
Its source must include the explicit batch-one MTP prefix guard, Python
`get_prefix_cache_stats()` API, and `--prefix-cache` sweep option.
The historical GenAI pin does not include these changes.

### Experimental MTP prefix reuse

Explicit `engine.dynamic_batching.prefix_caching=true` now admits batch-one
MTP target-prefix reuse when no DFlash drafter or sliding-window KV layout
is present. Implicit/default MTP prefix behavior remains unchanged.
Target paged KV and fixed-state checkpoints are adopted; each new request
starts a cold MTP shadow at its first generated token, as in the uncached path.
This is not restored draft-prefix state and is not qualified for concurrent
batching or all models.

With the rebuilt GenAI library/Python bindings and warp-heads provider:

```bash
python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model /absolute/path/to/long-context-selected-logits-model \
  --output-dir /absolute/path/to/new-cached-results \
  --gpu 0 --indexer-warp-heads --prefix-cache
```

Unlike the historical uncached benchmark, one Engine is shared across the
cold warmup and repeated identical requests. The cold request is excluded
from cached timing; cache-fill cost is not free. `get_prefix_cache_stats()`
records per-request adoption counts and the benchmark requires exact output
parity with its cold request. Speculative counters are differenced per request,
not pooled repeatedly from cumulative Engine telemetry.

The completed 20-row local experiment matched all 60 measured outputs to
their cold requests with zero MTP failures. Adopted tokens per request were
0/8192/24576/57344/122880 for requested 8K/16K/32K/64K/128K prompts.
The retained checkpoint leaves the final scheduler chunk to recompute, so
8K has no cache hit. Full repeated-request results follow. The actual 8K
prompt has 8190 tokens; all other prompts have exactly the requested length.
VRAM is device-wide sampled peak including load and warmup, not a hard peak
guarantee. Acceptance target is 75-85%; no samples were filtered.

| Requested prompt | Prompt length | MTP | TTFT (ms) | Decode TPS | E2E TPS | MTP acceptance | First-draft acceptance | Proposal yield | Within target | Sampled peak VRAM (MiB) | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|
| 8192 | 8190 | 1 | 1195.55 | 118.96 | 93.24 | 80.21% | 80.21% | 80.21% | yes | 91262 | completed |
| 8192 | 8190 | 3 | 1197.86 | 135.17 | 102.85 | 71.27% | 75.00% | 53.75% | no | 91268 | completed |
| 8192 | 8190 | 5 | 1198.44 | 121.15 | 94.53 | 68.30% | 74.43% | 38.17% | no | 91424 | completed |
| 8192 | 8190 | 7 | 1200.24 | 104.22 | 83.89 | 66.73% | 70.52% | 27.92% | no | 91422 | completed |
| 16384 | 16384 | 1 | 1247.50 | 114.97 | 89.95 | 76.47% | 76.47% | 76.47% | yes | 91262 | completed |
| 16384 | 16384 | 3 | 1249.64 | 139.70 | 104.33 | 74.31% | 75.94% | 57.86% | no | 91274 | completed |
| 16384 | 16384 | 5 | 1251.89 | 130.82 | 99.26 | 72.46% | 74.53% | 43.70% | no | 91416 | completed |
| 16384 | 16384 | 7 | 1252.68 | 112.31 | 88.23 | 70.52% | 75.16% | 32.51% | no | 91458 | completed |
| 32768 | 32768 | 1 | 1320.96 | 116.34 | 89.62 | 80.85% | 80.85% | 80.85% | yes | 91262 | completed |
| 32768 | 32768 | 3 | 1321.83 | 138.61 | 102.23 | 74.31% | 77.01% | 57.86% | no | 91276 | completed |
| 32768 | 32768 | 5 | 1323.40 | 123.01 | 93.47 | 70.02% | 75.88% | 40.21% | no | 91416 | completed |
| 32768 | 32768 | 7 | 1319.85 | 109.43 | 85.48 | 69.37% | 80.00% | 31.54% | no | 91440 | completed |
| 65536 | 65536 | 1 | 1440.96 | 115.72 | 87.42 | 79.93% | 79.93% | 79.93% | yes | 91246 | completed |
| 65536 | 65536 | 3 | 1444.15 | 133.57 | 97.15 | 72.73% | 77.89% | 56.34% | no | 91292 | completed |
| 65536 | 65536 | 5 | 1444.73 | 118.66 | 89.02 | 69.69% | 76.16% | 39.39% | no | 91432 | completed |
| 65536 | 65536 | 7 | 1444.16 | 100.21 | 78.25 | 67.46% | 74.71% | 28.81% | no | 91474 | completed |
| 131072 | 131072 | 1 | 1670.61 | 111.62 | 81.94 | 78.32% | 78.32% | 78.32% | yes | 96380 | completed |
| 131072 | 131072 | 3 | 1668.68 | 133.28 | 93.04 | 75.41% | 76.34% | 58.45% | yes | 96398 | completed |
| 131072 | 131072 | 5 | 1670.77 | 122.48 | 87.63 | 71.60% | 77.91% | 42.70% | no | 96518 | completed |
| 131072 | 131072 | 7 | 1673.45 | 105.49 | 78.56 | 70.44% | 78.85% | 32.87% | no | 96596 | completed |

MTP-three comparison against the same selected-logits/warp-heads variant
with fresh uncached Engines:

| Requested context | Uncached TTFT (ms) | Cached TTFT (ms) | Uncached E2E TPS | Cached E2E TPS | E2E gain |
|---:|---:|---:|---:|---:|:---|
| 8192 | 1201.22 | 1197.86 | 104.06 | 102.85 | No cache hit |
| 16384 | 2422.92 | 1249.64 | 84.95 | 104.33 | 1.23x |
| 32768 | 5009.35 | 1321.83 | 57.70 | 102.23 | 1.77x |
| 65536 | 10534.68 | 1444.15 | 35.44 | 97.15 | 2.74x |
| 131072 | 22932.74 | 1668.68 | 18.87 | 93.04 | 4.93x |

These are cached repeated-prefix results, not uncached prefill speedups.
Changing the prompt, eviction pressure, chunk size, or concurrency requires
fresh validation.

### Experimental warp-local indexer scoring

ORT commit `bfd45538f44fe0e72dbb08350ae98d84bed67f53` adds default-off
`ORT_PACKED_SPARSE_INDEXER_WARP_HEADS=1`, retaining query heads within a warp
to reuse keys and remove scoring barriers. Rebuild the matching CUDA provider
before testing it. The sweep runner accepts `--indexer-warp-heads` explicitly;
omitting it keeps the historical reproduction unchanged.

Two fresh flag-off/on comparisons, in opposite execution orders, used the
same selected-logits model and MTP width three:

| Context | Control TTFT (s), two invocations | Candidate TTFT (s), two invocations | Control E2E TPS | Candidate E2E TPS |
|---:|---:|---:|---:|---:|
| 65536 | 11.615 / 11.620 | 10.515 / 10.494 | 33.13 / 32.95 | 35.25 / 35.36 |
| 131072 | 27.534 / 27.511 | 22.904 / 22.895 | 16.18 / 16.35 | 18.93 / 18.88 |

Standalone FP16/FP32 tests match all outputs exactly across ten geometries
per type, including 32768 live slots. The first full-model pair matched six
measured streams. The reverse-order 64K pair also matched; the reverse-order
128K control differed from both candidates and the earlier control.
This observed control variability prevents a claim of full-model deterministic
equivalence or broad quality preservation. No samples were discarded.
The option remains experimental, not a default change or 100-E2E-TPS solution.

A further controlled 128K pair set `ep.cuda.enable_gemm_auto_tune` to `"0"`
in both decoder and MTP session options. All three measured token streams
matched exactly. TTFT was 27.517 seconds flag-off versus 22.935 flag-on;
decode was 124.16 versus 124.19 TPS and E2E was 16.19 versus 18.93 TPS.
This supports parity and a prefill-specific gain on that workload, without
proving that autotuning caused the earlier mismatch. Keep this diagnostic
separate from the historical tuned table.

## 8. Copilot validation checklist

1. Record source commits, toolchain, GPU, active interpreter and package paths.
2. Confirm all loaded runtime libraries/hashes in the manifest correspond to
   the locally built optimized runtime. Stock libraries invalidate the run.
3. Require 20 configurations and 60 measured completed requests, 512 output
   tokens each, zero MTP failures, expected context/MTP/overlay and prompt hashes.
4. Keep warmups excluded but saved; never remove slow or low-acceptance samples.
5. For 128K, require Engine `max_request_length >= actual_prompt + 512`.
   Earlier 5%/7% budgets failed admission, including after warmup; do not
   silently shorten prompts or count warmup results instead.
6. Compare manifests and raw counters, not just rounded table cells.
7. If GPUs are occupied or inference fails, preserve logs and report blocked/
   failed status. Do not mark performance validation complete.

CPU regression checks (with matching runtime packages plus `onnx`/`pytest`):

```bash
python -m pytest \
  "$GENAI_SRC/benchmark/python/qwen38/test_context_sweep.py" \
  "$GENAI_SRC/benchmark/python/qwen38/test_prepare_memory_candidate.py" -q
```

Publication checks passed 17 CPU tests and all five actual prompt hashes
matched the original measured sweep. The portable runner was not freshly
GPU-rerun before publication because the local GPUs were occupied.

For kernel history and the separate approximately 100 E2E TPS / below-80-GiB
8K memory experiment, see [README_MEMORY.md](README_MEMORY.md).
For additional build/background details, see [README.md](README.md).
