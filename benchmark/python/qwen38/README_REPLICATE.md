# Other-VM Copilot handoff: reproduce optimized Qwen context results

## Objective and boundaries

Reproduce the measured **uncached, batch-one Qwen3.8 Flash NVFP4**
source-code-review/chat context sweep on an otherwise idle H200.
Run the original graphs first. Do not substitute the short-context memory
candidate or claim the selected-logits experiment is already validated.
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
This README is a later documentation-only addition.
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
capacity and other settings. Preparation/tests passed, but **long-context GPU
performance and output-token parity are not yet verified**.

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

## 7. Copilot validation checklist

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
