# Reproducing Qwen3.8 Flash NVFP4 100 E2E TPS

This directory contains the Engine benchmark, exact configuration overlay,
launcher and pooled-result validator used to reproduce the optimized
8,192-input / 512-output workload. See [RESULTS.md](RESULTS.md) for all
optimization gains and [the detailed log](../../../docs/Qwen38MtpEngramIssues.md)
for operator validation and rejected experiments.

## Measured result and requirements

| Metric | Final result |
|---|---:|
| E2E throughput, ten requests | 100.481 tokens/s |
| Decode throughput | 126.419 tokens/s |
| Mean TTFT | 1.053 s |
| Mean request time | 5.095 s |
| Sampled peak GPU memory, separate diagnostic | 100.09 GiB |
| Observed peak host RSS, separate diagnostic | 3.61 GiB |

The two five-request invocations reached 100.167 and 100.798 E2E TPS.
Individual requests ranged from 98.22 to 104.92 TPS. This is a measured
aggregate, not a throughput guarantee.

Use Linux x86-64, one otherwise idle H200 (SM90), enough host RAM/disk for
building and loading the model, and the **same original
`qwen_38_flash_nvfp4_engine` model package**, including its tokenizer,
target NVFP4 graph, FP8 MTP graph, Engram graph and all external weights.
The model is not included or downloaded by these scripts. Another export,
tokenizer, prompt or model configuration is not an exact reproduction.
The earlier INT4 package described in the detailed log is not this model.

The measured environment used Python 3.12.15, CUDA 13.3.73 and NVIDIA driver
580.105.08. cuDNN headers/libraries and the usual repository build
dependencies are required. The runtime identified itself as ORT 1.31.0
and GenAI 0.18.0.dev0. New builds may have different binary hashes and timing.
Both tensor-attention flags and GEMM tuning are opt-in: tensor PV rounds
probabilities to FP16, and full-model lossless equivalence or broad quality
accuracy is not established.

## 1. Check out both optimization branches

Both branches were created from their respective existing
`kvaishnavi/qwen-38-flash` HEADs:

| Repository | Branch | Parent commit |
|---|---|---|
| ONNX Runtime | `asonawane/qwen-38-flash-100-e2e` | `55b790ace9593bb71a1ae66a8effbda34d05fbd8` |
| GenAI | `asonawane/qwen-38-flash-100-e2e` | `1b6a385902bd11b356b5d91b51d92455f3477161` |

```bash
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime.git
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime-genai.git
export ORT_SRC="$PWD/onnxruntime"
export GENAI_SRC="$PWD/onnxruntime-genai"
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/path/to/cudnn
```

Set the CUDA/cuDNN paths to real installations. Do not use a placeholder
unchanged. Record `git rev-parse HEAD` in both checkouts with each result.

## 2. Build and install into an isolated environment

Activate a dedicated Python 3.12 environment. Follow each repository's
dependency instructions before building; do not replace a shared runtime.
For an existing configured ORT build, an incremental provider rebuild
alone is sufficient only if its core/Python bindings already match this
branch. For a fresh build, build the matching wheel and shared libraries:

```bash
cd "$ORT_SRC"
./build.sh --config Release --update --build --parallel 8 \
  --build_shared_lib --build_wheel --use_cuda \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
    onnxruntime_USE_FP4_QMOE=ON onnxruntime_USE_FP8_QMOE=ON
python -m pip install --no-deps build/Linux/Release/dist/*.whl
```

The `--skip_tests` option avoids a full test suite during build, not an
assertion that tests passed. Later C++ optimization regressions were not
compiled/run in the recorded experiment; executed validation used
standalone numerical cases, graph replay and full requests.

GenAI's `--ort_home` needs an SDK layout, **not** the raw ORT build root.
Stage matching headers and libraries:

```bash
export ORT_HOME="$ORT_SRC/ort-sdk-qwen38"
mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$ORT_HOME/include/"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.inc "$ORT_HOME/include/"
cp -a "$ORT_SRC"/build/Linux/Release/libonnxruntime*.so* "$ORT_HOME/lib/"

cd "$GENAI_SRC"
python build.py --config Release --update --build --parallel \
  --use_cuda --cuda_home "$CUDA_HOME" --ort_home "$ORT_HOME" --skip_tests
find build/Linux/Release -name '*.whl'
```

Install the generated GenAI CUDA wheel from the path printed by `find`:

```bash
python -m pip install --no-deps /actual/path/to/onnxruntime_genai_cuda.whl
```

Keep the staged SDK in place: native library RPATHs may reference it.
Check the reported loaded libraries after running, not merely package
version numbers. A stock ORT wheel lacks these provider changes.

## 3. Run two independent five-request invocations

Set `MODEL` to the original model directory and `OUTPUT` to a writable
directory outside the source tree. Each invocation loads its own model,
excludes one warmup and retains all five measured requests. A fresh Engine
is created for every request so the prompt is not reused from prefix cache.
Cold tuning can make the excluded warmup much slower than measured requests.

```bash
export MODEL=/absolute/path/to/qwen_38_flash_nvfp4_engine
export OUTPUT=/absolute/path/to/qwen38-results
mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES=0
export PYTHON="$(command -v python)"
bash "$GENAI_SRC/benchmark/python/qwen38/run_benchmark.sh" \
  "$MODEL" "$OUTPUT/run1.json"
bash "$GENAI_SRC/benchmark/python/qwen38/run_benchmark.sh" \
  "$MODEL" "$OUTPUT/run2.json"
python "$GENAI_SRC/benchmark/python/qwen38/summarize_results.py" \
  "$OUTPUT/run1.json" "$OUTPUT/run2.json" --require-100
```

The launcher sets both tensor-attention flags before model initialization.
The overlay enables CUDA graphs, 8K scheduling, cache utilization 0.05,
width-two MTP and decoder/head GEMM tuning. It does not modify model files.
Do not use `--step-timings`, memory markers or profiler capture for the
throughput runs above.

The prompt must tokenize to SHA256
`91620c2b4341bcf39f996f412f5428de25b97f24fac17cc0d72018f4cbad0236`.
The validator checks this hash, exact token counts, the overlay, flags,
draft width and zero failures. It prints loaded-library paths for inspection.
Ensure they point to your rebuilt packages/staged SDK, not a stock runtime.

E2E TPS = total measured output tokens / total request seconds.
Decode TPS = total measured tokens after the first token / total decode
seconds. TTFT, EOS/minimum-token handling and greedy selection are preserved.
Do not discard slow measured requests or average invocation TPS numbers.
If the pooled result is below 100, the validator exits nonzero rather than
claiming success. Hardware load, arithmetic-dependent acceptance and cold
unseen-shape tuning can affect the result.

## Original local deployment

The original isolated provider hash was
`103a10ceae4aaac2ae5dbdb0c6f40c35c012dcc3a8dfd1491b9a5fb564e035d2`.
The fresh-build instructions above replace the original machine-specific
session-artifact deployment; these exact fresh builds have not been rerun
as part of publishing the branch. The checked-in benchmark is copied from
the measured script with explicit model/CUDA-runtime arguments and output
directory creation; request timing and execution remain the same.
