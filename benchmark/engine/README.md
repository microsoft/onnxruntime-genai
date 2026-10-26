# GenAI Engine Benchmark

Native benchmark harness for the ONNX Runtime GenAI engine. Scenarios are described in a JSON config
and results are written to per-scenario JSON files.

See [benchmark-design.md](docs/benchmark-design.md) for the architecture and
[benchmark-requirements.md](docs/benchmark-requirements.md) for the metrics contract.

Supports Linux/CUDA and macOS/WebGPU. WebGPU requires a compatible ONNX Runtime host and
provider plugin, plus a model exported for WebGPU.

## Environment setup

Create and activate a Conda environment, then install the Python dependencies used by the
ONNX Runtime GenAI build and the `patchelf` utility required to rewrite the staged libraries'
RPATH:

```bash
conda create -n engine-benchmark-venv python=3.11 -y
conda activate engine-benchmark-venv

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -r benchmark/requirements.txt

python --version
# Linux only:
patchelf --version
```

On Linux, the benchmark requirements file provides the `patchelf` dependency used while staging
the benchmark's runtime libraries. macOS does not use `patchelf`.

Run these commands from the `onnxruntime-genai` repository root. The CUDA Toolkit and a C++20
compiler must also be installed separately for a CUDA build.

## Build

Opt in with `--build_engine_benchmark`; it is not built by default:

```bash
python build.py --update --build --config RelWithDebInfo --parallel --skip_tests --skip_examples \
  --build_engine_benchmark --cuda_home <cuda_home>
```

The build stages the benchmark's runtime dependencies next to the executable in
`build/Linux/RelWithDebInfo/benchmark/engine/`:

- ONNX Runtime and the CUDA plugin EP, downloaded at the versions pinned in
  `tools/python/util/dependency_resolver.py` and cached under `benchmark/engine/dependencies/`
- the locally built `libonnxruntime-genai.so` and `libonnxruntime-genai-cuda.so`

Delete the `dependencies/` folder to force a re-download after changing the pinned versions.

`patchelf` must be on `PATH` (`pip install patchelf`) so the staged GenAI libraries load the pinned
ONNX Runtime rather than the one baked into their build-time RPATH.

### macOS / WebGPU build

Use a local ONNX Runtime SDK with `include/` and `lib/libonnxruntime.dylib`, and a matching
`libonnxruntime_providers_webgpu.dylib`. The SDK and plugin must target the same ORT API.
An Apple Clang toolchain, CMake, and the normal GenAI build dependencies are required; CUDA is not.

```bash
python build.py --update --build --config Release --parallel --skip_tests --skip_examples \
  --build_engine_benchmark --ort_home /path/to/ort-sdk
```

The build stages the local GenAI and ORT dylibs and a `versions.json` containing the runtime's
reported version in `build/macOS/Release/benchmark/engine/`. It does not download Linux/CUDA
packages. The executable locates its staged libraries via `@loader_path`; provide the WebGPU
plugin path in the scenario config or via the runner override below. Custom `--build_dir` values
change the executable's parent directory accordingly.

### Prefix-cache and scheduler microbenchmark

The model-free prefix-cache benchmark and CPU scheduler benchmark are built into
`engine_unit_tests` and disabled during normal test runs. Build the tests in `Release` or
`RelWithDebInfo`, then run:

```bash
build/Linux/Release/engine_unit_tests \
  --gtest_also_run_disabled_tests \
  --gtest_filter='PrefixCacheBenchmark.*:SchedulerBenchmark.*'
```

`PrefixCacheBenchmark` fills a 16,384-block cache with 128 independent 4,096-token prefixes before
timing cold misses, partial hits, and full hits. `SchedulerBenchmark` reports the CPU time spent
planning steady-state decode steps at batch sizes 1, 8, and 32. That planning time is one component
of inter-token latency; use the model-backed `decode_baseline` scenario for end-to-end inter-token
latency, which also includes model execution, sampling, synchronization, and event delivery.

## Run

```bash
export LD_LIBRARY_PATH=<cuda_home>/lib64:$PWD/build/Linux/RelWithDebInfo/benchmark/engine

python benchmark/engine/run.py \
  --executable build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/configs/config.json \
  --out benchmark/engine/out \
  --cuda_visible_devices 0,1,2,3
```

Use `run.py` for configs containing multiple scenarios. It runs each entry in a separate
`engine_benchmark` process, so CUDA, ONNX Runtime allocators, and the paged-cache capacity check
start cleanly for every scenario. The wrapper preserves numbered result files such as
`decode_baseline_results_001.json` and `long_prefill_results_002.json`.

For a single scenario, the executable can still be run directly:

```bash
build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/configs/config.json \
  --out benchmark/engine/out
```

To run scenarios in parallel across selected GPUs, pass a comma-separated list. Each scenario
waits for an available GPU, acquires its per-GPU slot, and receives that GPU through
`CUDA_VISIBLE_DEVICES`, so no scenario uses more than one GPU:

```bash
python benchmark/engine/run.py \
  --executable build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/configs/config.json \
  --out benchmark/engine/out \
  --cuda_visible_devices 0,1,2,3
```

The runner requires `--executable`, `--config`, and `--out`. CUDA scenarios additionally require
`--cuda_visible_devices`. WebGPU scenarios run serially on the default adapter and do not set
`CUDA_VISIBLE_DEVICES`. The runner saves a log alongside every result, prints child errors on
failure, and treats missing/failed result files as failures. `--verbose` prints successful child
output as well.

### WebGPU decode/prefill smoke

`configs/webgpu-decode-prefill-smoke.json` contains only the smallest existing decode/prefill
matrix entries: concurrency 1, a 4K RULER decode prompt (64 generated tokens), and a 32K RULER
prefill prompt (one generated token). Each has one warmup and three measured runs. The RULER bucket
is a nominal length; results report actual token counts for the model's tokenizer.

The example defaults refer to the sibling Qwen3.8-27B WebGPU/DFlash2 width-3 export and ORT main
WebGPU build used for this smoke test. Override the paths for your checkout:

```bash
BENCH="$PWD/build/macOS/Release/benchmark/engine"
ORT_LIB_PATH="$BENCH/libonnxruntime.dylib" \
  python benchmark/engine/run.py \
    --executable "$BENCH/engine_benchmark" \
    --config benchmark/engine/configs/webgpu-decode-prefill-smoke.json \
    --model_path /path/to/webgpu-model \
    --execution_provider_library /path/to/libonnxruntime_providers_webgpu.dylib \
    --out benchmark/engine/out-webgpu --verbose
```

Paths in JSON are relative to the working directory, not the config file. Run these examples from
the repository root. `--model_path` and `--execution_provider_library` override every scenario's path.
`ORT_LIB_PATH` explicitly selects the matching staged host, rather than an inherited runtime path.

The smoke config uses `model_config` overlays to reserve 256 paged-cache blocks, limit the batch to
one, disable prefix caching between repeats, and keep DFlash2 at three drafts when present. The
larger cache fits the 32K prefill and the drafter's cache without editing the exported model. For
other models, adjust the block budget and speculative settings as needed. An overlay cannot add
a missing drafter graph; DFlash2 must already be exported in the model.

Use normal WebGPU routing for these scenarios. Do not globally force the experimental small-M
SIMT diagnostic: it cannot execute large-M prefill. These smoke results are ORT main/WebGPU, not
the earlier benchmarks that switched to strict SIMT only after prefill.

### WebGPU prefill scaling

`configs/webgpu-prefill-scaling.json` measures exact 4,096, 8,192, 16,384, and 32,768-token
prefixes of the same deterministically selected 32K RULER fixture. This controls prompt content
across lengths and avoids assuming the fixture's nominal bucket matches the model's tokenizer.
The config uses `long_prefill`, one generated token, concurrency 1, one discarded warmup and three
measured runs per length. Cache capacity, scheduling budget and DFlash2 settings are identical at
every length, and prefix caching is disabled.

Use the same build, runtime and model/provider overrides as the smoke example:

```bash
ORT_LIB_PATH="$BENCH/libonnxruntime.dylib" \
  python benchmark/engine/run.py \
    --executable "$BENCH/engine_benchmark" \
    --config benchmark/engine/configs/webgpu-prefill-scaling.json \
    --model_path /path/to/webgpu-model \
    --execution_provider_library /path/to/libonnxruntime_providers_webgpu.dylib \
    --out benchmark/engine/out-webgpu-prefill-scaling --verbose
```

Use `scenario_metrics.prefill_ms` for measured TTFT samples and `scenario_metrics.prompt_tokens`
for the graph's actual token counts. Plot the median with a measured min/max range, not a
confidence interval from only three samples. Check `prompt_truncated` before interpreting a
requested length as measured. These measurements exclude model loading and tokenization, and
include the prefill/first-token engine work. One-token requests measure target prefill rather than
speculative decode speedup.

## Configuration

The `configs/` directory contains the complete matrix in `config.json`, individual scenario
matrices in `decode-baseline.json`, `long-prefill.json`, `mixed-workload.json`,
`capacity-pressure.json`, and `continuation.json`, and a smoke test in `smoke-test.json`.

Each config is a list of scenario entries:

```json
[
  {
    "scenario": "decode_baseline",
    "concurrency": 1,
    "prompt_length_k": 4,
    "model_path": "/models/qwen2.5-0.5b-instruct",
    "execution_provider": "cuda",
    "execution_provider_library": "build/Linux/RelWithDebInfo/libonnxruntime_providers_cuda.so",
    "generation_tokens": 64
  }
]
```

| Field | Notes |
| --- | --- |
| `scenario` | `decode_baseline`, `long_prefill`, `mixed_workload`, `capacity_pressure`, or `continuation`. |
| `concurrency` | Requests issued per run. One of 1, 2, 4, 8; `long_prefill` requires 1. |
| `prompt_length_k` | RULER prompt length in thousands of tokens; active decode length for `mixed_workload`. Required by scenarios that use it and rejected by `capacity_pressure`, which has a fixed prompt profile. |
| `prompt_tokens` | Optional positive exact prefix length for `long_prefill`. Must fit within the tokenized source fixture selected by `prompt_length_k`; the normal model-session limit can still truncate it. Other scenarios reject this field. |
| `model_path` | Folder containing the ONNX model and `genai_config.json`. |
| `execution_provider` | e.g. `cuda` or `webgpu`. |
| `execution_provider_library` | Path to the provider plugin. Required for `cuda` and `webgpu`, registered once per process. |
| `generation_tokens` | Tokens generated per request. |
| `warmup_runs`, `measured_runs` | Defaults: 5 warmups, 20 measured runs. Smoke config uses 1 and 3. |
| `model_config` | Optional GenAI config overlay applied before model creation; effective overrides are recorded in results. Useful for cache capacity, scheduler budgets, and speculation. |

`long_prefill` and `mixed_workload` truncate a RULER prompt that would leave no room for generation
within the model-configured session limit, reporting both requested and actual token counts.
`long_prefill` also reports `source_prompt_tokens`, the tokenized fixture length before an optional
`prompt_tokens` prefix is selected.
`capacity_pressure` does not truncate; over-ceiling prompts are reported as rejected admissions.

`mixed_workload` runs one long-prefill request alongside active decode requests. The full and
focused matrices request a 128K prefill at concurrency 4 and 8; the prompt is capped to leave room
for generation within the model's configured session limit. The smoke test uses the smallest 0.5B,
concurrency-4 entry. In this scenario, the long-prefill request is intentionally capped to one
generated token while decode requests keep `generation_tokens`; this keeps the prefill request from
pushing max-length/context usage into unstable CUDA/KV-pressure territory while still measuring
prefill-vs-decode interference.

`continuation` runs three appended turns for each logical request. Each turn submits the previous
turn's generated tokens as part of the next prompt, so the benchmark measures session-cache reuse
under concurrency 4 and 8.

`capacity_pressure` submits eight concurrent prompts that ramp from 32K toward 128K. This first
version measures explicit admission under memory pressure: admitted requests generate one token,
and rejected admissions are reported in `scenario_metrics`. Preemption is intentionally not modeled
yet and will be added in a later benchmark iteration.

## Adding a scenario

Scenarios self-register with `ScenarioBase::Create`, so the dispatcher needs no changes:

1. Create `scenarios/my_scenario.h`/`.cpp` with a class inheriting `ScenarioBase` (see
   `decode_baseline.h`/`.cpp` for reference).
2. At file scope in the `.cpp`, add:
   ```cpp
   static const ScenarioBase::Registrar<MyScenario> kRegistrar("my_scenario");
   ```
3. Add both files to `engine_benchmark_srcs` in `CMakeLists.txt`.

A config entry with `"scenario": "my_scenario"` will then dispatch to it automatically.

## Output

```
out/
└── decode_baseline_results_001.json
```

Each result file contains the run status, config metadata, TTFT / inter-token latency percentiles,
per-request records, and scenario-specific metrics.

Device memory is sampled on a background thread while the scenario runs. NVML is loaded lazily, so
`peak_device_memory_mb` and `steady_state_device_memory_mb` are 0 on machines without an NVIDIA
driver. When the driver reports per-process usage those numbers are attributed to this process;
otherwise they are the device-wide growth since before the model was loaded, so run on an otherwise
idle GPU for meaningful values. Note that ONNX Runtime and the CUDA driver cache allocations, so
these measure reserved rather than live memory.

On macOS, WebGPU device-memory telemetry is unavailable: its device-memory fields remain
0, **not an indication of zero GPU usage**, and `config_metadata.device_memory_source` identifies
the unavailable source. On macOS, `scenario_metrics.peak_host_memory_mb` reports process peak RSS
using the platform's byte-valued `ru_maxrss`. On Apple Silicon this is not a separate measurement
of Metal allocations or a device-memory substitute.

`decode_baseline.scenario_metrics.tokens_per_s` is end-to-end generated throughput including
prefill, not decode-only throughput. With speculative decoding, tokens arrive in bursts, so median
inter-token latency can be near zero; do not invert it to estimate decode throughput.

Runner regression tests (no model/GPU required):

```bash
python -m unittest discover -s benchmark/engine -p 'test_*.py'
```
