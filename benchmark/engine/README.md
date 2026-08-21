# GenAI Engine Benchmark

Native benchmark harness for the ONNX Runtime GenAI engine. Scenarios are described in a JSON config
and results are written to per-scenario JSON files.

See [benchmark-design.md](docs/benchmark-design.md) for the architecture and
[benchmark-requirements.md](docs/benchmark-requirements.md) for the metrics contract.

**Note:** currently hardcoded for linux platform, tested on a100 linux-x64 vm

## Environment setup

Create and activate a Conda environment, then install the Python dependencies used by the
ONNX Runtime GenAI build and the `patchelf` utility required to rewrite the staged libraries'
RPATH:

```bash
conda create -n engine-benchmark-venv python=3.11 -y
conda activate engine-benchmark-venv

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install patchelf

python --version
patchelf --version
```

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

## Run

```bash
export LD_LIBRARY_PATH=<cuda_home>/lib64:$PWD/build/Linux/RelWithDebInfo/benchmark/engine

python benchmark/engine/run.py \
  --executable build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/config.json \
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
  --config benchmark/engine/config.json \
  --out benchmark/engine/out
```

To run scenarios in parallel across selected GPUs, pass a comma-separated list. Each scenario
waits for an available GPU, acquires its per-GPU slot, and receives that GPU through
`CUDA_VISIBLE_DEVICES`, so no scenario uses more than one GPU:

```bash
python benchmark/engine/run.py \
  --executable build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/config.json \
  --out benchmark/engine/out \
  --cuda_visible_devices 0,1,2,3
```

The runner requires `--executable`, `--config`, `--out`, and `--cuda_visible_devices`. Verbose
child benchmark output is opt-in with `--verbose`.

## Configuration

`config.json` is a list of scenario entries:

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
| `scenario` | `decode_baseline`, `long_prefill`, or `mixed_workload`. |
| `concurrency` | Requests issued per run. One of 1, 2, 4, 8; `long_prefill` requires 1. |
| `prompt_length_k` | RULER prompt length in thousands of tokens; active decode length for `mixed_workload`. |
| `model_path` | Folder containing the ONNX model and `genai_config.json`. |
| `execution_provider` | e.g. `cuda`. |
| `execution_provider_library` | Path to the provider plugin. Required for `cuda`, registered once per process. |
| `generation_tokens` | Tokens generated per request. |

`mixed_workload` runs one long-prefill request alongside active decode requests. The full matrix
uses a hardcoded 128K prefill at concurrency 4 and 8; the smoke test uses one generated token.

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
