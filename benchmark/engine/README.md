# GenAI Engine Benchmark

Native benchmark harness for the ONNX Runtime GenAI engine. Scenarios are described in a JSON config
and results are written to per-scenario JSON files.

See [benchmark-design.md](docs/benchmark-design.md) for the architecture and
[benchmark-requirements.md](docs/benchmark-requirements.md) for the metrics contract.

**Note:** currently hardcoded for linux platform, tested on a100 linux-x64 vm

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

./build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/config.json \
  --out benchmark/engine/out
```

`--config` defaults to `config.json` and `--out` to `out`, both relative to the working directory.

Use `CUDA_VISIBLE_DEVICES=<n>` to pin the run to a specific GPU.

## Configuration

`config.json` is a list of scenario entries:

```json
[
  {
    "scenario": "decode_baseline",
    "concurrency": 1,
    "prompt_length_k": 4,
    "synthetic": true,
    "model_path": "/models/qwen2.5-0.5b-instruct",
    "execution_provider": "cuda",
    "execution_provider_library": "build/Linux/RelWithDebInfo/libonnxruntime_providers_cuda.so",
    "generation_tokens": 128,
    "warmup_runs": 2,
    "measured_runs": 10
  }
]
```

| Field | Notes |
| --- | --- |
| `scenario` | Currently only `decode_baseline`. |
| `concurrency` | Requests issued per run. One of 1, 2, 4, 8. |
| `prompt_length_k` | Approximate prompt length in thousands of tokens. |
| `synthetic` | Must be `true`; the prompt is generated rather than read from a dataset. |
| `model_path` | Folder containing the ONNX model and `genai_config.json`. |
| `execution_provider` | e.g. `cuda`. |
| `execution_provider_library` | Path to the provider plugin. Required for `cuda`, registered once per process. |
| `generation_tokens` | Tokens generated per request. |
| `warmup_runs` | Runs executed and discarded before measurement. Default 2. |
| `measured_runs` | Number of measured repetitions. Default 10. |

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
