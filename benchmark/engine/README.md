# GenAI Engine Benchmark

Native benchmark harness for the ONNX Runtime GenAI engine. Scenarios are described in a JSON config
and results are written to per-scenario JSON files.

See [benchmark-design.md](docs/benchmark-design.md) for the architecture and
[benchmark-requirements.md](docs/benchmark-requirements.md) for the metrics contract.

The dependency staging flow currently supports Linux x64 with CUDA.

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

`requirements-dev.txt` provides the Python build dependencies, including `requests`. Install
`patchelf` separately because the benchmark build uses its command-line tool while staging runtime
libraries.

Run these commands from the `onnxruntime-genai` repository root. The CUDA Toolkit and a C++20
compiler must also be installed separately for a CUDA build.

## Build

Opt in with `--build_engine_benchmark`; it is not built by default:

```bash
python build.py --update --build --config RelWithDebInfo --parallel --skip_tests --skip_examples \
  --use_cuda --cuda_home <cuda_home> --build_engine_benchmark
```

The build stages the benchmark's runtime dependencies next to the executable in
`build/Linux/RelWithDebInfo/benchmark/engine/`:

- ONNX Runtime and the CUDA plugin EP, downloaded at the versions pinned in
  `tools/python/util/dependency_resolver.py` and cached under the staged
  `build/Linux/RelWithDebInfo/benchmark/engine/dependencies/` directory
- the locally built `libonnxruntime-genai.so` and `libonnxruntime-genai-cuda.so`

Delete that `dependencies/` directory to force a re-download after changing the pinned versions.

`patchelf` must be on `PATH` (`pip install patchelf`) so the staged GenAI libraries load the pinned
ONNX Runtime rather than the one baked into their build-time RPATH.

## Run

```bash
export LD_LIBRARY_PATH=<cuda_home>/lib64:$PWD/build/Linux/RelWithDebInfo/benchmark/engine

python benchmark/engine/run.py \
  --executable build/Linux/RelWithDebInfo/benchmark/engine/engine_benchmark \
  --config benchmark/engine/configs/config.json \
  --out benchmark/engine/out \
  --cuda_visible_devices 0 \
  --verbose
```

Use `run.py` for configs containing multiple scenarios. It runs each entry in a separate
`engine_benchmark` process, so CUDA, ONNX Runtime allocators, and the paged-cache capacity check
start cleanly for every scenario. The wrapper preserves numbered result files such as
`decode_baseline_results_001.json` and `long_prefill_results_002.json`. It replaces the output
directory at the start of each invocation, prints a completion/failure summary, and returns nonzero
if any scenario fails.

To run scenarios in parallel, pass a comma-separated GPU list such as `0,1,2,3`. Each scenario
waits for a GPU slot and receives one device through `CUDA_VISIBLE_DEVICES`.

The runner requires `--executable`, `--config`, `--out`, and `--cuda_visible_devices`. Verbose
child benchmark output is opt-in with `--verbose`.

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
    "generation_tokens": 64
  }
]
```

| Field | Default | Notes |
| --- | --- | --- |
| `scenario` | `decode_baseline` | Scenario name from the table below. |
| `concurrency` | `1` | Number of requests issued per measured run; scenarios further restrict it. |
| `prompt_length_k` | none | RULER prompt bucket in thousands of tokens. Required except for `capacity_pressure`. |
| `model_path` | required | Folder containing the ONNX model and `genai_config.json`; `~` is expanded. |
| `execution_provider` | `cuda` | Execution provider used by the model. |
| `execution_provider_library` | beside executable | Optional plugin override. CUDA and WebGPU default to their staged provider library beside `engine_benchmark`. |
| `generation_tokens` | `64` | Tokens generated per request; some scenarios require a fixed value. |
| `warmup_runs` | `5` | Runs excluded from reported metrics. May be zero. |
| `measured_runs` | `20` | Runs included in reported metrics; must be positive. |

The checked-in configs intentionally omit `execution_provider_library`; use it only when the
provider plugin is not staged beside the executable.

## Scenarios

| Scenario | Concurrency | Prompt length | Generation tokens | Purpose |
| --- | --- | --- | --- | --- |
| `decode_baseline` | 1, 2, 4, or 8 | Required | Configurable | Steady decode TTFT, inter-token latency, end-to-end time, and throughput. |
| `long_prefill` | 1 | 32K, 64K, or 128K | 1 | Prefill TTFT, prompt-processing throughput, and memory scaling. |
| `mixed_workload` | 4 or 8 | Required for active decode requests | Configurable for decode; long prefill uses 1 | One 128K prefill alongside active decode requests; summary TTFT covers decode requests. |
| `capacity_pressure` | 8 | Must be omitted | 1 | Admission pressure using fixed 4K, 4K, 32K, 32K, 48K, 64K, 96K, and 128K prompts. |
| `continuation` | 4 or 8 | Required | Configurable | Three appended turns per logical request to measure session-cache reuse. |

`mixed_workload` records the prefill request's TTFT separately in `scenario_metrics.prefill_ttft_ms`.
Core TTFT percentiles contain only active decode requests, avoiding a mixed prefill/decode
population. Inter-token latency covers all emitted inter-token gaps.

`continuation` runs three appended turns for each logical request. Each turn submits the previous
turn's generated tokens as part of the next prompt, so the benchmark measures session-cache reuse
under concurrency 4 and 8.

`capacity_pressure` measures explicit admission under memory pressure. Admitted requests generate
one token, rejected admissions are reported in `scenario_metrics`, and preemption is not modeled.

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
per-request records, and scenario-specific metrics. A scenario exception or any request that emits
fewer tokens than requested sets `status` to `failed` and makes the process return nonzero.

Device memory is sampled on a background thread while the scenario runs. NVML is loaded lazily, so
`peak_device_memory_mb` and `steady_state_device_memory_mb` are 0 on machines without an NVIDIA
driver. When the driver reports per-process usage those numbers are attributed to this process;
otherwise they are the device-wide growth since before the model was loaded, so run on an otherwise
idle GPU for meaningful values. Note that ONNX Runtime and the CUDA driver cache allocations, so
these measure reserved rather than live memory.
