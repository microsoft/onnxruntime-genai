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
python -m pip install -r benchmark/requirements.txt

python --version
patchelf --version
```

The benchmark requirements file must be installed because it provides the `patchelf` dependency
used while staging the benchmark's runtime libraries.

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
timing cold misses, short two-block hits, partial hits, and full hits. The short hit measures the
same content-addressed lookup used when verified draft tokens complete a reusable target block;
this model-free benchmark measures CPU lookup overhead, not end-to-end target/drafter speedup.
`SchedulerBenchmark` reports the CPU time spent
planning steady-state decode steps at batch sizes 1, 8, and 32. That planning time is one component
of inter-token latency; use the model-backed `decode_baseline` scenario for end-to-end inter-token
latency, which also includes model execution, sampling, synchronization, and event delivery.

For model-backed DFlash prefix reuse, run `draft-prefix-benchmark.py` with two
model directories containing the same model: one with
`engine.dynamic_batching.prefix_caching` enabled and one with it disabled.
The script runs distinct prompts sharing a prefix, verifies the same prompt
produces identical greedy output with caching on and off, and reports cached
prompt tokens, draft forwards, TTFT, and total time. To isolate auxiliary replay from target-prefix reuse, compare otherwise
identical builds with and without DFlash auxiliary retention; merely disabling
prefix caching removes both optimizations.

The benchmark defaults to two distinct prompts with exactly 40,000 shared
leading token IDs and different short suffixes. It checks output parity for
each prompt separately and requires the warm prompt to reuse at least the
full blocks within the shared prefix:

```bash
python benchmark/engine/draft-prefix-benchmark.py \
  --enabled-model <prefix-enabled-model> --disabled-model <prefix-disabled-model> \
  --shared-prefix-tokens 40000 --generated-tokens 16 --repetitions 2
```

On one A100-SXM4-80GB GPU with a Qwen3.8 27B DFlash 2 model, each prompt
had 40,012 tokens; the second reused 39,936 tokens (156 full 256-token
blocks). Its total time was about 20.6 s with prefix caching disabled,
420 ms with target-only reuse (no drafting after the hit), and 307 ms
with target and drafter checkpoints (7 draft forwards). TTFT was 110 ms
target-only versus 117 ms with checkpoints. These numbers measure the older
four-ring checkpoint approach, not auxiliary replay: four FP16 drafter
checkpoint rings at this model's geometry cost about 200 MiB of extra GPU
allocation.

With auxiliary replay on the same A100 and model, a four-prompt run reused
39,936 tokens on each of three warm prompts. All four greedy outputs matched
the uncached runs. The warm prompts took 599-664 ms, with 7-9 draft forward
passes and TTFT around 440 ms; the uncached runs took 20.6-21.2 s. The cold
cached run took 25.0 s. These are single-GPU measurements of synthetic,
repetitive prompts, not throughput or concurrency results. One earlier
three-prompt run with checkpoint restoration failed greedy-output parity;
other runs passed, and four passing auxiliary-replay prompts do not establish
parity for every model and input.

After a 40,000-token cold prompt, separate-process snapshots measured
approximately 4,610 MiB host RSS and 67,740 MiB per-process GPU memory
with auxiliary replay, versus 2,677 MiB host RSS and 68,866 MiB GPU memory
with the four-ring checkpoint build. The observed host increase of about
1,933 MiB exceeds the GPU decrease of about 1,126 MiB; automatic target-pool
sizing and allocator behavior also affect the GPU figures. The packed
auxiliary rows in this model cost roughly 50 KiB per indexed token, so
retaining 39,936 tokens alone approaches 1,950 MiB. Its projected five-layer
FP16 draft K/V is only 20 KiB per token based on the configured 8 KV heads
and 128-wide heads. Thus auxiliary replay covers more prefix boundaries
than four ring snapshots, but is **not** the smallest-total-memory
representation for this model. Per-block projected K/V retention would need
a separate implementation and correctness check. This target also has fixed
state groups: a shorter prompt may adopt an earlier available fixed-state
checkpoint rather than all matching full KV blocks. Retaining auxiliary rows
for blocks that cannot themselves be adopted is another possible memory
reduction, but would require keeping exactly the DFlash window preceding each
usable fixed-state checkpoint. In a separate five-prompt run, prompts cut at
8,192, 15,872, and 25,600 tokens adopted a 3,840-token fixed-state
checkpoint, while a prompt cut at 39,936 adopted 39,936 tokens. All four
warm outputs matched uncached greedy output and continued drafting. A small
available hit does not guarantee a speedup: the 25,600-token prompt took
12.4 s with replay versus 12.3 s uncached.

For a no-restoration comparison, the original base commit's target-only
build reused 39,936 tokens in about 420 ms on a warm prompt, with zero
draft forward passes. A separate-process snapshot after the cold prompt
measured approximately 2,706 MiB host RSS and 69,790 MiB GPU memory.
Those GPU snapshots are not a fixed-pool memory comparison: the target pool
is automatically sized from free memory and differs between builds. More
importantly, the target-only build's third prompt in a four-prompt run
produced different greedy tokens from the uncached control. Neither this
timing nor four passing auxiliary-replay prompts establish universal
correctness; the target-only mismatch needs separate investigation.

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
| `model_path` | Folder containing the ONNX model and `genai_config.json`. |
| `execution_provider` | e.g. `cuda`. |
| `execution_provider_library` | Path to the provider plugin. Required for `cuda`, registered once per process. |
| `generation_tokens` | Tokens generated per request. |

`long_prefill` and `mixed_workload` truncate a RULER prompt that would leave no room for generation
within the model-configured session limit, reporting both requested and actual token counts.
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
