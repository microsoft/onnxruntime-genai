# GenAI Engine Benchmark

Native benchmark harness for the ONNX Runtime GenAI engine. Scenarios are described in a JSON config,
results are written to per-scenario JSON files plus a tabbed `visualize.html`.

See [benchmark-design.md](benchmark-design.md) for the architecture and
[benchmark-requirements.md](benchmark-requirements.md) for the metrics contract.

## Build

Built as part of the normal onnxruntime-genai build:

```bash
python build.py --update --build --config RelWithDebInfo --parallel --skip_tests --skip_examples \
  --cuda_home <cuda_home>
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
    "measured_runs": 2
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
| `measured_runs` | Number of measured repetitions. |

## Output

```
out/
├── decode_baseline_results_001.json
└── visualize.html
```

Each result file contains the run status, config metadata, TTFT / inter-token latency percentiles,
per-request records, and scenario-specific metrics. `visualize.html` renders every result file found
in the output directory as a tab.
