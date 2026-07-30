# GenAI Engine Benchmark Design

## Goal
Provide a reproducible native benchmark harness for ONNX Runtime GenAI that runs scenario-driven tests, captures latency/throughput metrics, and writes human-readable outputs for regression tracking.

## High-Level Architecture
```mermaid
flowchart LR
	A[config.json] --> C[scenario_dispatcher.cpp]

	C --> D[decode_baseline.cpp]
    C --> E[long_prefill.cpp]
    C --> G[other_scenarios.cpp]

	D --> H[decode_baseline/results.json]
	D --> I[decode_baseline/visualize.html]
    E --> J[long_prefill/results.json]
    E --> K[long_prefill/visualize.html]
	G --> N[other_scenarios/results.json]
	G --> O[other_scenarios/visualize.html]
```

## File Structure
Current benchmark engine layout:

```text
onnxruntime-genai/benchmark/engine/
|- .gitignore
|- benchmark-design.md
|- benchmark-requirements.md
|- scenario_dispatcher.cpp
|- CMakeLists.txt
|- config.json
|- data/
|   |- README.md
|   |- ruler/
|       |- prompts.json
|- README.md
|- scenarios/
    |- decode_baseline.cpp
    |- long_prefill.cpp
    |- {other_scenarios}.cpp
    |- utils.cpp
```

## Data Setup (RULER)

For reproducible long-context benchmarks, we keep a pinned prompt subset (adapted from https://github.com/NVIDIA/RULER) in this repository.

Example prompts.json:
```json
{
  "dataset": "RULER",
  "version": "subset_v1",
  "source_branch": "rulerv2-ns",
  "description": "Single-file curated subset with 5 prompts per prompt-length bucket.",
  "length_buckets": {
    "4k": [
      "[RULER 4K sample 0] Full prompt text goes here.",
      "[RULER 4K sample 1] Full prompt text goes here.",
      "[RULER 4K sample 2] Full prompt text goes here.",
      "[RULER 4K sample 3] Full prompt text goes here.",
      "[RULER 4K sample 4] Full prompt text goes here."
    ],
    "32k": [
      "[RULER 32K sample 0] Full prompt text goes here.",
      "[RULER 32K sample 1] Full prompt text goes here.",
      "[RULER 32K sample 2] Full prompt text goes here.",
      "[RULER 32K sample 3] Full prompt text goes here.",
      "[RULER 32K sample 4] Full prompt text goes here."
    ],
    "64k": [
      "[RULER 64K sample 0] Full prompt text goes here.",
      "[RULER 64K sample 1] Full prompt text goes here.",
      "[RULER 64K sample 2] Full prompt text goes here.",
      "[RULER 64K sample 3] Full prompt text goes here.",
      "[RULER 64K sample 4] Full prompt text goes here."
    ],
    "128k": [
      "[RULER 128K sample 0] Full prompt text goes here.",
      "[RULER 128K sample 1] Full prompt text goes here.",
      "[RULER 128K sample 2] Full prompt text goes here.",
      "[RULER 128K sample 3] Full prompt text goes here.",
      "[RULER 128K sample 4] Full prompt text goes here."
    ]
  }
}
```

## Scenario Dispatcher Responsibilities

The scenario dispatcher is responsible for:

- validating that required config entries/fields are present (without enforcing scenario-specific semantic validity)
- iterating over multiple scenario entries in a single config file
- dispatching each entry to the appropriate scenario implementation and coordinating multiple scenario runs per invocation

## Scenario Responsibilities
Each scenario implementation file (under `scenarios/`) is responsible for:

- validating that inputs are valid for that scenario
- running the benchmark and recording scenario-appropriate metrics
- producing that scenario's results.json and visualize.html outputs

## Data Contract (ie. config.json)
`model_uri` is used below as a short placeholder for the model location.
Example full value: https://foundrylocalmodels.blob.core.windows.net/staging/qwen2.5-0.5b-instruct

- Input config fields per entry:

```json
{
    "scenario": "decode_baseline", // choices=[other scenarios, ...]
    "concurrency": 1, // choices=[1, 2, 4, 8]
    "prompt_length_k": 4, // choices=[4, 32, 64, 128]
    "synthetic": false, // choices=[true, false]
    "model_path": $model_uri,
    "execution_provider": "cuda"
}
```

- Output artifacts:
	- results.json: run status, summary percentiles, and raw request-level records.
	- visualize.html: local chart/table view over results.json.

## Metrics Strategy

Core Metrics (all scenarios)

- config metadata:
	- scenario, model_path, execution_provider, concurrency, prompt length, synthetic
- run status:
	- status, error (if any)
- request-level latency/throughput:
	- ttft_ms, e2e_ms, generated_tokens_per_s
- summary percentiles:
	- ttft_p50_ms, ttft_p95_ms
	- e2e_p50_ms, e2e_p95_ms
	- tokens_per_s_p50
- memory baseline:
	- peak_memory_bytes (required for all scenarios)

Scenario-Specific Metrics (optional extensions)

- ex. decode_baseline:
	- output text capture for correctness checks
	- optional inter-token latency distribution details

Example results.json shape (decode_baseline):

```json
{
    "scenario": "decode_baseline",
    "config_metadata": {
        "model_path": $model_uri,
        "execution_provider": "cuda",
        "concurrency": 4,
        "prompt_length_k": 4,
        "synthetic": true,
        "generation_tokens": 128,
        "measured_runs": 2
    },
    "status": "success",
    "error": null,
    "core_metrics": {
        "summary": {
            "ttft_ms": {
                "p50": 0,
                "p95": 0
            },
            "e2e_ms": {
                "p50": 0,
                "p95": 0
            },
            "tokens_per_s": {
                "p50": 0
            },
            "peak_memory_bytes": 0
        },
        "raw_requests": [
            {
                "request_id": 0,
                "ttft_ms": 0,
                "e2e_ms": 0,
                "generated_tokens_per_s": 0
            }
        ]
    },
    "scenario_metrics": {
        "outputs": [],
        "inter_token_latency_ms": {
            "p50": 0,
            "p95": 0
        }
    }
}
```

This hybrid model keeps cross-scenario comparison clean while still allowing each scenario to report what matters most.

## Build and Runtime
- Build system: CMake + C++17.
- Links against onnxruntime-genai and onnxruntime libraries.
- Runs as a native C++ executable (no Python required to execute).
