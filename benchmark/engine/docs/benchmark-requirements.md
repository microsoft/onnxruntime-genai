# Engine Benchmark Requirements

## Current scope

The benchmark qualifies realistic paged-attention engine workloads with deterministic RULER prompts,
warmup runs, repeated measured runs, and machine-readable JSON output.

Implemented scenarios:

| Scenario | Coverage |
| --- | --- |
| `decode_baseline` | Steady decode latency and throughput at concurrency 1, 2, 4, and 8. |
| `long_prefill` | 32K, 64K, and 128K prefill latency, throughput, and memory at concurrency 1. |
| `mixed_workload` | One 128K prefill request alongside active decodes at concurrency 4 and 8. |
| `capacity_pressure` | Admission and rejection behavior for a fixed eight-prompt pressure profile. |
| `continuation` | Three appended turns at concurrency 4 and 8 to exercise session-cache reuse. |

The exact config contract and scenario constraints live in the [README](../README.md#configuration).

## Result contract

Every scenario reports:

- status and error text;
- model, runtime version, provider, concurrency, prompt, and run metadata;
- request-level TTFT, median inter-token latency, completion state, and workload role;
- TTFT p5/p50/p95 and inter-token-latency p50/p95;
- peak and steady device memory; and
- scenario-specific metrics.

Scenario-specific metrics include:

- decode end-to-end time, throughput, prompt tokens, and peak host memory;
- long-prefill duration and prompt-processing throughput;
- mixed-workload prefill TTFT and decode/prefill prompt sizes;
- capacity-pressure admitted/rejected counts and the fixed prompt profile; and
- continuation turn count and final context size.

A scenario fails when execution throws or an admitted request emits fewer tokens than requested.
Rejected admissions in `capacity_pressure` are measured results, not automatic failures.

## Deferred scope

These remain useful future benchmark extensions but are not implemented by the current harness:

- shared-prefix and partially shared-prefix workloads;
- staggered arrivals and cancellation latency;
- preemption, recomputation, and resume latency;
- scheduler queue time, fairness, and starvation gates;
- prefix-cache hit rates and cache block telemetry;
- output-token correctness comparisons; and
- CSV or HTML report generation.

New metrics should be added to `scenario_metrics` unless they apply uniformly to every scenario. New
qualification gates should produce `status: "failed"` and a useful `error` message rather than
requiring consumers to infer failure from metric values.
