## Benchmarking workstream

Build a reusable benchmark tool as part of the baseline, before performance-oriented Engine
changes. It must exercise realistic agent workloads rather than report aggregate throughput only.

The tool should support:

- 1, 2, 4, and 8 concurrent requests with configurable RULER prompt and generation lengths;
- deterministic RULER prompt fixtures for reproducible capacity and representative latency tests;
- shared-prefix, partially shared-prefix, and unrelated-prefix workloads;
- staggered arrivals, request cancellation, continuation, and memory-pressure scenarios;
- prefill-only, decode-only, and mixed prefill/decode phases;
- ~~fixed seeds and captured outputs so performance changes also detect correctness regressions;~~
  **(Decided later: deprioritized, see note below)**
- warmup and repeated measured runs with JSON/CSV output suitable for before/after comparison;
- configurable model, execution provider, block count/size, token budget, chunk size, and
  concurrency without modifying the benchmark source.

Record at minimum:

- request-level TTFT, end-to-end latency, generated tokens/second, and cancellation latency;
- p50/p95/p99 inter-token latency, including active decode latency while another request prefills;
- scheduler queue time, per-step token count, prefill/decode split, and fairness/starvation;
- peak and steady device memory, model/workspace/cache bytes, blocks used/free, prefix-cache hit
  rate, preemption count, recomputed tokens, preemption resume latency, and peak transient
  prefill-chunk activation/workspace memory;
- ~~failures, rejected admissions, output mismatches, and incomplete requests.~~
  **(Decided later: keep failures/rejected admissions/incomplete requests; output mismatches deprioritized, see note below)**
- failures, rejected admissions, and incomplete requests.

The primary benchmark matrix is:

| Scenario | Concurrency | Prompt length | Purpose |
| --- | ---: | ---: | --- |
| Decode baseline | 1, 4, 8 | 4K | Steady-state inter-token overhead |
| Long prefill | 1 | 32K, 64K, 128K | TTFT, peak memory, chunk scaling |
| Mixed workload | 4, 8 | One 128K prefill plus active decodes | Responsiveness and fairness |
| Shared coding context | 4, 8 | 32K-128K shared prefix | Prefix-cache value |
| Capacity pressure | 8 | Growing toward 128K each | Admission now; preemption later |
| Cancellation | 4, 8 | Cancel during long prefill | 500 ms cancellation target |
| Continuation | 4, 8 | Repeated appended turns | Session-cache reuse |

Initial performance gates:

- all admitted requests complete correctly without Engine-wide failure;
- cancellation latency is at most 500 ms at p95;
- active decode p95 inter-token latency during another request's prefill is at most 1.5x its
  steady decode baseline;
- last-token LM-head work makes prefill logits memory independent of prompt length;
- no request starves, and memory-pressure admission is explicit and reproducible;
- under the mixed workload, every runnable decode request is scheduled at least once within a
  configured maximum scheduling delay; report the bound and fail the run when it is exceeded;
- ~~a fixed-seed request produces the same token stream alone and when co-scheduled with unrelated
  requests;~~ **(Decided later: deprioritized, see note below)**
- benchmark results include enough environment metadata to compare commits and model builds.

Implement the benchmark as a native C++ executable using the public C++ interface and C ABI where
ABI coverage is required. Python may orchestrate optional experiments or visualize saved results,
but must not be required to run or qualify the benchmark. Back it with native Engine telemetry;
do not derive core scheduler/cache measurements from Python wall-clock timing alone.

> **Note (post-review decision):** Full correctness verification (captured/hashed outputs,
> fixed-seed token-stream comparison across standalone and co-scheduled runs, output-mismatch
> detection) is deprioritized for now in favor of performance work; the crossed-out items above
> reflect the original design intent. We still detect and report **incomplete requests**
> (a request generating fewer tokens than requested) via a `completed` flag per request in
> `raw_requests`, which now fails the run's `status` instead of silently reporting `success`.

> **Note (capacity pressure scope):** The current `capacity_pressure` scenario implements admission
> coverage only. It submits memory-pressure prompts, records admitted requests and rejected
> admissions, and leaves preemption/resume behavior for a later benchmark iteration.
