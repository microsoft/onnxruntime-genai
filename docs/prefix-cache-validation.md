# Prefix-cache validation

Use `examples/python/engine/prefix-cache-qa.py` to check a release model's
prefix-cache correctness and reuse on A100/H100 or the intended deployment GPU.
The matrix checks three separate contracts:

| Contract | Requirement |
|---|---|
| Safety | A hit represents the same token history, KV ownership, and, for hybrid models, fixed state. The final prompt token still executes. |
| Correctness | Cached, uncached, and concurrently scheduled requests produce the same bounded greedy token sequence. An optional drafter snapshot must not change target hit lengths. |
| Progress | Once conflicting owners are released and capacity is available, a completed longer request can publish a deeper reusable boundary. Repeated requests must not remain stranded behind an intermediate duplicate block. |

A large hit is not proof of correct state. Correct output alone is not proof that
prefill was skipped. Low latency alone is not proof of either.

## Before running

Check that the selected GPU is available and use the Python environment with
the intended GenAI build, model package, and provider stack. This is an
operator-run tool, not a CI job; it does not download models, modify packages,
reserve GPUs, or stop other processes.

Profiles use recorded `Config.overlay()` settings. Omit `--execution-provider`
to preserve the package's providers and graph-capture options. A fixed block
budget is required: use the package's explicit `num_blocks` or pass
`--num-blocks`. Automatic free-memory sizing is not reproducible across fresh
Engines.

## Matrix coverage

The default matrix covers:

| Suite | Sequence and assertion |
|---|---|
| `boundaries` | Prompt lengths immediately before/on/after KV-block and prefill-chunk boundaries; fresh run, exact warm replay, another replay |
| `order` | Ascending and descending lengths sharing the same system corpus; each completed request is immediately replayed |
| `branching` | Single-token divergence before/on/after block and checkpoint boundaries; branch replay, then return to the original history and replay it |
| `alternation` | Warm two distinct histories, then alternate repeatedly; require a deep hit on every switch, not merely on immediate repeats |
| `concurrency` | Both short/long admission orders and single-row execution while a completed short request holds its state slot; subsequent sequential repair and warm replay; repeated cache-disabled admission controls |
| `leases` | Keep a completed original request open while executing a conflicting branch; release the original owner, repair, and require deeper reuse |
| `cancellation` | Cancel after an Engine run during partial prefill, drain the terminal event, then retry and replay |
| `pressure` | Separate bounded-pool profile; distinct first-block histories whose aggregate publishable prefix blocks exceed the configured pool; require observable eviction/checkpoint turnover, then recovery |
| `random` | Seeded divergence positions and token mutations, each followed by replay |

Every prompt has a **cache-disabled, sequential greedy reference** with the
same generation policy and drafter. The first output remains the baseline;
references repeat twice by default. `--reference-repeats 1` leaves stability
unchecked. Unstable references fail the run even if a cached output matches one
repeat.

Parity means exact output-token equality, not decoded-text similarity.
Non-cancelled turns must reach `--generated-tokens`; early EOS fails coverage.
The runner does not impose a minimum-token floor because that would suppress
speculative proposals.

Concurrency controls repeat each admission sequence with caching **disabled**
in a fresh Engine, preserving request order, retained owners, and releases.
`uncached_controls` checks sequential-reference parity; `control_stability` and
row-level `repeat_parity` separately check schedule repeatability. One repeat
leaves control stability unchecked. These controls do not waive reference or
cached failures, and matching schedules need not have identical execution
geometry once a cached request adopts a prefix.

Prompts use complete system/user chat templates and a shared system corpus
(`--prompt-file` can supply one). Branch/pressure probes mutate valid token IDs:
these test cache state, not answer quality. Lengths too short for the template
fail explicitly; use larger `--lengths` for those packages.

## Running the matrix

### Quick boundary/order check

PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
python examples\python\engine\prefix-cache-qa.py `
  --model-path C:\models\qwen-hybrid-dflash `
  --num-blocks 1024 --suites boundaries order `
  --device-label A100-80GB --build-label YOUR_RUNTIME_BUILD `
  --output prefix-cache-boundaries.json
```

On the Linux GPU machines:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash \
  --num-blocks 1024 --suites boundaries order \
  --device-label H100 --build-label YOUR_RUNTIME_BUILD \
  --output prefix-cache-boundaries.json
```

The report records the imported runtime module and version; an operator-supplied
`--build-label` alone does not prove binary identity.

### Full sequence matrix and configuration sweeps

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --chunk-sizes 256 512 1024 --max-batch-size 8 \
  --draft-mode both --seed 20261002 --random-cases 8 \
  --output prefix-cache-full.json
```

`--draft-mode both` runs separate configured-drafter and target-only profiles,
each with its own uncached reference. Target-only disables MTP and clears the
DFlash2/DSpark filename; parity between drafter modes is not assumed.

Repeat the sequential suites with `--max-batch-size 1` to exercise the smallest
hybrid checkpoint pool. Select suites that do not require multiple resident
requests:

```bash
python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --max-batch-size 1 --suites boundaries order branching cancellation pressure random \
  --output prefix-cache-one-checkpoint.json
```

Also qualify a paged-only model and the unmodified release chunk size. Chunk
sizes must fit the graph's fixed query limit and scheduler token budget.
Different KV-block sizes require model packages exported for that geometry.

### Long-context checks

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --lengths 4097 14126 45884 104705 --suites boundaries order \
  --output prefix-cache-long-context.json
```

Start branching/pressure with small lengths; long branches require many uncached
prefills. Requests beyond Engine capacity fail rather than being shortened.

### Two long histories without cache thrashing

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --lengths 45884 --suites alternation --alternation-split 14079 \
  --alternation-rounds 4 --draft-mode both \
  --output prefix-cache-alternating-46k.json
```

`--alternation-split` selects the divergence token. After immediate warm replays,
every A/B switch must retain a deep hit. Both histories must fit the block
budget; hybrid alternation needs at least two fixed-state checkpoint slots.
Insufficient capacity fails explicitly. Use the other sequential suites for
the one-checkpoint profile.

Append `--plan-only` to preview profiles, lengths, mutations, and actions without
loading GenAI or a model. `--pressure-num-blocks` controls the separate churn
budget; churn count grows to exceed it. Regular and pressure models are loaded
sequentially.

## Reading the results

JSON reports include runtime/configuration identity, overlays, Engine capacities,
prompt hashes, output token IDs, hit lengths, timings, actions, finish reasons,
and per-phase speculative counters. Treat token IDs as potentially sensitive
generated content.

Only `status: passed` is a pass. Failures/interruption produce a nonzero exit
and preserve the partial report; there are no missing-model skips or parity
waivers. `--timeout-seconds` and `--max-run-calls` bound host-side progress
checks, not a GPU call that itself hangs.

Assertions are collected separately for safety, parity, generation coverage,
reuse, and optional latency. `safety` checks public hit range/alignment, not
internal ownership or fixed-state identity. `null` means unchecked or not
applicable, not passed. `--fail-fast` stops at the first assertion; execution,
lifecycle, or unsafe-boundary failures always stop the run.

Reuse checks account for the adopted cursor and later chunk endpoints. Hybrid
hits require a block-aligned checkpoint before the final prompt token;
paged-only replay can reuse all preceding complete blocks. Repair after a
branch or owner release must reach the newly publishable boundary, but does
not assume older histories survive eviction.

Windowed DFlash2 must resume proposals after recomputing a complete window.
Without a draft snapshot, a deep target hit may leave too few tokens to rebuild
that window during a short decode; target reuse alone does not prove drafting
recovered. Drafter failures/disables fail the run. Concurrent speculative
counters are per phase, not per request.

Timing is diagnostic unless `--max-warm-ttft-ratio` gates warm/cold TTFT.
Warmup, scheduling, thermals, providers, and other GPU users affect it.
First reasoning-content time measures Engine token delivery, not application
streaming or visible-answer latency.

Before blaming prefix adoption for parity failure, inspect uncached admission
controls, repeat in fresh processes, and run target-only. Stable sequential
output does not prove batch-shape invariance. A failure with caching disabled
shows adoption is not a necessary trigger, not which operation caused it.
Reversed admission and the pinned-slot control help separate batching from
state-slot effects. Preserve seed/profile/prompt hashes and minimize failures
into named regressions; never waive them as numerical noise.

## Measuring checkpoint protection in scheduler planning

`PagedKeyValueCache::PlanStepResources()` queries reclaimable prefix capacity
on every planned step. The opt-in component benchmark measures that exact
`PrefixCache::ReclaimableBlocks()` query with real fixed-state checkpoints from
the checked-in hybrid test model:

```powershell
.\engine_unit_tests.exe --gtest_also_run_disabled_tests `
  --gtest_filter=PrefixCacheBenchmark.DISABLED_HybridCheckpointReclamationScaling:PrefixCacheBenchmark.DISABLED_SparseHybridCheckpointReclamation:PrefixCacheBenchmark.DISABLED_SharedEvictedCheckpointAncestry
```

On Linux, use `./engine_unit_tests` with the same arguments. This benchmark is
disabled in ordinary test runs because timings are diagnostic, not portable
pass/fail thresholds; the lease-safety regressions run normally.

The matrix uses 256-token blocks, histories of 128/512/2,048 blocks
(32,768/131,072/524,288 tokens), and 1/8/32 checkpoint histories, limiting the
aggregate pool to 16,384 blocks. Each geometry measures no leases, half the
histories leased, and all histories leased. The `batch` column is checkpoint
history count, not a measured execution batch; `matched_blocks` is aggregate
indexed blocks. Each row checks the exact reclaimable count before timing.
One calibration query selects 1-10,000 calls per timed sample, targeting about
250 microseconds per sample. Three warmup samples precede 31 measured samples;
queries whose calibration takes at least 10 milliseconds use 11 measured
samples to bound slow baseline runs. Timings are per-call mean/p50/p95
microseconds, with `ops/sample`, `samples`, and `calls` printed per row.
The accumulated result must equal the expected reclaimable count multiplied
by `1 + ops/sample * (3 + samples)`; that includes calibration and warmup, but
not the separate untimed correctness assertion.
These token histories exercise metadata; they do not execute a long-context
model graph or prove that a deployment model supports those context lengths.

The sparse matrix holds only four blocks per history, with 1/8 histories in
4,096/65,536/262,144-block pools. Unleased and all-leased queries distinguish
endpoint-only inspection from the cost of clearing pool-sized scratch.
The shared-evicted matrix leases 8/32 endpoints sharing 128/512/2,048 missing
physical ancestors. It first evicts those ancestors while the endpoints remain
referenced, then leases the endpoints and reuses every evicted physical block
ID for an unrelated indexed chain. The exact reclaimable-count and reclamation
checks ensure a stale identity never protects the unrelated replacement.

Compare the same benchmark source and Release compiler settings against the
baseline and candidate. Checkpoint protection uses one preallocated CPU byte
per physical pool block, no additional GPU storage, and no query-time
allocation. Unleased queries inspect checkpoint endpoints without scanning
blocks. Leased queries clear that bounded scratch buffer and walk each
retained physical ancestor once, deduplicating shared ancestry. Scratch is
rebuilt for each query because fixed/draft leases can change without a cache
mutation.

Two costs remain: the first leased endpoint clears one byte for every pool
block, even in a sparse cache; missing physical ancestors are not marked and
can be revisited by each leased endpoint. Thus the leased query is bounded by
pool size plus retained ancestry and repeated missing-ancestry walks, not
solely by the number of indexed blocks. The shared-evicted matrix exposes that
limit; this change does not optimize it or replace exact identity validation
with a block-ID-only check.

Component timings isolate the scheduler scan, not end-to-end token latency.
Also run the operator matrix on an available GPU with fixed pool sizing,
long prompts, multiple requests, retained owners, and repeated uncached
controls. Compare host RSS and per-process GPU high-water marks through
alternation/churn; neither the microbenchmark nor a single peak snapshot
establishes a memory plateau or deployment-device fit.

### Outstanding full-model parity qualification

The ORT 1.31 target-only review of the scheduler change compared baseline
`2472a360` with candidate `92b91fab` on the same A100, with CUDA graphs disabled,
1,024 blocks, 512-token chunks, max batch size 32, and 64 generated tokens.
Each full run covered 34 scenarios, 126 reference turns, and six cache-disabled
admission controls. Preserve the reviewer's reported failures:

| Paired run | GEMM autotuning | Baseline scenarios | Candidate scenarios |
|---|---|---|---|
| Original | On | 34/34 passed | 33/34; 513-token second warm replay failed |
| Repeat 1 | On | 33/34; 511-token cold request failed | 33/34; 513-token cold request failed |
| Repeat 2 | On | 34/34 passed | 33/34; 1,537-token second warm replay failed |
| Off comparison | Off | 34/34 passed | 32/34; boundary-1023 and short-first failed; one reference repeat also failed |

The baseline also showed the token-35 change from `32645` to `32471`, on a
different prompt. With autotuning off, the candidate produced different outputs
for the same cache-disabled `churn-5` prompt at zero-based index 34.
Safety, output-length, cache-count, and separate cache-disabled concurrency
checks passed in those reported runs; they do not waive exact output parity.

Four failed candidate full runs versus one failed baseline full run were raised
as an approval blocker during review and remain an unexplained qualification
gap. Baseline/cache-disabled failures show that the scheduler
patch and prefix adoption are not necessary triggers, but do not explain the
observed frequency difference, identify the failing operation, or justify a
"CUDA noise" attribution. Compare raw scores and corresponding fixed/paged
state under identical token history at the first divergence before deciding
whether the issue is independent. Numerical-parity investigation is tracked
in #2701 and the broader qualification gaps in #2698; neither tracking item
counts as passing qualification.

A subsequent paired run with the official ORT 1.31 CUDA 13 release and the
existing Qwen3.8 27B DFlash package passed 34/34 scenarios, all reference
repeats, and all cache-disabled controls on both revisions. Native process
maps confirmed that both loaded ORT 1.31, rather than the Python environment's
older runtime. These target-only runs used the settings above and left the
model package unchanged.

Separate instrumented 513-token boundary runs also passed on both revisions.
At zero-based output index 34, raw logits, every fixed conv/recurrent input
state, and the live INT8 paged KV input contents were bit-identical across
baseline/candidate and cold/512-token-hit replay comparisons. Physical block
IDs were normalized through each request's block table, and unwritten tail
slots were excluded. The unique maximum logit selected token `30` at score
`23.234375`; this was not a reproduced `32645`/`32471` divergence.

These passing runs do not supersede the failures above or explain their
frequency. First-divergence scores/state for a failing run, including fresh
process and autotuning-off comparisons, are still required. Do not describe
the instrumentation or a passing repeat as a runtime fix.

## Testing the runner

```bash
python -m pytest --noconftest test/python/test_prefix_cache_qa.py
```

These model-free tests validate planning, failure detection, and lifecycle
cleanup without the parent model-dependent pytest configuration. They do not
replace real GPU runs or C++ ownership/fault-injection tests. The matrix does
not cover sampled generation, multimodal/pipeline models, or every asynchronous
lifetime.
