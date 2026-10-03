# Prefix-cache validation

Prefix caching needs three separate contracts:

| Contract | Requirement |
|---|---|
| Safety | A hit represents the same token history, KV ownership, and, for hybrid models, fixed state. The final prompt token still executes. |
| Correctness | Cached, uncached, and concurrently scheduled requests produce the same bounded greedy token sequence. An optional drafter snapshot must not change target hit lengths. |
| Progress | Once conflicting owners are released and capacity is available, a completed longer request can publish a deeper reusable boundary. Repeated requests must not remain stranded behind an intermediate duplicate block. |

A large hit is not proof of correct state. Correct output alone is not proof that
prefill was skipped. Low latency alone is not proof of either.

## What the existing validation covers

| Layer | Actual coverage | Remaining gap |
|---|---|---|
| C++ cache and Engine tests | Identity/parent checks, duplicate guards, ownership, checkpoint leases, transaction rollback, allocation failures, eviction, selected branching/concurrency sequences | Mostly individually constructed examples; recording executors do not exercise real target numerical behavior |
| Executable synthetic Python tests | Packed paged/fixed bindings, scheduling, state persistence, event routing | The composite fixture increments fixed state per model call, not per logical token; it is not an independent numerical oracle for arbitrary prefill chunking |
| Windows/Linux builds and lint | Compiler, platform, formatting, and binding compatibility for exercised targets | Do not establish reuse effectiveness or output parity on release GPU graphs |
| PR review | Review of #2657 identified allocation-before-reclamation ordering and retry-safety requirements | Cannot establish hardware execution or cover every history/order combination |
| Real-model integration tests | Opt-in CUDA Engine generation, batching, staggered admission, and model identity checks | No systematic prefix-boundary/order matrix; omitted model-dependent cases are not runtime evidence |
| Interactive `examples/python/engine/model-qa.py` | Conversational Engine execution and event errors | Not a prefix-cache regression test; it does not assert hit lengths, replay progress, or cache-disabled parity |
| Ad hoc GPU runs | Established the short-first plateau, checkpoint-interval branch plateau, and missing-windowed-drafter recovery case | Coverage depended on which request histories happened to be tried |

The highest-ROI addition is a reproducible **operator-run real-model matrix**,
not another interactive prompt or a machine-specific latency threshold.
It complements the allocation/ownership tests rather than replacing them.

## Operator runner

Run `examples/python/engine/prefix-cache-qa.py` on the team's A100/H100 machines
using the intended release package and provider stack. It is not installed as a
CI job. It does not download a model, edit the model package, start a service,
reserve a GPU, or stop other processes. Check that the selected GPU is available
before running it.

The runner uses `Config.overlay()` for temporary profiles and records every
overlay. Omitting `--execution-provider` preserves the package's provider
options, including graph-capture settings. Use a fixed `--num-blocks` budget if
the package uses automatic sizing: free-memory-based sizing is not reproducible
across fresh Engines and is unsuitable on shared GPUs.

The default matrix covers:

| Suite | Sequence and assertion |
|---|---|
| `boundaries` | Prompt lengths immediately before/on/after KV-block and prefill-chunk boundaries; fresh run, exact warm replay, another replay |
| `order` | Ascending and descending lengths sharing the same system corpus; each completed request is immediately replayed |
| `branching` | Single-token divergence before/on/after block and checkpoint boundaries; branch replay, then return to the original history and replay it |
| `alternation` | Warm two distinct histories, then alternate repeatedly; require a deep hit on every switch, not merely on immediate repeats |
| `concurrency` | Simultaneous short/long admission; subsequent sequential repair and warm replay |
| `leases` | Keep a completed original request open while executing a conflicting branch; release the original owner, repair, and require deeper reuse |
| `cancellation` | Cancel after an Engine run during partial prefill, drain the terminal event, then retry and replay |
| `pressure` | Separate bounded-pool profile; distinct first-block histories whose aggregate publishable prefix blocks exceed the configured pool; require observable eviction/checkpoint turnover, then recovery |
| `random` | Seeded divergence positions and token mutations, each followed by replay |

Every selected prompt also runs through a **cache-disabled, greedy reference**
with the same generation policy and configured drafter. By default it executes
each reference twice, keeping the first result as the comparison baseline and
reporting any instability separately. `--reference-repeats` controls this
check; one run leaves reference stability explicitly unchecked. A detected
unstable reference keeps the overall run failed, even if the cached scenarios
match one of its outputs. The runner requires
exact output-token parity, not decoded-text similarity. It requires the selected
generated-token budget to be reached; an early EOS fails the coverage check
rather than silently bypassing decode coverage. It does not set a minimum-token
floor: speculative verification rejects proposals while that floor is active,
so forcing it would unintentionally turn a drafter test into target-only testing.

Complete system/user chat templates provide the base prompts. Exact-length
system-corpus truncation controls the token count. Branch and pressure probes
deliberately mutate valid token IDs; these are cache stress inputs, not an
assessment of answer quality. Small requested lengths that cannot fit the chat
template fail explicitly rather than silently dropping cases.

### Quick boundary/order run

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

Use the Python environment containing the intended GenAI build. A source commit
label does not prove the installed binary was built from that commit; the report
also records the imported module location and available version.

### Full sequence matrix and configuration sweeps

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --chunk-sizes 256 512 1024 --max-batch-size 8 \
  --draft-mode both --seed 20261002 --random-cases 8 \
  --output prefix-cache-full.json
```

`--draft-mode both` runs separate configured-drafter and target-only profiles.
The target-only overlay disables MTP and clears the applicable DFlash2/DSpark
filename. It does not treat a target-only initialization or execution failure
as successful ablation evidence. Each mode has its own uncached reference;
parity between the two drafter modes is not assumed.

Repeat the sequential suites with `--max-batch-size 1` to exercise the smallest
hybrid checkpoint pool. Select suites that do not require multiple resident
requests:

```bash
python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --max-batch-size 1 --suites boundaries order branching cancellation pressure random \
  --output prefix-cache-one-checkpoint.json
```

Also run a paged-only model: that path must retain full-block reuse without
requiring recurrent checkpoints. Changing `--chunk-sizes` changes a temporary
QA profile, so run the unmodified release chunk size separately. Test different
KV-block sizes using model packages exported for those sizes; the runner does
not assume a runtime overlay can change the graph's block geometry.

### Long-context checks

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --lengths 4097 14126 45884 104705 --suites boundaries order \
  --output prefix-cache-long-context.json
```

Use the small default geometry matrix for branching/pressure first. Selecting
many long-context branches causes many independent uncached prefills and can
take substantially longer. A request that exceeds the actual Engine capability
fails with the scenario and capacity; it is never silently shortened.

### Two long histories without cache thrashing

```bash
CUDA_VISIBLE_DEVICES=1 python examples/python/engine/prefix-cache-qa.py \
  --model-path /models/qwen-hybrid-dflash --num-blocks 1024 \
  --lengths 45884 --suites alternation --alternation-split 14079 \
  --alternation-rounds 4 --draft-mode both \
  --output prefix-cache-alternating-46k.json
```

The mutation offset selects where the two equal-length prompts diverge. Warmup
includes immediate replays, followed by strict A/B switches. Compare each
switch's cached-token count and first-reasoning latency with its own immediate
replay; timings alone cannot establish that both histories remained reusable.
Both histories must fit the block budget, and hybrid runs need at least two
fixed-state checkpoint slots. Insufficient requested capacity fails explicitly,
not as an excluded scenario. Keep the one-checkpoint profile on the other
sequential suites.

`--plan-only` prints the selected profile, exact requested lengths, mutation
positions, and actions without importing GenAI or loading a model. Preview
expensive runs with it. `--pressure-num-blocks` controls the separate churn
profile; the runner increases churn count as necessary to exceed that budget.
Regular and pressure model instances are loaded sequentially, not together.

## Reading the results

The JSON report records the configuration/script hashes, operator build/device
labels, runtime module, overlays, actual Engine capacities, prompt hashes, token
IDs, cached-token counts, TTFT, first reasoning-content time when identifiable,
finish reasons, scenario actions, and per-phase speculative statistics. Reports
can contain generated content encoded as token IDs; handle them like other model
QA artifacts. `--prompt-file` supplies an optional local UTF-8 system corpus.

Each run starts as `running` and ends as `passed`, `failed`, or `interrupted`. A failed selected
case produces a nonzero exit and retains the partial report and error. There
are no missing-model skips, numerical-parity waivers, or successful empty-output
fallbacks. An unfinished report is not a pass. Deadline/run-call limits bound
host-side progress checks; they cannot interrupt a GPU call that itself hangs.

Healthy-Engine assertion failures are collected across the matrix by default,
including separate safety, parity, generation-coverage, reuse, and optional
latency results. A parity failure does not hide the cache-reuse result.
The `safety` field checks the public hit boundary (range and block alignment);
it cannot inspect internal ownership or prove fixed-state identity. A `null`
check means that assertion was not applicable, not that it passed.
`--fail-fast` stops at the first assertion. Engine execution/lifecycle failures
and unsafe hit boundaries always stop the run; continuing to drive a possibly
broken Engine is not a valid recovery strategy. Collecting additional findings
never changes a failed overall result into a pass.

For controlled sequential replay, the reuse oracle accounts for the adopted
cursor and subsequent chunk endpoints. Hybrid checkpoints must be block-aligned
and precede the final prompt token; paged-only replay can reuse every preceding
complete block. After branching/concurrency/lease release, the runner requires
at least the newly publishable boundary rather than assuming an older history
must always survive eviction. A graph with a smaller fixed query limit should
be tested with a chunk size within that limit; the configured scheduler budget
must cover the chunk size.

For configured windowed DFlash2, a phase that recomputes a complete window must
resume proposals. Drafter failures/disables are failures even when target-only
fallback emits plausible text. Counters for concurrent requests are reported
per phase, not falsely attributed to individual requests.

A hit without an optional draft snapshot may leave fewer uncached tokens than
the drafter's window. In that case, even successful target reuse cannot restore
drafting during a short decode. Distinguish this expected admission constraint
from an unexpected failure to resume after a complete window. If output parity
also fails, zero proposals alone does not prove target-cache corruption: compare
the target-only profile and execution shapes. The runner still reports the
parity failure; it does not waive it as numerical noise.

Timing is diagnostic by default. `--max-warm-ttft-ratio` adds an explicit ratio
gate against the cache-disabled reference. It is not a statistically robust
latency benchmark: kernel/graph warmup, scheduling, thermals, provider versions,
and other GPU users still matter. First reasoning-content time is direct Engine
token delivery, not Foundry SSE or visible-answer latency.

Before assigning a parity failure to cache ownership, repeat the failing
history in fresh processes, check repeated cache-disabled execution, and run
the target-only control. Differences across those controls narrow the
investigation; they do not turn a failed comparison into a pass. Preserve the
seed, profile, and prompt hashes, minimize the failing history, and add a named
regression before rerunning the broader matrix.

## Architectural conclusions and remaining work

The current transactional ownership model is worth preserving. Removing the
duplicate guard or adopting KV beyond fixed state would trade a visible
performance bug for possible state corruption. Checkpoints and optional draft
snapshots have different budgets and lifetimes; assertions must keep those
contracts separate.

Logical token equality is not numerical equivalence. Bounded hybrid retention
must keep each useful branch's exact physical KV ancestry and corresponding
fixed checkpoint together, rather than splicing recomputed blocks into another
history. Independent deep endpoints should survive while the current history
replaces its own intermediate checkpoints, subject to the existing block and
checkpoint budgets. Unlimited retention of all branches is not its contract;
one-checkpoint or insufficient-memory profiles must still reclaim or decline
publication safely.

This matrix improves confidence, not completeness. It does not inject allocator
or device-copy failures, prove every asynchronous lifetime, cover sampled
generation, or validate multimodal/pipeline models. Keep the C++ fault/lease
tests for those ownership guarantees. Cross-shape numerical differences remain
explicit failures to investigate, not permission to weaken parity checks.

After this operator matrix, the next highest-value layer is a small,
independent token-count-based hybrid fixture plus seeded state-machine tests
over publish/adopt/reclaim/release operations. Its fixed state must depend on
logical tokens rather than number of model calls. That would broaden fast
coverage without reproducing the production cache algorithm inside its oracle.
Any failure discovered on A100/H100 should become a minimal deterministic
regression and a named matrix sequence, with the original seed/profile preserved.

The runner itself has model-free failure-detector/lifecycle tests:

```bash
python -m pytest --noconftest test/python/test_prefix_cache_qa.py
```

These validate the QA tool, not the real GPU cache. They intentionally avoid the
model-dependent parent pytest configuration and do not claim hardware coverage.
