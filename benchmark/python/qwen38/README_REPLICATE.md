# Other-VM Copilot handoff: reproduce optimized Qwen context results

## Objective and boundaries

Reproduce the measured **uncached, batch-one Qwen3.8 Flash NVFP4**
source-code-review/chat context sweep on an otherwise idle H200.
Run the original graphs first. Do not substitute the short-context memory
candidate or treat workload-specific selected-logits validation as universal.
Do not overwrite another user's environment, model, results, or GPU workload.
Keep all samples; never filter for acceptance or throughput targets.

## Original Tianlei v12b package qualification (2026-10-14)

This is a separate experiment from the historical paged-runtime measurements
below and from the rejected locally reconstructed folded graph. The supplied
`/home/asonawane/flash/pkg_v12b_engine` package was tested without modifying it.
A hardlinked clone changes only EOS to valid token 248319 so each request
delivers exactly 512 tokens. This is a forced-length workload, not natural
conversation termination.

Runtime: clean ORT `453ac97a6873b91f8e9dc166ad8945829bbc4119`, locally built
core and plugin CUDA provider (CUDA 13.3); the existing GenAI
`9eef282be7017f2d8e4e1c5677bc00af59296680` binary; supplied custom-op
libraries, including the plane. This is not an all-libraries-from-source
CUDA-13.0 reproduction. Actual loaded library paths and binary hashes are
persisted in the evidence directory. All runs used Engine on exclusive H200
GPU 5, a static single slot, no prefix reuse or continuous batching.

Each dense-prefill cell below aggregates three measured requests following
one retained warmup. E2E includes prefill and generation, not model load.
Decode TPS is 511 divided by first-to-last output-token time.

| Prompt tokens | Continuation decode TPS | Continuation E2E TPS | Chat-review decode TPS | Chat-review E2E TPS |
|---|---:|---:|---:|---:|
| 512 | 234.81 | 219.78 | 232.76 | 216.49 |
| 8192 | 253.02 | 184.97 | 218.06 | 165.74 |
| 16384 | 250.69 | 143.64 | 214.32 | 131.07 |
| 32000 | 248.89 | 99.92 | 214.98 | 94.01 |

**Qualification is workload-dependent, not an unconditional accuracy pass.**
The continuation workload repeats technical documentation; its long-context
draft acceptance is 100%, so do not generalize its approximately 250 TPS to
arbitrary chat. All 16 continuation streams repeat exactly. The supplied CUDA
provider, tested with the same new ORT core, matches all 8K/16K/32K streams
and essentially matches throughput, but its 512-prompt streams do not repeat.

The chat workload preserves the chat-template prefix and final
448-token instruction/assistant suffix around a documentation body. Its
measured acceptance is about 70-77%. **None of its four dense-prefill context
groups is exactly repeatable**, so those chat TPS numbers are diagnostic,
not exact-output-qualified results.

Step-prefill qualification passes all 12 paired 512-output comparisons:
three short questions, an 8192-token continuation, and 512/8192-token
chat-review prompts, each repeated twice in normal MTP and forced rejection.
There are zero MTP failures across all 72 valid requests. The 8K continuation
also matches dense prefill exactly. The chat step-prefill streams repeat and
match forced rejection, but differ from dense prefill. This isolates an
observed fast-prefill-path repeatability problem, not its underlying kernel
cause. Stable 8K chat step prefill measured 218.54 decode TPS but only 14.09
E2E TPS because prefill is sequential; it is not a fast-prefill replacement.
No universal model-accuracy or original-FP16 equivalence claim is made.

### Measured original-package memory (2026-10-14)

An exclusive GPU5 run sampled device memory and process RSS through model
loading, warmup, and 16 fixed-512-output chat-review requests at
512/8192/16384/32000 prompt tokens. The original package was unchanged;
the fixed-EOS clone and clean upstream runtime described above were used.

- Sampled GPU peak: **115.69 GiB**, during loading/warmup.
- Retained GPU allocation after generation, model still loaded: **112.52 GiB**.
- Process peak and retained RSS: **49.53 GiB**.
- Resident PLE mappings: **47.68 GiB** (51.20 decimal GB), verified with
  `mincore`; all mapped PLE pages resident.
- Remaining resident host overhead: approximately **1.85 GiB**.
- GPU memory after process exit: **0 MiB**.

Linux page cache is additional shared/reclaimable storage, not an independent
copy to add to PLE RSS. Global `Cached` increased approximately 114.71 GiB
during the run, but other users and pre-existing cache prevent attributing
that delta exclusively to this process. These results do not mean the full
425 GB package is retained as application RAM. Virtual address reservations
also must not be confused with physical RAM.

The GPU samples averaged 0.328 seconds apart; short allocation peaks may be
missed. Process peak uses Linux `VmHWM`. Model load took approximately
1087 seconds in this run; cache state affects loading substantially.
The profiler is `upstream-qwen38-genai/profile_package_memory.py`.
Run with the flash Python environment; `--summarize` validates and summarizes
existing evidence without loading the model again.
Raw samples, `smaps` category summaries, residency checks, and final summary
are in `upstream-qwen38-genai/qualification/memory/`.

### Original-package kernel profile (2026-10-14)

Nsight Systems 2026.1.3 captured one warmed 8192-token chat-review request
with exactly512 outputs and zero MTP failures. The new upstream provider,
existing pinned GenAI binary and supplied custom ops were unchanged.
Capture starts after loading and a request warmup; CPU NVTX ranges divide
prefill-to-first-token from remaining decode. CUDA graph node tracing and
CUDA allocation tracing are enabled. This is a diagnostic, not a fresh TPS
benchmark: graph tracing perturbs timing, and overlapping kernel durations
cannot be added as wall time.

| Phase | Wall ms under profiler | Kernel calls | GPU kernel union busy fraction |
|---|---:|---:|---:|
| Prefill through first token | 754.98 | 5180 | 98.07% |
| Remaining decode | 2344.24 | 266950 | 94.36% |

Leading decode opportunities, as percentages of **summed kernel durations**:

- Two target NVFP4 MoE GEMV variants: **22.24%**, plus routing
  finalization **3.03%**. Evaluate tile/register tuning and our guarded
  routing/finalize improvements against this exact package.
- Skinny projection family: **26.60%**. Focus on the expensive two-row
  shapes, direct split outputs and row-shared weight reads.
- Leading merged HC kernel: **17.25%** (~16.16us/call).
  Controlled ablations of existing PDL/early-weight/row-sharing settings
  precede kernel changes. Preserve software-grid-barrier co-residency.
- Sparse indexer compact/merge/score top-k kernels: **5.35%** combined;
  grouped sparse attention and its merge: **5.93%**. Preserve index state
  precision, selection and tie order.

Prefill hotspots: chunked GDN **15.13%**, NVFP4 weight dequantization
**14.32%** (96 calls,106.01ms), DSA prefill **13.32%**. Packed-weight
dequant+GEMM fusion or tiled scratch is a concrete prefill/memory candidate,
but the existing BF16 graph must be qualified separately from earlier FP16
paged-kernel work.

Several prefill CPU copy calls take ~57ms to read **four bytes**; their GPU
copy takes ~2.4us. This is a wait for preceding device work, not a PCIe
bandwidth problem. Defer or batch validation only after locating exact call
sites and preserving explicit error reporting. Aggregate CPU copy API time
(675ms) is not independently removable wall time.

Decode has only two stream synchronizations (0.61ms aggregate), so further
host synchronization work is not the first priority. PLE gather kernels
account for ~0.22% of summed decode kernel time on this warm workload.
Neither observation establishes hardware bandwidth/occupancy bottlenecks;
Nsight Compute counters were not collected.

Direct `cudaMemGetInfo` reads show ~112.54GiB used at model-ready,
after request warmup, and after profiling. This capture excludes loading
allocations, so it does not establish which part of the gap above the
72.17GiB shared pool is recoverable. A separate load-time allocation trace
or per-session instrumentation is required for that attribution.

Profile MTP counters:289rounds,223accepted drafts,77.16% acceptance.
The profiled and warmup streams differ, consistent with the documented
fast-prefill repeatability limitation; zero MTP failures is not an exact
accuracy pass.

Scripts: `upstream-qwen38-genai/profile_original_kernels.py` and
`summarize_kernel_profile.py`. Persistent local evidence (placed on the
larger `/var/tmp` filesystem):
`/var/tmp/asonawane-v12b-profile/{engine.nsys-rep,engine.sqlite,run.json,kernel-summary.json}`.
The summarizer validates output length/failures and separates phase kernel,
CUDA API and transfer costs. Candidate rankings are hypotheses for controlled
experiments, not measured optimization speedups.

### Ordered v12b optimization screening

Follow-up experiments used the same original package/runtime, an exclusive
GPU5, four dense-prefill requests (one warmup plus three measured), and
four step-prefill8192 requests to compare decode on exactly identical
512-token streams and speculative counters. Short step-prefill questions
were also checked. Dense prefill remains non-repeatable: its acceptance
changes between runs, so dense TPS differences are not isolated kernel gains.

| Variant | Exact step8K decode TPS | Retained VRAM GiB | Dense8K TTFT ms |
|---|---:|---:|---:|
| Fresh control | 234.64 | 112.52 | 745.57 |
| MoE FC1 PDL enabled | 236.05 | 112.52 | 745.40 |
| Skinny early trigger | 234.58 | 112.52 | 744.47 |
| HC early weights | 234.74 | 112.52 | 746.00 |
| HC one row per CTA | 234.56 | 112.52 | 745.00 |
| Control repeat | 234.59 | 112.52 | 745.35 |
| Prefill4096 + matching warmup | 242.24 | 100.40 | 891.13 |
| Prefill2048 + matching warmup | 234.50 | 98.85 | 1244.02 |
| Prefill4096 repeat | 234.41 | 100.40 | 891.86 |

All listed step8K output streams and counters match the fresh control exactly;
short gates also pass, with zero MTP failures. MoE FC1 PDL's ~0.6% is a small
screening result, not a broader qualified speedup. Skinny/HC switches do not
justify changed defaults. Sparse top-k has no safe runtime knob; a kernel
rewrite remains unimplemented rather than claimed tested.

The4096 option consistently saves **12.11GiB** retained memory (~10.8%) for
the measured8K workload at the cost of ~20% higher TTFT. Its apparent first
decode gain did not repeat. The2048 option saves13.67GiB but raises TTFT ~67%.
Neither option is exact-output-qualified for dense prefill or validated over
all contexts; these are intentional opt-in memory/latency tradeoffs. An overlay
is persisted at `upstream-qwen38-genai/qualification/prefill-4096-overlay.json`.
Apply through `og.Config.overlay`, keeping original package unchanged.

Loading Nsight allocation trace reconstructs **110.66GiB live device
allocations**, with **113.84GiB tracked allocation peak**; direct model-ready
CUDA use is112.54GiB. The first625 live device allocations match the shared
pool manifest in order and byte size exactly (**72.17GiB**), leaving
**38.49GiB** of other tracked live allocations. Largest non-pool blocks:
8GiB x1,4GiB x2,2GiB x2,1GiB x3. No allocation stacks/session labels were
collected, so arena versus packed-weight attribution is not established.
Driver/context accounting explains some difference from direct CUDA readings;
this trace does not prove38.49GiB is removable.

No production defaults, custom-op binaries, provider binaries or original
model files were changed. Actual sparse-kernel rewrites, fused BF16 prefill
dequant/GEMM, and per-session allocation-stack attribution remain follow-up
implementation work, not completed optimizations.

Evidence: `/var/tmp/asonawane-v12b-experiments/evaluation.json`, individual
runJSONs/logs, and `load.nsys-rep`/`load.sqlite`. Scripts in the isolated GenAI
worktree: `run_v12b_experiment.py`, `profile_v12b_load.py`,
`summarize_v12b_experiments.py`. Rerun the summarizer to check streams,
counters and allocation accounting without another GPU run.

Local persistent evidence:

Safetensors cleanup (2026-10-14): after recursively auditing tensor external
data in all four ONNX models (including nested messages), shared-pool ranges,
PLE shard ranges and JSON metadata, two unreferenced files were removed from
the supplied package:
`head_r2/source_weights/model-bf16-00010.safetensors` and
`target/head_data/source_weights/model-bf16-00010.safetensors`.
They total647128402bytes; both have SHA256
`0c2c3ef0030f5134edca0f3bbffc111cfa5da4b39b839cb8248ec762009be0aa`.
Every referenced file and checked range still resolves after deletion.
Package logical size decreases from425.387GB to424.740GB. No physical space
is expected to be freed because the test clone retains hardlinks.
Graph/config/library contents were not changed. This is dependency validation,
not a new GPU inference run. Audit script:
`upstream-qwen38-genai/audit_package_safetensors.py`; before/after evidence:
`/var/tmp/asonawane-v12b-experiments/safetensors-cleanup.json`.

- `upstream-qwen38-genai/qualification/evaluation.json`: final gates,
  throughput, provenance, and limitations.
- `upstream-qwen38-genai/qualification/{gate*,benchmark*}.json`: all token
  streams, prompt hashes, timestamps, and speculative counters.
- `upstream-qwen38-genai/qualification/*loaded-libraries.json`: actual loaded
  core, CUDA provider, GenAI, and custom-op paths.
- `upstream-qwen38-genai/qualify_original_package.py` and
  `summarize_original_qualification.py`: executable measurement and assertion
  scripts. Initial invalid NumPy-array token-multiplication results are
  explicitly archived as `invalid-input-*` and excluded.

To rerun in this workspace (one command at a time on an otherwise idle GPU):

```bash
cd /home/asonawane/flash/upstream-qwen38-genai
PY=/home/asonawane/.conda/envs/flash/bin/python
"$PY" qualify_original_package.py --prepare
export CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/qualification/runtime"
export LD_LIBRARY_PATH="/home/asonawane/flash/upstream-qwen38-ort/build-upstream:/usr/local/cuda-13.3/lib64:/usr/lib/python3.12/site-packages/nvidia/cudnn/lib"
"$PY" qualify_original_package.py --gate
"$PY" qualify_original_package.py --gate --force-reject
"$PY" qualify_original_package.py --gate-long
"$PY" qualify_original_package.py --gate-long --force-reject
"$PY" qualify_original_package.py
"$PY" qualify_original_package.py --shipped-provider
"$PY" qualify_original_package.py --chat-workload
"$PY" qualify_original_package.py --gate --chat-workload
"$PY" qualify_original_package.py --gate --chat-workload --force-reject
"$PY" summarize_original_qualification.py
```

All published changes use **`asonawane/qwen-38-flash-100-e2e` in both repositories**.

| Repository | Pinned source commit | Purpose |
|---|---|---|
| microsoft/onnxruntime | `16289abceb0eb1d2330bca6e8aa1de708d227f86` | Optimized CUDA provider and memory kernels |
| microsoft/onnxruntime-genai | `2a69912474e29275ad2cccb01bcd96593ea29c7f` | Matching runtime, graph preparation, portable benchmark scripts |

The measured GenAI native runtime was built at
`37e2b4773655427afa493098c7f44cb593cff0fa`. The pinned later GenAI commit
adds Python scripts/documentation without changing that native runtime.
Selected-logits preparation was added in separate commit `25ac6ecb`.
These pins reproduce the historical table. The experimental prefix-cache and
warp-heads follow-ups below require newer source and freshly rebuilt libraries.
Do **not** mix these with newer `kvaishnavi/qwen-38-flash` HEADs or stock wheels.
The API32 compatibility patch used for the kvaishnavi comparison is not
needed for this pinned optimized runtime.

## Prerequisites

- Linux x86-64, Python 3.12, a dedicated environment.
- H200 / SM90 with enough free VRAM: this long-context sweep sampled about
  92-98 GiB, so it is **not an 80 GiB workload**.
- Original model export with decoder NVFP4 weights, FP8 MTP, Engram,
  tokenizer, chat template, and all external weights.
  Model files are not in Git and this procedure does not download them.
- CUDA/cuDNN/compiler/CMake/Ninja and each repository's build dependencies.
  Reference machine: CUDA 13.3.73, driver 580.105.08, GCC 13.2.
  A different toolchain may change results or require compatibility work.
- Enough disk for source/builds and the model. Selected-logits preparation
  additionally copies the external weights and hashes them; allow substantial
  disk space and time.

Use real absolute paths instead of the placeholders below.

## 1. Get the pinned sources

Use new clone directories, not an unrelated dirty checkout:

```bash
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime.git onnxruntime-qwen
git clone --recursive --branch asonawane/qwen-38-flash-100-e2e \
  https://github.com/microsoft/onnxruntime-genai.git onnxruntime-genai-qwen

export ORT_SRC="$PWD/onnxruntime-qwen"
export GENAI_SRC="$PWD/onnxruntime-genai-qwen"
git -C "$ORT_SRC" switch --detach 16289abceb0eb1d2330bca6e8aa1de708d227f86
git -C "$GENAI_SRC" switch --detach 2a69912474e29275ad2cccb01bcd96593ea29c7f
git -C "$ORT_SRC" submodule update --init --recursive
git -C "$GENAI_SRC" submodule update --init --recursive
```

## 2. Build and install matching wheels

Activate/create a dedicated Python 3.12 environment first. Follow repository
dependency instructions; install missing dependencies into this environment
only. Do not install a stock ORT GPU wheel to resolve a local-runtime problem.

```bash
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/absolute/path/to/cudnn
unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd "$ORT_SRC"
./build.sh --config Release --update --build --parallel 8 \
  --build_shared_lib --build_wheel --use_cuda \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --skip_tests --cmake_generator Ninja \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
    onnxruntime_USE_FP4_QMOE=ON onnxruntime_USE_FP8_QMOE=ON
python -m pip install --no-deps build/Linux/Release/dist/*.whl

export ORT_HOME="$ORT_SRC/ort-sdk-qwen38"
mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$ORT_HOME/include/"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.inc "$ORT_HOME/include/"
cp -a "$ORT_SRC"/build/Linux/Release/libonnxruntime*.so* "$ORT_HOME/lib/"

cd "$GENAI_SRC"
python build.py --config Release --update --build --parallel \
  --use_cuda --cuda_home "$CUDA_HOME" --ort_home "$ORT_HOME" \
  --skip_tests --cmake_generator Ninja \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90
find build/Linux/Release -name '*.whl'
```

Install the GenAI CUDA wheel printed by `find`:

```bash
python -m pip install --no-deps /actual/path/to/onnxruntime_genai_cuda.whl
export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import sys, onnxruntime as ort, onnxruntime_genai as og; print(sys.executable); print(ort.__version__, ort.__file__); print(og.__file__)'
```

Keep the SDK directory in place. If cuDNN uses `lib64` or another layout,
adjust the library path accordingly. Configure failures must be resolved
explicitly, not by mixing SDK headers/providers from another branch.
`--skip_tests` is not a statement that runtime tests passed.

## 3. Confirm the original model and idle GPU

```bash
export MODEL=/absolute/path/to/qwen_38_flash_nvfp4_engine
sha256sum "$MODEL/genai_config.json"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

Reference original configuration SHA256:
`bc7316907c666870c2065729f746d498306c69854f54d2ffcd367ee700b170fa`.
Reference original external weight SHA256:

| File | SHA256 |
|---|---|
| model.onnx.data | `493cba9faace381022c749fc210a87c9d5553795ffed303f74aa7dbd13376928` |
| mtp.onnx.data | `11f178b98e15cd889a9fbc5e570e76b0f71637d27e13a63cee97a1363af65f47` |
| engram.onnx.data | `b910f6c322efea60a5d641c42d5c518e8a57c19b336b78d9606c921320a0b993` |

A changed export/config is not an exact reproduction. Hashing these large
weight files takes time. Never kill another user's process to free a GPU.

## 4. Run the measured long-context workload

```bash
python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model "$MODEL" \
  --output-dir /absolute/path/to/new-original-context-results \
  --gpu 0
```

`--gpu` is the **physical nvidia-smi index**. The output directory must not
exist and must be outside the model directory.

The runner sets these variables before model loading:

```bash
ORT_QMOE_NVFP4_PACKED_PREFILL=1
ORT_QMOE_NVFP4_FUSED_SWIGLU=1
ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH=1
ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT=1
ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1
ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS=0
```

No manual flag exports are needed for this runner. It preserves explicit
`PYTHONPATH`/`LD_LIBRARY_PATH` for deliberate runtime discovery, but does not
forward unrelated inherited ORT debug/calibration overrides.

Fixed protocol:

- Contexts 8192, 16384, 32768, 65536, 131072; MTP widths 1, 3, 5, 7.
- Source-code-review **chat** prompt built from repeated Python worker modules.
  It is not the older plain distributed-inference-service prompt.
- Batch one, greedy, 512 output tokens.
- Fresh process per configuration, fresh Engine/request per run.
- One excluded warmup plus three measured runs; no reused prompt cache.
- CUDA graphs and decoder/MTP GEMM tuning enabled.
- Scheduler 8192; KV-cache budget 0.05 for 8K-64K and 0.12 for 128K.
- Nominal 20 ms device-wide VRAM sampling, including load/warmup and timing.
  This is sampled peak, not a hard cap; occupancy is checked before each row,
  not continuously.
- Mean TTFT; pooled decode TPS = total 511-token counts / total decode time;
  pooled E2E TPS = total 512-token counts / total request time.
- Acceptance = accepted/evaluated; first-draft acceptance =
  (partial + full acceptance rounds)/completed rounds; proposal yield =
  accepted/proposed. Target 75-85% is descriptive, not a filter.

Outputs: `TABLE.md`, `manifest.json`, per-row JSON, logs, overlays and memory
samples. Failures remain explicit and produce a nonzero exit.

## 5. Expected reference results

MTP width 3 produced the highest measured E2E TPS at each context:

| Requested context | Actual prompt | TTFT (ms) | Decode TPS | E2E TPS | Sampled peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8190 | 1226.31 | 129.52 | 99.00 | 94340 |
| 16384 | 16384 | 2517.40 | 134.54 | 81.07 | 94336 |
| 32768 | 32768 | 5329.79 | 132.43 | 55.72 | 94342 |
| 65536 | 65536 | 11748.26 | 131.11 | 32.72 | 94332 |
| 131072 | 131072 | 27779.95 | 121.41 | 16.01 | 100490 |

E2E counts generated tokens only and includes prompt processing. Long
contexts therefore have lower E2E TPS even when decoding stays fast.
These are measured aggregates, not strict thresholds or performance guarantees.
Fresh hardware/builds may differ. Matching prompts do not establish numerical
or answer-quality equivalence; output hashes differed from the kvaishnavi baseline.

## 6. Selected-logits experiment: separate from reproduction

This newly published option projects only requested hidden-state rows through
the LM head. It preserves the original 262144 configured context, indexer
capacity and other settings. Preparation/tests passed. A subsequent fresh
64K/128K MTP-width-three comparison measured the following results, with one
excluded warmup and three measured 512-token requests per configuration:

| Context | Original TTFT (s) | Selected-logits TTFT (s) | Original E2E TPS | Selected-logits E2E TPS | Sampled VRAM saved |
|---:|---:|---:|---:|---:|---:|
| 65536 | 11.747 | 11.630 | 32.42 | 32.68 | 4 GiB |
| 131072 | 27.733 | 27.510 | 16.13 | 16.24 | 4 GiB |

All six measured candidate token streams exactly matched their paired
controls. This single control-then-candidate comparison shows a useful
memory reduction and a small observed latency improvement, not a robust
speedup guarantee or broad answer-quality qualification. Other context/MTP
configurations remain unverified for the transformed graphs.

A separate CUDA-profiler capture of 128K candidate prefill attributed 27.3%
of summed GPU kernel time to sparse indexer scores, 18.5% to sparse paged
attention, and 26.4% to packed NVFP4 expert GEMMs. These are kernel-time
shares, not wall-clock speedup predictions. Profiling timings were excluded
from the paired throughput results.

```bash
python "$GENAI_SRC/benchmark/python/qwen38/prepare_memory_candidate.py" \
  --source "$MODEL" \
  --destination /absolute/path/to/new-long-context-selected-logits-model \
  --preserve-context

python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model /absolute/path/to/new-long-context-selected-logits-model \
  --output-dir /absolute/path/to/new-selected-logits-context-results \
  --gpu 0
```

Preparation requires `onnx`, copies external weights independently, checks
SHA256, and refuses an existing destination. Use a separate model/output path.
Compare prompt hashes, overlays, full `token_ids` streams, TTFT/decode/E2E,
acceptance and memory against fresh original-graph controls. Report mismatches;
do not claim a speedup or quality preservation from preparation tests alone.
An existing matching runtime does not need a kernel rebuild for this option.

Do not omit `--preserve-context`: default preparation creates the bounded
8K memory candidate, configured at 10240 with a previously measured 9217-total-
token Engine ceiling. That package cannot reproduce the long-context table.
The historical `run_benchmark.sh` and `benchmark_qwen_mtp.py` are 8K/512 tools,
not the context-sweep entry point.

## 7. Experimental prefix-cache and warp-heads follow-ups

### Source selection for the follow-up results

Both repositories publish these follow-ups on
`asonawane/qwen-38-flash-100-e2e`. Unlike historical-table reproduction in
section 1, select the following source before rebuilding both wheels/SDK
using section 2:

```bash
git -C "$ORT_SRC" fetch origin asonawane/qwen-38-flash-100-e2e
git -C "$GENAI_SRC" fetch origin asonawane/qwen-38-flash-100-e2e
git -C "$ORT_SRC" switch --detach bfd45538f44fe0e72dbb08350ae98d84bed67f53
git -C "$GENAI_SRC" switch --detach origin/asonawane/qwen-38-flash-100-e2e
git -C "$ORT_SRC" submodule update --init --recursive
git -C "$GENAI_SRC" submodule update --init --recursive
git -C "$ORT_SRC" rev-parse HEAD
git -C "$GENAI_SRC" rev-parse HEAD
```

Record the resolved GenAI commit: the remote branch may move after publication.
Its source must include the explicit batch-one MTP prefix guard, Python
`get_prefix_cache_stats()` API, and `--prefix-cache` sweep option.
The historical GenAI pin does not include these changes.

### Experimental MTP prefix reuse

Explicit `engine.dynamic_batching.prefix_caching=true` now admits batch-one
MTP target-prefix reuse when no DFlash drafter or sliding-window KV layout
is present. Implicit/default MTP prefix behavior remains unchanged.
Target paged KV and fixed-state checkpoints are adopted; each new request
starts a cold MTP shadow at its first generated token, as in the uncached path.
This is not restored draft-prefix state and is not qualified for concurrent
batching or all models.

With the rebuilt GenAI library/Python bindings and warp-heads provider:

```bash
python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model /absolute/path/to/long-context-selected-logits-model \
  --output-dir /absolute/path/to/new-cached-results \
  --gpu 0 --indexer-warp-heads --prefix-cache --output-tokens 512
```

Unlike the historical uncached benchmark, one Engine is shared across the
cold warmup and repeated identical requests. The cold request is excluded
from cached timing; cache-fill cost is not free. `get_prefix_cache_stats()`
records per-request adoption counts and the benchmark requires exact output
parity with its cold request. Speculative counters are differenced per request,
not pooled repeatedly from cumulative Engine telemetry.

The completed 20-row local experiment matched all 60 measured outputs to
their cold requests with zero MTP failures. Adopted tokens per request were
0/8192/24576/57344/122880 for requested 8K/16K/32K/64K/128K prompts.
The retained checkpoint leaves the final scheduler chunk to recompute, so
8K has no cache hit. Full repeated-request results follow. The actual 8K
prompt has 8190 tokens; all other prompts have exactly the requested length.
VRAM is device-wide sampled peak including load and warmup, not a hard peak
guarantee. Acceptance target is 75-85%; no samples were filtered.

| Requested prompt | Prompt length | MTP | TTFT (ms) | Decode TPS | E2E TPS | MTP acceptance | First-draft acceptance | Proposal yield | Within target | Sampled peak VRAM (MiB) | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|
| 8192 | 8190 | 1 | 1195.55 | 118.96 | 93.24 | 80.21% | 80.21% | 80.21% | yes | 91262 | completed |
| 8192 | 8190 | 3 | 1197.86 | 135.17 | 102.85 | 71.27% | 75.00% | 53.75% | no | 91268 | completed |
| 8192 | 8190 | 5 | 1198.44 | 121.15 | 94.53 | 68.30% | 74.43% | 38.17% | no | 91424 | completed |
| 8192 | 8190 | 7 | 1200.24 | 104.22 | 83.89 | 66.73% | 70.52% | 27.92% | no | 91422 | completed |
| 16384 | 16384 | 1 | 1247.50 | 114.97 | 89.95 | 76.47% | 76.47% | 76.47% | yes | 91262 | completed |
| 16384 | 16384 | 3 | 1249.64 | 139.70 | 104.33 | 74.31% | 75.94% | 57.86% | no | 91274 | completed |
| 16384 | 16384 | 5 | 1251.89 | 130.82 | 99.26 | 72.46% | 74.53% | 43.70% | no | 91416 | completed |
| 16384 | 16384 | 7 | 1252.68 | 112.31 | 88.23 | 70.52% | 75.16% | 32.51% | no | 91458 | completed |
| 32768 | 32768 | 1 | 1320.96 | 116.34 | 89.62 | 80.85% | 80.85% | 80.85% | yes | 91262 | completed |
| 32768 | 32768 | 3 | 1321.83 | 138.61 | 102.23 | 74.31% | 77.01% | 57.86% | no | 91276 | completed |
| 32768 | 32768 | 5 | 1323.40 | 123.01 | 93.47 | 70.02% | 75.88% | 40.21% | no | 91416 | completed |
| 32768 | 32768 | 7 | 1319.85 | 109.43 | 85.48 | 69.37% | 80.00% | 31.54% | no | 91440 | completed |
| 65536 | 65536 | 1 | 1440.96 | 115.72 | 87.42 | 79.93% | 79.93% | 79.93% | yes | 91246 | completed |
| 65536 | 65536 | 3 | 1444.15 | 133.57 | 97.15 | 72.73% | 77.89% | 56.34% | no | 91292 | completed |
| 65536 | 65536 | 5 | 1444.73 | 118.66 | 89.02 | 69.69% | 76.16% | 39.39% | no | 91432 | completed |
| 65536 | 65536 | 7 | 1444.16 | 100.21 | 78.25 | 67.46% | 74.71% | 28.81% | no | 91474 | completed |
| 131072 | 131072 | 1 | 1670.61 | 111.62 | 81.94 | 78.32% | 78.32% | 78.32% | yes | 96380 | completed |
| 131072 | 131072 | 3 | 1668.68 | 133.28 | 93.04 | 75.41% | 76.34% | 58.45% | yes | 96398 | completed |
| 131072 | 131072 | 5 | 1670.77 | 122.48 | 87.63 | 71.60% | 77.91% | 42.70% | no | 96518 | completed |
| 131072 | 131072 | 7 | 1673.45 | 105.49 | 78.56 | 70.44% | 78.85% | 32.87% | no | 96596 | completed |

MTP-three comparison against the same selected-logits/warp-heads variant
with fresh uncached Engines:

| Requested context | Uncached TTFT (ms) | Cached TTFT (ms) | Uncached E2E TPS | Cached E2E TPS | E2E gain |
|---:|---:|---:|---:|---:|:---|
| 8192 | 1201.22 | 1197.86 | 104.06 | 102.85 | No cache hit |
| 16384 | 2422.92 | 1249.64 | 84.95 | 104.33 | 1.23x |
| 32768 | 5009.35 | 1321.83 | 57.70 | 102.23 | 1.77x |
| 65536 | 10534.68 | 1444.15 | 35.44 | 97.15 | 2.74x |
| 131072 | 22932.74 | 1668.68 | 18.87 | 93.04 | 4.93x |

These are cached repeated-prefix results, not uncached prefill speedups.
Changing the prompt, eviction pressure, chunk size, or concurrency requires
fresh validation.

### Experimental warp-local indexer scoring

ORT commit `bfd45538f44fe0e72dbb08350ae98d84bed67f53` adds default-off
`ORT_PACKED_SPARSE_INDEXER_WARP_HEADS=1`, retaining query heads within a warp
to reuse keys and remove scoring barriers. Rebuild the matching CUDA provider
before testing it. The sweep runner accepts `--indexer-warp-heads` explicitly;
omitting it keeps the historical reproduction unchanged.

Two fresh flag-off/on comparisons, in opposite execution orders, used the
same selected-logits model and MTP width three:

| Context | Control TTFT (s), two invocations | Candidate TTFT (s), two invocations | Control E2E TPS | Candidate E2E TPS |
|---:|---:|---:|---:|---:|
| 65536 | 11.615 / 11.620 | 10.515 / 10.494 | 33.13 / 32.95 | 35.25 / 35.36 |
| 131072 | 27.534 / 27.511 | 22.904 / 22.895 | 16.18 / 16.35 | 18.93 / 18.88 |

Standalone FP16/FP32 tests match all outputs exactly across ten geometries
per type, including 32768 live slots. The first full-model pair matched six
measured streams. The reverse-order 64K pair also matched; the reverse-order
128K control differed from both candidates and the earlier control.
This observed control variability prevents a claim of full-model deterministic
equivalence or broad quality preservation. No samples were discarded.
The option remains experimental, not a default change or 100-E2E-TPS solution.

A further controlled 128K pair set `ep.cuda.enable_gemm_auto_tune` to `"0"`
in both decoder and MTP session options. All three measured token streams
matched exactly. TTFT was 27.517 seconds flag-off versus 22.935 flag-on;
decode was 124.16 versus 124.19 TPS and E2E was 16.19 versus 18.93 TPS.
This supports parity and a prefill-specific gain on that workload, without
proving that autotuning caused the earlier mismatch. Keep this diagnostic
separate from the historical tuned table.

### Variable output length on current source

The current context benchmark and sweep default to `--output-tokens 512`
for controlled optimization comparisons. Explicit `--output-tokens 0` generates
until EOS or the smaller of model context and Engine request capacity,
without a fixed 512-token cap or EOS suppression. A 1800-second
per-request safety timeout still fails explicitly rather than truncating output.
Use `--output-tokens 512` to reproduce the historical tables, or
`--output-tokens 2048` for an exact longer decode comparison. The older pinned
scripts in section 1 have a hard-coded 512 length and do not accept this option;
use current branch source for variable-length benchmarking.

```bash
python "$GENAI_SRC/benchmark/python/qwen38/run_context_sweep.py" \
  --model /absolute/path/to/long-context-selected-logits-model \
  --output-dir /absolute/path/to/new-natural-output-results \
  --gpu 0 --indexer-warp-heads --prefix-cache --output-tokens 0
```

Actual output counts and per-request capacity limits are saved in raw results;
the manifest also records measured output counts. Decode TPS uses actual
generated tokens minus the first token per request divided by decode time.
E2E TPS uses all actual generated tokens divided by total request time.
A request with no measurable decode interval reports decode TPS as null/N/A.
Longer outputs change context during decode and amortize prefill in E2E TPS;
they are not a kernel speedup. EOS-ended text and fixed-length chat/code
workloads must not be compared as if they were identical. The quoted external
2K/8K/32K text TPS is not reproduced or qualified by this change.

## 8. Copilot validation checklist

### Engine-native sparse attention follow-up (local)

The current sweep accepts `--sparse-mtp-grouped-heads` and
`--sparse-prefill-async-kv`, both default off. Rebuild the matching ORT provider
before enabling either. These retain the Engine API and do not require the
Generator-only `qwen38_flash_graph` package or its external custom-op libraries.
The first uses same-reduction grouped scalar attention for small MTP batches;
the second stages prefill KV asynchronously. Exact output qualification used
GEMM autotuning disabled in both decoder and MTP sessions. Autotuning-enabled
controls themselves varied; the cause of that variability is not established.

For a qualified 8190-token prompt, MTP one, exact512 output, CUDA graphs, and
4096-token scheduler chunks with genuine prefix reuse (4096 adopted tokens),
paired Engine measurements were:

| Metric | Control | Both sparse options |
|---|---:|---:|
| Decode TPS | 112.68 | 120.97 |
| E2E TPS | 98.46 | 105.50 |
| TTFT (ms) | 665.35 | 628.90 |

All twelve cold/measured streams matched exactly. With 8192 scheduler chunks,
the 8K prompt has no reusable checkpoint; that separate pair measured E2E
95.30 to 102.70 TPS. Do not compare these diagnostic autotuning-off protocols
directly with the historical tuned table or assume long-context qualification.
Original defaults remain unchanged; the output default remains 512.

### Combined Engine-native kernel qualification (local)

The sweep additionally accepts `--hyperconnection-vectorize`,
`--nvfp4-fused-routing`, and `--nvfp4-single-pass-finalize`. Each defaults off,
is explicitly written into the child environment and manifest, and requires
a matching rebuilt ORT provider. No Generator-only graph runtime is used.
The MoE port includes only fused routing and ordered single-pass finalization;
experimental register-prefetch, PDL changes and the rejected prefill scale
pipeline are excluded. Hyperconnection vectorization is FP16 prefill-only,
with fallback guards; it does not fuse or alter projection arithmetic.

A combined rebuilt provider was tested with all five new options off/on/off
in separate processes. Engine, MTP one, actual8190 prompt, exact512 outputs,
CUDA graphs, GEMM autotuning off, scheduler4096 and true prefix reuse were
identical on both sides:

| Invocation | Decode TPS | E2E TPS | TTFT (ms) |
|---|---:|---:|---:|
| Control | 111.32 | 97.43 | 664.54 |
| All five options | 124.22 | 108.49 | 605.70 |
| Control repeat | 111.77 | 97.81 | 662.77 |

All twelve cold/measured output streams matched exactly, with zero MTP
failures; each measured request adopted4096 prefix tokens. This is an 8K
MTP-one qualification, not a new all-context table or a200TPS claim.
Flags remain opt-in pending broader context/MTP qualification. The current
sweep's historical8192 scheduler/tuning settings differ from this diagnostic
protocol; add the options to investigate it, not to assume identical numbers.

### Follow-up single-draft device path screening (not retained)

A short CUDA-profiler capture of the optimized Engine MTP-one decode path
showed176 stream synchronizations and47 graph launches. Synchronization
accounted for74.5% of summed CUDA API duration, but includes waiting for GPU
execution and must not be interpreted as removable CPU overhead. Device
argmax and compact token transfers already exist for target verification.

A candidate extended the existing CUDA device-draft-chain path from multiple
drafts to a single draft, allowing the asynchronous decoder/device-argmax
handoff and reusable draft-token storage in MTP-one as well. It built and
passed63 native MTP/speculative tests. Cached Engine control/candidate/control
with the combined kernels, autotuning off, scheduler4096, and exact512 outputs
produced decode123.98/124.28/124.10TPS and E2E108.23/108.49/108.34TPS.
All twelve streams matched, and each measured request reused4096 prefix tokens.
The approximately0.2% observed difference is within run variability; the
candidate was reverted rather than retained as an optimization. Profiler
timings are excluded from throughput claims.

### Remaining Engine experiments (screened, not integrated)

All candidates below were tested with Engine and exact512 outputs. Isolated
candidate sources/runtimes were not copied into the qualified runtime.
Compare only paired runs within each experiment; the differing baseline TPS
across experiments are not interchangeable.

| Experiment | Measured result | Decision |
|---|---|---|
| Binding-name validation cache | Microbenchmark20.9 to3.0us; Engine ABBA decode131.37 to131.92TPS and E2E113.77 to113.96TPS | No demonstrated gain beyond variation; reverted |
| Fused verified-token CUDA handoff | Decode131.72 to128.14TPS;128 fused launches confirmed | Slower; not integrated |
| Explicit MTP-head CUDA graph option | Decode123.78TPS, E2E108.08TPS versus prior bracketed decode123.98/124.10TPS | No measured benefit; not integrated |
| Terminal prefix checkpoint | Reused7936/8190 tokens; cached TTFT control/candidate/control613/140/611ms | Changes output tokens; rejected |
| Scheduler2048 fallback | Reused6144 tokens; cached TTFT360ms | Changes output tokens; not an equivalent optimization |

Host-cache qualification matched all20 output streams and passed913 native
tests, including eight new tests. The fused handoff matched all36 streams
with zero MTP failures;888 CPU Engine tests,17 CUDA Engine tests per toggle,
and six handoff tests per toggle passed. Explicit MTP-head graph screening
matched all four streams.

Terminal checkpoint qualification passed all twelve512-token cached versus
cold-shadow cases, including changed suffixes and cache pressure with31
prefix-block evictions. However, changing the cold prefill segmentation
changed the same-budget control output: first difference at generated token46
with MTP and495 target-only. Candidate-off matched control exactly.
The2048 fallback also diverged, at token53. Cache self-consistency is not
enough to establish equivalence to the qualified baseline. Final targeted
native tests passed183 cases with one skip; neither checkpoint alternative
is retained, despite the lower TTFT.

A full device-only MTP cycle remains a larger redesign: host acceptance
currently determines prefix commits, MTP packing, shadow transactions and
binding lifetimes. No synchronization or correctness guards were removed
to manufacture a speedup. The five qualified opt-in kernels above remain
the retained improvements; these follow-ups establish no additional
throughput gain and do not establish200TPS.

### GenAI v12b branch evaluation (isolated)

Branch `tlwu/20261013/ga03-engine-v12b-main`, inspected/built at
`9eef282be7017f2d8e4e1c5677bc00af59296680`, now hosts its specialized folded
MTP-one backend through Engine, not Generator. It is not a replacement for
the dynamic prefix-cached backend: it has no prefix caching, serves one
resident request, rejects `min_generated_tokens`, and needs externally
exported graphs/custom ops and a plugin CUDA provider. Its fast dense
prefill is documented as tolerance-level, not run-to-run reproducible.

The generic hybrid-prefix retention change
`2472a3608fccf7e86794d31e63903e8e25bcedde` was applied to an isolated checkout
of the qualified GenAI source and built against the same ORT stack. This
retains complete physical histories for alternating hybrid conversations
instead of repeatedly falling back to an intermediate common prefix.

Target-only Engine control/candidate/control, two checkpoint slots, block256,
chunk2048, fixed256-block budget, 8449-token histories, exact512 outputs:

| Metric on six alternating switches | Control | Candidate | Control repeat |
|---|---:|---:|---:|
| Reused prefix tokens, every switch | 2048 | 8192 | 2048 |
| Mean TTFT (ms) | 1198.52 | 137.83 | 1194.87 |
| Pooled E2E TPS | 72.98 | 85.99 | 72.99 |

Every completed stream matched its cache-disabled reference, and repeated
reference streams matched across builds. Both controls failed the deep-reuse
assertion, not parity; the candidate passed the alternating-history suite.
This is a cache-retention improvement, not a decode-kernel TPS claim.
Relative to the bracketed controls, switch TTFT fell88.48% and pooled E2E
rose17.82%. Additional candidate GPU suites passed boundary replay,
active-owner lease/replacement, cancellation during prefill with retry,
and bounded-pool eviction/recovery, with512-token completed requests.
The port passed897 CPU Engine tests (one skip) and66 model-free QA tests.
Evidence and the standalone port patch remain in the isolated
`engine-hybrid-retention/results/` directory, outside this repository.

Batch-one MTP-one retains only one checkpoint slot in the qualified layout.
Its separate exact512 control/candidate/control test measured decode
123.82/123.97/123.82TPS and E2E108.04/108.22/108.09TPS; all twelve streams
matched, and every measured request reused4096 tokens. This is no demonstrated
MTP throughput improvement. No batch-one eligibility guard was relaxed.

The isolated folded branch built with CUDA/SM90 and `ENABLE_QWEN38_PLANE=ON`.
All59 targeted native tests passed (one disabled); these use scripted
devices/configuration checks, not a real model. Fifteen additional synthetic
CUDA checks passed: three bitwise recurrent-state transposes and twelve
pooled-key cases within rtol1e-4/atol1e-5 (max absolute error2.87e-6).
No local v12b graph package was available, so full-model parity, memory and
TPS for the folded backend remain **blocked**, not passed. The package
builder consumes an existing export; it does not convert the current model.
The qualified generic hybrid-prefix retention port is now integrated locally;
the original GenAI build passed897 CPU Engine tests (one skip) and91 combined
Python QA/benchmark tests after integration. The folded model backend remains
isolated. Neither change has been committed or pushed.
An additional GPU replay using the rebuilt original GenAI library also passed
the exact512 alternating-history suite; its record is
`engine-hybrid-retention/results/integrated.json`.

After the custom-op sources became available in `tianleiwu/qwen38-flash-ops`,
all eight libraries built locally and loading/CUDA smoke checks passed.
An isolated first-stage BF16 exporter was also implemented for the available
paged graphs: it converts FP16 weights with round-to-nearest-even, shares
unchanged external weights and retains the supported FP16 CPU Engram lookup
with an explicit BF16 boundary cast. Four exporter unit tests pass, including
all65536 FP16 bit patterns against torch rounding.

This first-stage export **is not a folded graph package**. Its exact512
precision gate diverges from the qualified FP16 stream at generated token41
in both target-only and MTP modes; cached/cold BF16 streams agree and MTP
failures remain zero. Diagnostic paged BF16 MTP decode121.50/E2E93.42TPS
must not be reported as folded-backend TPS or an output-equivalent speedup.
Static state layout, per-row rollback outputs and folded target/head graph
construction were subsequently implemented and tested as described below.
The FP16 production model remains unchanged.

### Reconstructed folded export: rejected accuracy gate (2026-10-14)

An isolated exporter now emits static target/head graphs, intermediate and
final recurrent states, a PLE table description, and an Engine-loadable
folded target/head graph using `qwen.fold::FoldMid` / `FoldTail`. It retains
the existing paged attention operators; this is **not** the upstream v12b
fused-attention/dense-prefill export. The earlier full-model "blocked"
status above is superseded by this completed reconstruction experiment,
not by qualification of the unavailable upstream package.

The two-row target component passed248 bitwise state/logit comparisons
against two single-row full-target calls on synthetic inputs. Twelve
exporter/operator tests pass, including CUDA convolution mid/final-state
equivalence and all finite FP8 PLE codes through the original FP16 lookup
boundary. A separate, default-off `QWEN38_PLE_FP16_ROUND_TRIP` data-plane
build preserves that boundary; the standard data-plane behavior is unchanged.
Native snapshot configuration is retained. `ConstantSharing` is disabled
for this export because sharing equal zero-valued CPU and GPU control
initializers caused a host dereference of device memory.

Generation qualification used one63-token chat prompt, capacity1024,
exactly512 outputs/request, and the qualified kernel flags. Sixteen
requests across normal folded, forced rejection, uncaptured folded,
paged BF16 reference and single-row-prefill BF16 reference completed with
zero MTP failures. Every mode repeated exactly; captured/uncaptured folded
streams also matched exactly. **Normal folded and forced-rejection streams
differ at generated token32**, and folded differs from the paged oracle
at token32 (single-row-prefill oracle at token28). Correcting PLE boundary
rounding did not resolve these differences.

| Mode | Decode TPS | E2E TPS | Qualification |
| --- | ---: | ---: | --- |
| Folded, captured | 77.51 | 68.92 | Rejected: token parity |
| Folded, forced rejection | 40.29 | 37.87 | Rejected: differs from normal folded |
| Folded, uncaptured | 63.89 | 56.65 | Matches captured folded, not reference |

These are **short-prompt diagnostic timings**, not an output-equivalent
speedup, cached throughput, or an8K-128K benchmark. Warmups are excluded.
No long-context performance promotion was attempted after the accuracy
gate failed. This static backend has no prefix cache or continuous batching.
No production backend/model was replaced, and no commit or push was made.

Local evidence: `engine-folded-v12b/static-evaluation-summary.json`,
`evaluation.json`, and the raw `static-final-*.json` results. The candidate
package is `/var/tmp/asonawane-qwen38-static-fold-native-state`; its export
manifest explicitly records the rejected qualification.

1. Record source commits, toolchain, GPU, active interpreter and package paths.
2. Confirm all loaded runtime libraries/hashes in the manifest correspond to
   the locally built optimized runtime. Stock libraries invalidate the run.
3. Require 20 configurations and 60 measured completed requests, zero MTP failures,
   expected context/MTP/overlay and prompt hashes. Historical reproduction
   requires 512 output tokens each; variable-output runs require recorded
   actual lengths consistent with EOS/capacity or the requested exact length.
4. Keep warmups excluded but saved; never remove slow or low-acceptance samples.
5. Require Engine `max_request_length >= actual_prompt + output_token_limit`.
   Earlier 5%/7% budgets failed admission, including after warmup; do not
   silently shorten prompts or count warmup results instead.
6. Compare manifests and raw counters, not just rounded table cells.
7. If GPUs are occupied or inference fails, preserve logs and report blocked/
   failed status. Do not mark performance validation complete.

CPU regression checks (with matching runtime packages plus `onnx`/`pytest`):

```bash
python -m pytest \
  "$GENAI_SRC/benchmark/python/qwen38/test_context_sweep.py" \
  "$GENAI_SRC/benchmark/python/qwen38/test_prepare_memory_candidate.py" -q
```

Publication checks passed 17 CPU tests and all five actual prompt hashes
matched the original measured sweep. The portable runner was not freshly
GPU-rerun before publication because the local GPUs were occupied.

For kernel history and the separate approximately 100 E2E TPS / below-80-GiB
8K memory experiment, see [README_MEMORY.md](README_MEMORY.md).
For additional build/background details, see [README.md](README.md).
