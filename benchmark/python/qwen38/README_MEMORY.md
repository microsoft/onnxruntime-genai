# Qwen3.8 Flash: approximately 100 E2E TPS below 80 GiB sampled

Measured on one H200, batch one, greedy Engine generation, **8,192 uncached
input tokens and 512 output tokens**, width-two MTP. Loading and warmup are
excluded from throughput; memory sampling includes them.

| Metric | Original baseline | Fast, higher-memory configuration | Latest memory candidate |
|---|---:|---:|---:|
| Independent E2E TPS | 30.89 | 100.48 | **100.58** |
| Decode TPS | 56.76 | 126.42 | **133.60** |
| TTFT | 7.57 s | 1.05 s | **1.27 s** |
| Sampled peak GPU memory | Not measured in original run | 100.09 GiB | **79.75 GiB** |

Latest throughput pools two independent five-request invocations:
100.71 and 100.44 E2E TPS. Memory is measured separately. The preceding
unfused candidate measured 99.79 E2E TPS and 79.95 GiB sampled peak.
An additional optional routing-scratch reuse screen measured **100.28
E2E TPS / 133.20 decode TPS / 79.73 GiB sampled peak**. Its fresh control
measured 100.01 TPS / 79.75 GiB: a small memory saving, not an established
speedup. The main table retains the preceding fused comparison.
The newest optional tiled-down candidate observes **79.58 GiB sampled
peak**, saving another 160 MiB device-wide. Throughput screens range from
about 100 to 104 TPS, with a repeat at 99.996 TPS; prefill is slightly
slower. It is a memory tradeoff, not a consistent throughput improvement.

## Changes that produced the result

1. **Scheduling and speculation:** enabled CUDA graphs, FP8 MTP and
   speculative EOS-floor handling; selected draft width two; enabled
   decoder/MTP dense GEMM tuning. Fixed the benchmark's persisted draft-width
   reporting.
2. **Sparse attention:** paired-channel value accumulation, grouped/tiled
   prefill with twelve query heads, split-partition decode, short-query
   verification tuning, interleaved dot products, vectorized K/V staging,
   and opt-in tensor-core QK/PV prefill.
3. **Sparse indexer:** hierarchical warp-sorted selection and padding-aware
   merges; bounded scoring and graph replay to live work rather than full
   cache capacity.
4. **Other kernels:** vectorized NVFP4 dequantization, branchless FP4
   conversion, exact FP8 conversion, greater small-N GEMV split-K
   parallelism, FP8 MTP scale/address hoisting, and temporally parallel
   prefill convolution with alias-safe state updates. Skipped unused
   grouped-GEMM autotuning when GEMV serves decode.
5. **CUDA allocation correctness:** implemented arena-config pointer
   set/get and factory ownership in ORT; rejected unknown pointer updates;
   forwarded numeric arena settings to GenAI's global allocator and checked
   pointer compatibility. Fixed BFC's zero-sized split when an exact-size
   allocation meets `max_dead_bytes_per_chunk=0`.
6. **Memory configuration:** same-as-requested arenas, zero dead-byte
   allowance, decoder/MTP memory patterns disabled, and a fixed 40-block
   cache. These are separate allocator pools, not a device-wide cap.
7. **Selected logits:** inserted `Gather(logits_indices)` before both
   decoder and MTP LM heads. Emit only requested vocabulary rows, retaining
   full hidden-state outputs and all verification rows.
8. **Packed NVFP4 prefill:** default-off grouped WMMA W4A16 kernel avoids
   full FP16 expert-weight buffers. Uses 64x128x64 tiles, FP32 accumulation,
   vectorized FP4 decoding, paired rounding, padded shared strides,
   asynchronous double-buffered input/packed-weight loads, one reusable
   dequantized tile, and shared input/epilogue storage. Decode GEMV is
   unchanged. A **384 MiB NVFP4 scratch bound** allows one 4,096-row MoE
   pass; integer prefill retains 256 MiB.
9. **Bounded indexer graph:** reduced thirteen indexers from 65,536 to
   2,560 compressed-key slots, matching the candidate's existing 40x256 KV
   allocation. Matched input/output shapes and configured context bounds;
   selected **4,096-token scheduler chunks** instead of latency-heavy
   eight-token chunks.
10. **Fused gate/up + SwiGLU epilogue:** optional
    `ORT_QMOE_NVFP4_FUSED_SWIGLU=1` writes the activated down-projection
    input directly, removing the 100 MiB gate/up intermediate workspace
    for a 4,096-row top-10 Qwen chunk and a separate activation launch.
    GEMM activation-dtype rounding, biases and alpha/beta/clamping are
    preserved. Fresh same-binary controls measured 99.54 E2E TPS /
    1.299 s TTFT versus 100.58 TPS / 1.266 s with fusion; sampled
    device-wide peak fell by 198 MiB. The larger device-wide saving reflects
    allocator pools, not just the removed workspace.
11. **Routing scratch lifetime reuse:** optional
    `ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH=1` overlaps dead sorting
    temporaries with expanded-input/down-output storage after sorting
    finishes on the same stream. Live routing maps remain independent.
    Saves 8.02 MiB workspace for the 4,096-row Qwen chunk; fresh
    device-wide sampled memory drops by 16 MiB. Reservations account for
    either buffer being larger.
12. **Tiled down output:** optional `ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT=1`,
    requiring fused packed prefill, gathers gate/up inputs directly and
    materializes only 1,024 down-output columns before deterministic
    top-k routing reduction. Saves exactly 120 MiB workspace for the
    4,096-row Qwen chunk. Preserves rounding/bias/reduction order, without
    atomic additions. TTFT increases by about 24 ms in final paired runs.

Original NVFP4 weights, scales and biases, CPU Engram, FP8 MTP and
tokenizer are unchanged. INT2/INT3, simulated INT4 activations, expert
pruning and expert offloading were screened but **not selected**: accuracy
regressions or throughput penalties prevented meeting the objective.
Their diagnostic paths remain explicitly opt-in.
Direct input gathering was also tried and rejected: it passed exact
numerical parity but increased TTFT by about 22 ms and saved no workspace.
It is revisited only inside tiled-down execution, where it enables actual
workspace savings; standalone direct gathering remains rejected.

## Reproduce

Build both `asonawane/qwen-38-flash-100-e2e` branches using the matching
CUDA ORT SDK; see [build instructions](README.md). A stock wheel lacks
these changes. Prepare a **separate** candidate from the original NVFP4
Engine package; this copies external weights and verifies their hashes:

```bash
cd "$GENAI_SRC/benchmark/python/qwen38"
python prepare_memory_candidate.py \
  --source /path/to/qwen_38_flash_nvfp4_engine \
  --destination /path/to/qwen38-memory-candidate
```

### Complete runtime environment

For an experimental **long-context selected-logits-only** candidate, add
`--preserve-context` to the preparation command. This inserts the selected-row
LM-head gather and its decoder/MTP input bindings, but leaves the original
indexer capacities, context limit, scheduler, cache budget, and all other
configuration settings unchanged. It is not the bounded 8K memory candidate
described here. Long-context performance and output equivalence must be
measured before deployment; do not assume the 8K results apply.

```bash
python prepare_memory_candidate.py \
  --source /path/to/qwen_38_flash_nvfp4_engine \
  --destination /path/to/qwen38-long-context-selected-logits \
  --preserve-context
```

Run this from `benchmark/python/qwen38` with `onnx` installed. The destination
must not exist. Preparation makes an independent copy of external weights
and verifies their SHA256 hashes; allow sufficient disk space and time.
Use matching `asonawane/qwen-38-flash-100-e2e` ORT/GenAI builds and point
your long-context Engine benchmark at the new directory. Keep the same
prompt, scheduler, MTP width, and cache budget for paired comparisons.
Ensure the Engine's `max_request_length` admits the prompt plus generation.
The bundled `benchmark_qwen_mtp.py` remains an 8K/512 benchmark; this option
does not turn it into a context sweep. No new runtime environment variable
or kernel rebuild is required when those matching builds are already installed.

Activate the isolated Python environment containing **both matching branch
builds** first. Set these variables before starting Python/model loading.
The following block selects the **lowest sampled-memory candidate**,
including all three optional follow-ups:

```bash
# Choose one otherwise idle GPU; replace 0 as appropriate.
export CUDA_VISIBLE_DEVICES=0

# Required packed prefill and selected attention/indexer optimizations.
export ORT_QMOE_NVFP4_PACKED_PREFILL=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
export ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1

# Optional follow-ups, enabled here for the 79.58 GiB sampled candidate.
export ORT_QMOE_NVFP4_FUSED_SWIGLU=1
export ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH=1
export ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT=1

# Disable rejected activation quantization and inherited diagnostic overrides.
export ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS=0
unset ORT_QMOE_CALIBRATION_CAPTURE_DIR ORT_QMOE_ROW_TILE_SIZE
unset ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK
unset ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO

python benchmark_qwen_mtp.py --mode mtp --repetitions 5 \
  --model /path/to/qwen38-memory-candidate --output /path/to/run1.json
```

| Variable | Value for tiled memory candidate | Purpose / dependency |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Idle GPU index | Select one GPU; kernels see it as device 0 |
| `ORT_QMOE_NVFP4_PACKED_PREFILL` | `1` | Enable packed weight-only prefill; default is off |
| `ORT_SPARSE_PREFILL_TENSOR_CORE_QK` | `1` | Selected tensor-core query/key prefill |
| `ORT_SPARSE_PREFILL_TENSOR_CORE_PV` | `1` | Selected tensor-core probability/value prefill; uses FP16 probability rounding |
| `ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY` | `1` | Enable bounded CUDA indexer replay |
| `ORT_QMOE_NVFP4_FUSED_SWIGLU` | `1` | Fuse gate/up activation; requires eligible packed prefill |
| `ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH` | `1` | Reuse dead sorting storage; requires eligible packed prefill |
| `ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT` | `1` | Bound down-output columns; requires packed prefill **and** fused SwiGLU |
| `ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS` | `0` | Keep rejected INT4 activation simulation disabled |

All three NVFP4 follow-up flags default to `0`. To reproduce the
lower-latency **fused-only** result in the main table, keep fused SwiGLU
at `1` and set routing reuse and tiled output to `0`. To use fused +
routing reuse without the tiled latency tradeoff, set only tiled output
to `0`. These flags do not replace the separate graph preparation step.

`ORT_QMOE_INT2_LARGE_EXPERTS` is **not required** for this NVFP4 model.
`PYTHONPATH` is needed only when deliberately using an isolated unpacked
runtime directory rather than installed matching wheels: set it to that
directory, not a stock ORT checkout. `LD_LIBRARY_PATH` is not a performance
flag; configure it only if your installation requires CUDA/cuDNN/SDK
library discovery. `CUDA_HOME`, `CUDNN_HOME`, `ORT_HOME`, `ORT_SRC` and
`GENAI_SRC` are build/path variables described in [README.md](README.md),
not additional inference optimization switches.

The preparation script persists scheduler 4,096, 40 KV blocks, draft
width two, GEMM tuning, memory-pattern and arena settings in the candidate
configuration; they are **not environment variables**. Run the Python
benchmark directly as above, **without an overlay**. The historical
`run_benchmark.sh` applies an 8K baseline overlay and must not be used
for this memory-candidate reproduction.

Use an otherwise idle GPU and repeat with a new output file. Do not use
memory markers/profiling for throughput validation. Sample GPU memory
separately through loading, warmup and execution. Models, binaries and
local raw evidence are not committed.

## Validation and limits

- CUDA provider build and **173 numerical/regression tests passed**,
  including exact fused/unfused FP16/BF16 parity, custom/default/clamped
  activations, exact workspace-byte reductions, empty experts, tiling,
  capture/replay and fallback coverage. Routing-reuse tests include 512
  experts and both routing/input buffer-size orderings.
  Publication validation also passed both C++ arena regressions and five
  graph-preparation tests; the portable transformation reproduced the
  measured decoder/MTP graphs byte-for-byte and matched their configuration.
- All **110 control-correct answers across 128 observed cases** were
  preserved against fresh dense NVFP4 controls; three token streams
  differed without answer regressions in the preceding candidate.
  Fresh fused/unfused comparisons matched all 128 token streams exactly.
  Fresh routing-reuse flag-off/flag-on comparisons also matched all 128.
  Fresh tiled-down comparisons likewise matched all 128.
  Bounded/unbounded graph comparisons with the same packed provider also
  matched all 128 streams exactly.
- No MTP, empty-output or repetition failures were observed.
- Indexer/weight transformations preserve initializer metadata and weight
  bytes. The stock ONNX checker cannot register the source MTP's
  `SimplifiedLayerNormalization` at opset22; actual ORT execution passed.
- **100.58 TPS is a measured aggregate, not a throughput guarantee.**
- **79.7498 GiB is sampled, not a hard cap:** only approximately **256 MiB**
  headroom; the fused sample gap reached 125 ms. Actual 80 GiB hardware,
  concurrent traffic and arbitrary workloads remain unqualified.
  Routing scratch reuse separately observes 79.7322 GiB and about 274 MiB
  headroom, with a maximum 120 ms sample gap; the same limitations apply.
  Tiled-down separately observes 79.5760 GiB and about 434 MiB headroom,
  with maximum gaps of 114 ms initially and 129 ms in the final-source
  repeat. Strict at-least-100 E2E is not established
  for that option.
- Configured context is 10,240 tokens; the tested Engine enforces a
  **9,217-total-token request ceiling** including prompt and generation.
  This is **not a 262K-context replacement**.

See [RESULTS.md](RESULTS.md) for intermediate measurements and rejected
experiments, and [CUDA MoE documentation](https://github.com/microsoft/onnxruntime/blob/asonawane/qwen-38-flash-100-e2e/docs/contrib_ops/cuda/moe_qmoe.md)
for kernel details.
