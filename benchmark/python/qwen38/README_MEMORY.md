# Qwen3.8 Flash: approximately 100 E2E TPS below 80 GiB sampled

Measured on one H200, batch one, greedy Engine generation, **8,192 uncached
input tokens and 512 output tokens**, width-two MTP. Loading and warmup are
excluded from throughput; memory sampling includes them.

| Metric | Original baseline | Fast, higher-memory configuration | Latest memory candidate |
|---|---:|---:|---:|
| Independent E2E TPS | 30.89 | 100.48 | **99.79** |
| Decode TPS | 56.76 | 126.42 | **133.33** |
| TTFT | 7.57 s | 1.05 s | **1.30 s** |
| Sampled peak GPU memory | Not measured in original run | 100.09 GiB | **79.95 GiB** |

Latest throughput pools two independent five-request invocations:
99.64 and 99.93 E2E TPS. Memory is measured separately.

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

Original NVFP4 weights, scales and biases, CPU Engram, FP8 MTP and
tokenizer are unchanged. INT2/INT3, simulated INT4 activations, expert
pruning and expert offloading were screened but **not selected**: accuracy
regressions or throughput penalties prevented meeting the objective.
Their diagnostic paths remain explicitly opt-in.

## Reproduce

Build both `asonawane/qwen-38-flash-100-e2e` branches using the matching
CUDA ORT SDK; see [build instructions](README.md). A stock wheel lacks
these changes. Prepare a **separate** candidate from the original NVFP4
Engine package; this copies external weights and verifies their hashes:

```bash
python prepare_memory_candidate.py \
  --source /path/to/qwen_38_flash_nvfp4_engine \
  --destination /path/to/qwen38-memory-candidate

export ORT_QMOE_NVFP4_PACKED_PREFILL=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
export ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY=1
export ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS=0
unset ORT_QMOE_CALIBRATION_CAPTURE_DIR ORT_QMOE_ROW_TILE_SIZE
unset ORT_QMOE_INT_DEQUANT_MAX_SCRATCH_BYTES ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK
python benchmark_qwen_mtp.py --mode mtp --repetitions 5 \
  --model /path/to/qwen38-memory-candidate --output /path/to/run1.json
```

Use an otherwise idle GPU and repeat with a new output file. Do not use
memory markers/profiling for throughput validation. Sample GPU memory
separately through loading, warmup and execution. Models, binaries and
local raw evidence are not committed.

## Validation and limits

- CUDA provider build and **54 numerical/regression tests passed**.
  Publication validation also passed both C++ arena regressions and five
  graph-preparation tests; the portable transformation reproduced the
  measured decoder/MTP graphs byte-for-byte and matched their configuration.
- All **110 control-correct answers across 128 observed cases** were
  preserved against fresh dense NVFP4 controls; three token streams
  differed without answer regressions. Bounded/unbounded graph comparisons
  with the same packed provider matched all 128 streams exactly.
- No MTP, empty-output or repetition failures were observed.
- Indexer/weight transformations preserve initializer metadata and weight
  bytes. The stock ONNX checker cannot register the source MTP's
  `SimplifiedLayerNormalization` at opset22; actual ORT execution passed.
- **99.79 TPS is approximately 100, not strictly at least 100.**
- **79.9451 GiB is sampled, not a hard cap:** only approximately **56 MiB**
  headroom; sample gaps reached 120-126 ms. Actual 80 GiB hardware,
  concurrent traffic and arbitrary workloads remain unqualified.
- Configured context is 10,240 tokens; the tested Engine enforces a
  **9,217-total-token request ceiling** including prompt and generation.
  This is **not a 262K-context replacement**.

See [RESULTS.md](RESULTS.md) for intermediate measurements and rejected
experiments, and [CUDA MoE documentation](https://github.com/microsoft/onnxruntime/blob/asonawane/qwen-38-flash-100-e2e/docs/contrib_ops/cuda/moe_qmoe.md)
for kernel details.
