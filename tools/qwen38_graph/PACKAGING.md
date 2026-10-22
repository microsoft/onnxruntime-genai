# Qwen graph package construction and optimization

This standalone tooling change targets upstream GenAI main directly. It does
not require the graph-runtime PR to import the optimizer or run its unit tests.
Running the resulting `qwen38_flash_graph` packages requires the companion
GenAI graph runtime and a compatible ORT CUDA/contrib operator build.

## Tools

- `build_package.py`: separate verify/head graph package.
- `build_package_v12.py`: folded decode with optional dense prefill.
- `build_package_v12b.py`: shared-weight folded package and provider selection.
- `optimize_ported_package.py`: guarded graph rewrites and isolated optimized
  package construction from a compatible exported package.

Pass explicit source bundle, tokenizer/checkpoint, data-plane, provider, and
custom-operator library paths. Research-machine defaults are not portable.
Each script supports `--help`. PLE table construction additionally requires
the external `mobius` research package and checkpoint configuration; no
inherited Qwen4 model builder or model export changes are included here.
These tools package existing compatible graphs, not arbitrary checkpoints.
Generated CUDA provider options use the numeric arena strategy (`1`, meaning
`kSameAsRequested`) accepted by upstream GenAI; no symbolic-option parser
extension is required.

Use a fresh output bundle directory. Builders reject existing bundle destinations
(including dangling symlinks) before copying any graphs, libraries, or configuration;
they also reject destinations inside any resolved source bundle to avoid recursive
self-copying. They do not support in-place refresh of a prior package. File and directory symlinks
are preserved without walking external directory targets. Relative symlink targets
must remain reachable from the new package location.

The folded builder inventories sharing from the source models, including the source
filesystem/device, inode, byte range, element type, and shape. The destination pool
therefore preserves shared source initializers even with `--copy` or a
cross-filesystem hard-link fallback; unrelated tensors are not merged merely because
their names match. QSA normalization and rotary constants come only from the
packaged target/head graphs, never an implicit research reference graph.
BF16 constants support both inline raw/typed storage and external data; omitted
external-data lengths mean the bytes from the specified offset through EOF.
Optimizer-created weights retain sharing only when both the original pool
identity and the applied transformation match, not merely the generated name.

The optimizer rewrites supported normalization, rotary, GDN preparation,
gates, hyper-connections, FP8 projection, MoE residual, and sparse-attention
cache handoff chains. It preserves external weight sharing and avoids
mutating qualified input packages through hard-linked graph files.
FP8 weight fusion requires FP32 scales and allocates collision-free weight names.
Low-rank hyper-connection fusion requires a proven flat per-branch scale matching
the stream dtype and inputs available at the insertion point. Convolution fusion
preserves carry-output consumer ordering, including nested graph captures.
Head cache quantization rejects captured head outputs before rewriting; it does
not silently leave an incompatible unquantized cache handoff.
Generated graph schemas must match the companion ORT operators:
normalization/rotary, GDN/convolution, quantized MoE/FP8, hyper-connections,
packed sparse-attention/indexer, and speculative fold/state operations.

## Native convolution state-update fusion

After projection fusion, the optimizer can absorb private carry-layout
transposes and the first-token `Slice`/`Concat` snapshot chain into the existing
`CausalConvWithState` final/first-state outputs. It proves rank-three
channels-last input, weight/history/dilation compatibility, default extension
attributes, and exact slice semantics. Shared or externally captured
intermediates are not removed, and graph replacement is atomic.

On the experimental INT2 two-token verification package this removes six
graph nodes. Paired Nsight captures measured four fewer actual GPU kernels
per verification cycle (1046 to 1042), across 266 cycles per capture.
Twenty-four measured requests at 2K/8K/16K matched all 512 output token IDs
against the unfused baseline, with no generation failures. Timing varied
between processes; this does not establish a repeatable multi-percent gain.
Weights are unchanged, and full model accuracy qualification remains deferred.

## Experimental native decode preset

Recent native CUDA experiments are carried by the companion ORT PRs:

- [GDN/convolution #33471](https://github.com/microsoft/onnxruntime/pull/33471)
- [Sparse attention/indexer #33472](https://github.com/microsoft/onnxruntime/pull/33472)
- [Quantized MoE/FP8 #33473](https://github.com/microsoft/onnxruntime/pull/33473)

These are native dispatch changes, not additional GenAI graph-runtime features.
Use a compatible ORT build; environment flags alone cannot add missing kernels.
For an already qualified INT2-expert, FP8-projection package using two-token
verification (current token plus one draft), the experimental preset is:

```bash
export ORT_GDN_DECODE_WARPS=8
export ORT_GDN_S2_SPECIALIZE=1
export ORT_CAUSAL_CONV_S2_SPECIALIZE=1
export ORT_FP8_GEMV_PER_COLUMN_S2=1
export ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=0
export ORT_PACKED_SPARSE_INDEXER_COMPACT_TOPK=1
export ORT_DSA_INT8_VECTOR_STAGING=1
export ORT_DSA_SHARED_SCALE_MERGE=1
export ORT_MOE_ROUTING_WARP_SELECT=1
```

Set options before starting the process. The specialization switches default
to disabled and reject integer values other than 0/1. Unsupported shapes retain
their generic paths. Compact and radix QSA selection are mutually exclusive.
The MoE PR also fuses small-decode routing with stable expert-map construction.
Consult the ORT operator documentation for exact eligibility and fallback rules.

Keep the qualified package unchanged and evaluate a separate output bundle.
Do not overwrite hard-linked graphs, libraries, or manifests in place: replacing
a destination file atomically avoids modifying its source inode.

These experiments do not establish model accuracy or sustained 250 TPS at 8K.
The latest paired H200 measurements remained below the reference at 2K, 8K,
and 16K; full accuracy qualification is deferred. The focused native suites
pass with the new switches disabled and enabled. A broader GDN empty-request
regression still fails in the incremental retained-dependency build, including
with the unmodified PR-head kernel. Earlier experimental QSA/CSA
malformed-state-length failures also preclude a blanket qualification claim.
Do not interpret the preset or token-parity checks as a clean full-suite result
or deployment qualification; a clean upstream build and CI remain required.

## CPU unit tests

With ONNX, NumPy, and ml-dtypes available, run:

```bash
python -m unittest discover -s tools/qwen38_graph -p 'test_*.py'
```

Tests construct synthetic graphs and need no model weights or GPU.
Builder regressions also use temporary bundle directories to check safe destination
handling, symlink preservation, and sharing through hard links, copies, and fallback.
Unit tests and Python static/lint checks do not qualify complete generated
packages, native runtime compatibility, model accuracy, or GPU performance.
No benchmark/profiling/evaluation drivers, weights, or task metadata are
included.
