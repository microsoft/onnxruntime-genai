# Qwen INT2 target + FP8 DFlash2: source, conversion, build and run

This guide describes the combined integration source snapshots, not an upstream
release. The supported package uses three DFlash2 drafts, four-row target
verification, BF16 activations, batch one, greedy decoding and Linux CUDA.
Seven-draft support is not implemented.

## Pinned sources

| Repository | Integration branch | Runtime source commit |
| --- | --- | --- |
| [ORT](https://github.com/microsoft/onnxruntime/tree/asonawane/remaining-standard-op-fusions) | `asonawane/remaining-standard-op-fusions` | `13aa2a1e428a1e31ebc6f027e774674bc27d0aef` |
| [GenAI](https://github.com/microsoft/onnxruntime-genai/tree/asonawane/dflash2-device-pipeline) | `asonawane/dflash2-device-pipeline` | `db154bc477b86aca03eed47694252c743bb56459` |

Both commits are published and contain the required source; no uncommitted
overlay, model-local helper or custom-op library is required. Later GenAI
documentation commits do not change this runtime source pair. Downloaded model
assets and checkpoints are not part of either repository.

The ORT branch combines generic CUDA operator migrations, INT2/FP8 paths, S4
attention/hyperconnection tuning, recurrent prefix state, native fold v2,
schemas, tests and documentation. GenAI includes the graph executor, DFlash
pipeline, export/packing/context tools and integrated CUDA data plane.
It builds the helper kernels into the normal `libonnxruntime-genai-cuda.so`;
the core uses a version/size-checked interface without linking cudart.

The incremental review PRs remain separate:
[ORT #33546](https://github.com/microsoft/onnxruntime/pull/33546),
[#33547](https://github.com/microsoft/onnxruntime/pull/33547),
[#33548](https://github.com/microsoft/onnxruntime/pull/33548),
[#33550](https://github.com/microsoft/onnxruntime/pull/33550) and
[GenAI #2771](https://github.com/microsoft/onnxruntime-genai/pull/2771).
Their bases have additional prerequisites. Do not cherry-pick only these PR tips
onto main and expect the full integration. Use the pinned combined sources:

```bash
# Use new directories, not an existing dirty working tree.
git clone --branch asonawane/remaining-standard-op-fusions \
  https://github.com/microsoft/onnxruntime.git ort-integration
git -C ort-integration checkout --detach 13aa2a1e428a1e31ebc6f027e774674bc27d0aef
git -C ort-integration submodule update --init --recursive
git clone --branch asonawane/dflash2-device-pipeline \
  https://github.com/microsoft/onnxruntime-genai.git genai-integration
git -C genai-integration checkout --detach db154bc477b86aca03eed47694252c743bb56459
git -C genai-integration submodule update --init --recursive
```

## Build prerequisites

Validated platform: H200/Hopper (`90a`), CUDA 13.3, cuDNN 9, GCC 13, Python 3.12,
CMake and Ninja. Use a compatible driver. Other GPU/toolkit combinations need
their own validation. The existing development environment is `qwen-flash`.
Use repository build requirements in a new environment. Conversion additionally
uses NumPy, ONNX, onnx_ir, PyTorch and safetensors; CPU PyTorch suffices for the
drafter exporter and context tool.

From the directory containing the two new clones:

```bash
export ROOT="$PWD"
export ORT_SRC="$ROOT/ort-integration"
export GENAI_SRC="$ROOT/genai-integration"
export ORT_BUILD="$ROOT/build/ort"
export ORT_SDK="$ROOT/build/ort-sdk"
export GENAI_BUILD="$ROOT/build/genai"
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/path/to/cudnn  # Must contain include/ and lib/.
export TOOLS="$GENAI_SRC/tools/qwen38_graph"
export MODEL=/models/qwen-3.8-flash-next-int2-fp8-lm-head-dflash2
```

Allow about 88.01 GiB logical storage for the prepared model, additional space
for conversion/builds, sufficient host RAM for host-registered PLE tables, and
GPU memory/headroom above the observed 67.13 GiB full-window peak.

## Build ORT

Use the CUDA **plugin** configuration and matching source headers/core/provider.
Contrib operators and low-bit types must not be disabled:

```bash
bash "$ORT_SRC/build.sh" \
  --config Release --update --build --build_shared_lib \
  --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --cmake_generator Ninja --build_dir "$ORT_BUILD" \
  --parallel 4 --nvcc_threads 1 --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90a \
    onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON \
    onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON \
    onnxruntime_BUILD_UNIT_TESTS=ON onnxruntime_USE_NCCL=OFF

mkdir -p "$ORT_SDK/include" "$ORT_SDK/lib"
cp -a "$ORT_SRC/include/onnxruntime/core/session/." "$ORT_SDK/include/"
cp -a "$ORT_BUILD/Release"/libonnxruntime.so* "$ORT_SDK/lib/"
cp -a "$ORT_BUILD/Release/libonnxruntime_providers_cuda.so" "$ORT_SDK/lib/"
cp -a "$ORT_BUILD/Release/libonnxruntime_providers_shared.so" "$ORT_SDK/lib/"
```

`--skip_tests` avoids running the full suite during the build; it is not a test
pass. Existing plugin targets can be rebuilt with:

```bash
cmake --build "$ORT_BUILD/Release" \
  --target onnxruntime onnxruntime_providers_cuda_plugin onnxruntime_provider_test -j 4
```

## Build GenAI

```bash
cmake -S "$GENAI_SRC" -B "$GENAI_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_GUIDANCE=OFF \
  -DENABLE_PYTHON=ON -DBUILD_WHEEL=ON \
  -DENABLE_TESTS=ON -DENABLE_CUDA_KERNEL_TESTS=ON \
  -DORT_HOME="$ORT_SDK" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=90a \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build "$GENAI_BUILD" -j 4
```

The importable package is under `$GENAI_BUILD/wheel/onnxruntime_genai`.
No separate `qwen38_plane` target or model-local `.so` is needed.
Deploy matching GenAI core/CUDA add-on binaries together.

## Conversion starting points and limitations

**The prepared target is required. A portable raw-Hugging-Face-checkpoint to
optimized INT2/S4 target converter is not yet provided.** Target preparation
included INT2 packing, auxiliary hidden-state capture, multi-row recurrent/prefix
state and operator migrations. The old research bundle assemblers and generic
GenAI model builder do not reproduce this target by themselves.

Use either:

- The prepared native-262K package: skip conversion and run it.
- A compatible native/custom-fold **128K DFlash S4** package with unified prefill,
  correct PLE/shared-pool manifests and the matching target/draft contracts:
  migrate if necessary, consolidate, then extend context.

Do not overwrite the working package or edit immutable hardlinked data in place.
The following output paths must not already exist.

### Fold migration (only if still legacy)

```bash
python "$TOOLS/convert_fold_to_contrib.py" \
  --input /models/legacy/target.onnx \
  --output /models/legacy/target.contrib.onnx
```

The converter accepts the specific legacy multi-row `com.microsoft` fold contract.
Output must be beside input to preserve relative weights. In a separate candidate
config, point `model.decoder.filename` at the new graph. Skip if already v2.
This is not a general custom-op migration tool: remove registrations only after
all operators are available natively. The consolidator rejects remaining custom-op
registrations. See [fold migration details](./README.md#native-speculative-fold-migration).

### Consolidation and native context

```bash
python "$TOOLS/consolidate_dflash_package.py" \
  --source /models/compatible-dflash-s4-128k \
  --output /models/consolidated-128k \
  --report /models/consolidation-report.json
python "$TOOLS/extend_dflash_context.py" \
  --source /models/consolidated-128k \
  --output /models/native-262k
```

The first tool packs target weights, shared-pool ranges and PLE tables into one
`model.onnx.data`, verifies copied ranges by SHA-256, retains the separate drafter
through immutable hardlinks and drops inactive sessions and `plane_library`.
Small shape constants stay embedded. It is not intended for arbitrary S2/MTP
packages with required standalone head execution.

The second tool requires the expected consolidated **131072 public / 131076 KV**
input contract. It updates target KV dimensions, compressed-index capacity,
fold capacity, rotary caches and shared-pool rotary bindings together. Original
rotary rows and learned weights are preserved. Do not rerun it on a 262K package
or just increase config limits.

Final contract: 262144 total prompt/output tokens, 262148 internal KV slots,
8192-token prefill chunks, draft block size 4, three drafts, selector top-k 16,
draft sliding window 2048. The runtime preallocates full target KV/index capacity;
allocation is not prompt-sized.

### Optional drafter-only export

After setting the runtime environment below, use a local
`incoai/Qwen3.8-Flash-Next-DFlash2` checkpoint and the compatible target:

```bash
python "$TOOLS/export_dflash2.py" \
  --checkpoint /models/dflash2-checkpoint \
  --target "$MODEL/model.onnx" \
  --output /models/drafter-new --precision fp8 --drafts 3
```

This exports only draft graph/data, not the target, manifests or complete config.
It takes the target's exact embedding and FP8 head. Body weights use FP8 with
BF16 activations; BF16 body mode is available but not separately qualified.
The native rotary length comes from checkpoint `max_position_embeddings` plus
four speculative guards. Replace graph/data only in an independent candidate,
keep config/IO metadata aligned, and validate before activation.

## Run

The final package has `model.onnx`, `model.onnx.data`, `dflash2.onnx`,
`dflash2.onnx.data`, `weights_pool.txt`, `ple_table.txt`, `genai_config.json`,
tokenizer/config files and the chat template. No `libs/`, bundled provider/runtime,
unused standalone heads or `numerics_original` is needed.

```bash
export CUDA_VISIBLE_DEVICES=1  # Select a free physical GPU; becomes device 0.
export ORT_LIB_PATH="$ORT_BUILD/Release/libonnxruntime.so"
export QWEN_PROVIDER="$ORT_BUILD/Release/libonnxruntime_providers_cuda.so"
export PYTHONPATH="$GENAI_BUILD/wheel"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64:$ORT_BUILD/Release"
unset LD_PRELOAD
export ORT_DSA_INT8_GROUPED_MAX_ROWS=4
export ORT_QMOE_INT2_DECODE_MAX_EXPANDED_ROWS=40
export ORT_HC_DECODE_ROWS=1
export ORT_INT2_S4_CTA_N=2
export ORT_INT2_S4_FUSED_FINALIZE=1
export CUDA_MODULE_LOADING=EAGER
export ORT_MOE_ROUTING_WARP_SELECT=2
export ORT_ENABLE_FP4_GEMV=1
export ORT_NVFP4_FC1_EXPERT_REUSE=0
export ORT_FORCE_DETERMINISTIC_MOE=1
export ORT_GDN_DECODE_WARPS=8
export ORT_GDN_S2_SPECIALIZE=1
export ORT_CAUSAL_CONV_S2_SPECIALIZE=1
export ORT_FP8_GEMV_PER_COLUMN_S2=1
export ORT_PACKED_SPARSE_INDEXER_COMPACT_TOPK=1
export ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=0
export ORT_DSA_INT8_VECTOR_STAGING=1
export ORT_DSA_SHARED_SCALE_MERGE=1

nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv
python "$TOOLS/smoke.py" "$MODEL" --tokens 64 --provider "$QWEN_PROVIDER" \
  --prompt "Explain how speculative decoding preserves greedy target output."
```

This uses the chat template and normal Generator API with production EOS settings.
Do not use the example's `--no-mtp` option for this folded DFlash package.
If the GPU is occupied, wait or select a free device; short prompts still require
full-capacity allocations. Do not stop other workloads.

## Validate and interpret results

With the environment above:

```bash
"$GENAI_BUILD/engine_unit_tests" --gtest_filter='*Qwen38*:*Dflash2*:*DFlash2*'
"$ORT_BUILD/Release/onnxruntime_provider_test" --gtest_filter='FoldContribV2Test.*'
(
  cd "$TOOLS"
  python -m unittest test_consolidate_dflash_package test_extend_dflash_context \
    test_convert_fold_to_contrib test_integrated_plane -v
)
```

Before publication, the existing combined ORT core/plugin/provider-test and GenAI
builds passed, along with five fold GPU tests, 150 native GenAI tests (one skipped,
one disabled), 15 tool tests and four integrated CUDA tests. These are incremental
build checks, not a new build from empty directories or a full-suite qualification.

Earlier helper-free H200 validation on these runtime source contents preserved
512 token IDs at 2048, 8192, 16384, 32768 and 8193 prompt tokens (warmup/measured),
passed 262112 prompt + 32 output tokens twice, and passed EOS/full-logit checks.
Fallback/failure/disable counters were zero; process mappings contained no
model-local helper. Full-window sampled peak was 67.13 GiB.
Full-model reruns during publication were not possible with other workloads
occupying the GPUs. These checks do not establish long-context task accuracy or
accuracy qualification of the experimental quantized target.

For each candidate, compare token IDs against a retained reference, check repeated
requests/EOS/full logits and assert zero `standard_fallback_steps`,
`dflash2_failures` and `dflash2_disables`. Do not infer correctness from throughput.

Troubleshooting:

- Legacy `plane_library`: remove the config setting and use the matching GenAI
  build, rather than restoring an external helper.
- Missing/mismatched `GetQwen38PlaneFunctions`: deploy matching core/CUDA add-on.
- Unknown contrib ops: check pinned ORT core/provider and explicit library paths.
- Wrong Python runtime: print `onnxruntime_genai.__file__` and fix `PYTHONPATH`.
- Allocation/host-registration errors: check free GPU/host memory and locked-memory
  limits. `past_present_share_buffer` does not make this executor input-sized.
- Context-boundary errors: check graph, rotary and shared-pool capacities together.

More graph contracts and migration details: [README](./README.md).
