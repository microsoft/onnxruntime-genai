# Qwen graph execution through Generator

## Published combined stack

Both repositories publish `asonawane/qwen-combined-local`. The historical
`-local` suffix does **not** mean these branches are local-only. Clone these
branches instead of cherry-picking the individual PRs:

```bash
git clone --branch asonawane/qwen-combined-local \
  https://github.com/microsoft/onnxruntime.git
git clone --branch asonawane/qwen-combined-local \
  https://github.com/microsoft/onnxruntime-genai.git
```

The companion ORT revision for this GenAI integration is
`555dd69d236f7da7c7f9e40b72d0f8f25f81e829`. Pin ORT to that commit when
reproducing this integration, and record `git rev-parse HEAD` in GenAI as
well. Individual PR branches are preserved and remain the review surfaces.

Included source heads:

| Repository | PR branch | Integrated commit |
| --- | --- | --- |
| ORT | `asonawane/main-ops-normalization-rotary` | `17662fc00406e9058cca4a57b1907fa52b1acd76` |
| ORT | `asonawane/main-ops-hyperconnection-tensor` | `e587c56161de193f06fc722e890b8cb144c39c52` |
| ORT | `asonawane/main-ops-gdn-convolution` | `c1e138f9e7f5c05df679b618a3455114f623f779` |
| ORT | `asonawane/main-ops-sparse-attention-indexer` | `9356555892c9779438cc08d533486daaf84c2317` |
| ORT | `asonawane/main-ops-quantized-moe-fp8` | `25d286b5a0acb04957e81896b8bfb0e28e89b235` |
| ORT | `asonawane/main-ops-mtp-fold-state` | `24625bf3d29342960b73d564958a76551c91e437` |
| ORT | `asonawane/qwen-cuda-kernel-optimizations` | `b7be0b3d5386cd4a7dbe1441e15e51fe15e0f320` |
| ORT | `asonawane/generic-cuda-memory-kernels-main` | `a788b476bdbd2aba3e8e2f9e09f044ecaf69124d` |
| GenAI | `asonawane/main-qwen-graph-runtime` | `5bc476f3d679f7c8a09b0797829a73f188c9b407` |
| GenAI | `asonawane/main-qwen-package-tooling` | `438cb2187852b643104dfb8c1d8d5dfaae740d66` |
| GenAI | `asonawane/qwen-mtp-prefix-cache-cuda-replay` | `9005d0d4db5e91d4a61afde5746069a3e9b0cef5` |
| GenAI | `asonawane/generic-cuda-arena-main` | `56a84bd5064379ce929e7daf04a9ad1395f59479` |

Where upstream Python builder refactoring overlapped, the combined branch
retains the builder layout from the MTP/runtime snapshot and its latest
fixtures. The standalone graph packagers and optimizer use the reviewed
package-tooling implementation. The shared CUDA arena normalization
supersedes the earlier local symbolic-arena parsing workaround.

**Qualification:** this published pair has source/merge checks only. No
build, test, linter or GPU qualification was run for this publication.
Earlier measured binaries are not builds of these exact combined commits;
their throughput, accuracy and memory figures must not be attributed to
this source pair. Rejected HC and paired-column NVFP4 experiments were not
promoted. Weights, prebuilt libraries, local benchmark drivers and private
reference modules are not distributed by these branches.

### Building the pair

Use Linux with a supported NVIDIA driver, CUDA toolkit and cuDNN development
installation. H200 uses architecture `90`; adapt the architecture for other
GPUs. Install the repositories' normal build dependencies in your own
environment. The following is a source build recipe, not a claim that this
new source pair has already passed it:

```bash
# From the parent directory containing both clones.
# CUDA_HOME and CUDNN_HOME must name your local toolkit/development installations.
git -C onnxruntime switch --detach 555dd69d236f7da7c7f9e40b72d0f8f25f81e829
(
  cd onnxruntime
  ./build.sh --config Release --update --build --build_shared_lib \
    --build_dir "$PWD/build/qwen" --use_cuda \
    --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
    --cmake_generator Ninja --parallel 8 --skip_tests \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
      onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON onnxruntime_USE_FP4_QMOE=ON
)
mkdir -p ort-sdk/include ort-sdk/lib
cp -a onnxruntime/include/onnxruntime/core/session/*.h ort-sdk/include/
cp -a onnxruntime/build/qwen/Release/libonnxruntime*.so* ort-sdk/lib/
cmake -S onnxruntime-genai -B onnxruntime-genai/build/qwen \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DORT_HOME="$PWD/ort-sdk" \
  -DENABLE_PYTHON=ON -DENABLE_TESTS=OFF
cmake --build onnxruntime-genai/build/qwen --parallel 8 \
  --target onnxruntime-genai python qwen38_plane
```

Use compatible **existing exported graphs** and tokenizer/PLE data.
See [package construction](PACKAGING.md); this integration does not export
or quantize an arbitrary checkpoint automatically. In an isolated model
package, install the newly built ORT runtime/CUDA provider, GenAI library,
and `libqwen38_plane.so` at the paths used by its configuration. Do not
overwrite a qualified package's libraries through hard links.

### Upgrading legacy folded graphs

The reviewed ORT `FoldMid2` and `FoldTail` contracts return explicit control
state instead of relying on undeclared input mutation. The optimizer
automatically upgrades legacy fold nodes during normal package optimization.
For an already optimized or compact package, upgrade only its target graph:

```bash
python onnxruntime-genai/tools/qwen38_graph/optimize_ported_package.py \
  /path/to/isolated-package/target.onnx \
  /path/to/isolated-package/target-explicit-fold.onnx \
  --upgrade-fold-state-only
```

The output must be a new file beside the source graph, so relative external
weight paths remain valid. This command does not read or rewrite the weight
payloads, change precision, or overwrite the original graph. Set
`model.decoder.filename` in that isolated package's `genai_config.json` to
`target-explicit-fold.onnx`. Noncompact packages can use the same command
beside their `target/model.onnx`, then set the filename to the corresponding
relative path. Do not run the full weight-fusing optimizer again on a
single-prepacked INT2 package.

The upgraded graph exposes `fold.updated_ctl` and
`fold.updated_ctx_commit`. GenAI binds them to the persistent control/context
buffers, and the graph explicitly connects the mid-state update to the tail.
The upgrader rejects partial contracts, name collisions and nested control
captures rather than guessing. Older seven-/one-output fold graphs cannot
be loaded by the newly reviewed ORT fold kernels without this upgrade.
This migration and its model-dependent behavior still require qualification
with the exact newly built pair.

### Running

Point the loader at the freshly built ORT SDK and GenAI Python package,
then use the normal smoke example:

```bash
export ORT_LIB_PATH="$PWD/ort-sdk/lib/libonnxruntime.so"
export LD_LIBRARY_PATH="$PWD/ort-sdk/lib:$PWD/onnxruntime-genai/build/qwen${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/onnxruntime-genai/build/qwen/wheel${PYTHONPATH:+:$PYTHONPATH}"
python onnxruntime-genai/tools/qwen38_graph/smoke.py \
  /path/to/isolated-package \
  --provider "$PWD/ort-sdk/lib/libonnxruntime_providers_cuda.so" --tokens 64
```

Compatible large-expert INT2/shared packing and NVFP4 packed prefill select
automatically. No `ORT_QMOE_INT2_LARGE_EXPERTS=1`,
`ORT_QMOE_INT2_SHARE_PACKED=1` or `ORT_QMOE_NVFP4_PACKED_PREFILL=1` is needed.
Explicit zero opt-outs remain available, subject to the documented
single-prepacked INT2 requirement that a compatible packed path stays enabled.
Hardware, layout, shape and activation restrictions still apply.

The `qwen38_flash_graph` model type exposes fixed-shape CUDA graph execution
through the normal Model, Tokenizer, and Generator interfaces. It supports
greedy MTP(1), folded decode, dense chunked prefill, and shared device weights.

## Build and dependencies

Build GenAI with CUDA enabled on Linux and build the `qwen38_plane` target.
The model loads this data-plane library from its package. A compatible ORT
CUDA plugin and the companion generic contrib operator stack are required:
packed sparse-attention/indexer caches, speculative fold/state operations,
normalization/rotary, GDN/convolution, hyper-connections, and quantized MoE/FP8.
These operators are not provided by this GenAI change. Existing upstream
allocator, model-package, decoder-state, and builder APIs are preserved.

Use `model.type = "qwen38_flash_graph"` and the `model.qwen38_graph` settings
in `src/config.h`. Configure target/head graph paths, optional folded and
dense-prefill sessions, the data-plane library, and PLE table relative to the
model package. Provider registration and session-option inheritance use the
existing GenAI conventions. Model packages must already contain compatible
fixed-shape graphs; this runtime is not a standalone model exporter.

## API and limits

Use the regular Generator lifecycle with greedy search. An accepted draft
produces two tokens internally; the API returns one per call.

- Linux CUDA, batch size one, greedy decoding only.
- One active generator per model.
- Append prompt chunks only before generation starts.
- Only `rewind_to(0)` is supported.
- Cache capacities bound the combined prompt and generated lengths.
- Production EOS settings control termination.
- `get_logits()` is available after appending the prompt and while generation is
  active, including accepted/buffered MTP emissions. Repeated reads do not advance
  generation. After generation completes (`is_done()`), `get_logits()` throws.

## Validation

`test_api.py` exercises a supplied compatible model package.
It checks repeated logits reads and exact emission counters after every emitted
token, observes an accepted/buffered MTP cycle, verifies completed-generation
logits rejection, and repeats these checks after `rewind_to(0)` clears counters.
Logits argmax must predict the next emitted token; repeated reads preserve both
sequence and statistics. Each step also requires
`tokens_queued == tokens_emitted + tokens_buffered + tokens_discarded`.
These model-dependent regressions remain pending native/GPU qualification;
CPU/mock checks validate test logic only, not device execution or row selection.
`test_plane_kernels.py` checks a supplied CUDA data-plane library.
`test_prompts.py` checks supplied prompt/reference cases.
`smoke.py` and `smoke_cpp.cpp` provide integration examples.
These tests require compatible model artifacts and native libraries; they
are not standalone CPU unit tests. Unit/static checks do not replace a native
rebuild and GPU/model qualification of this exact main-based branch.

The Python API scripts accept `--provider /path/to/libonnxruntime_providers_cuda.so`
(or `QWEN_PROVIDER`). Registration occurs only when a path is supplied;
otherwise the normal GenAI provider-loading configuration is used.

The reference implementations are external inputs, not included in this
repository. Supply explicit local paths; no private home-directory layout is
assumed:

```bash
python tools/qwen38_graph/test_prompts.py /path/to/package \
  --prompts-module /path/to/references/prompts.py --out prompt-results.json
python tools/qwen38_graph/test_plane_kernels.py /path/to/package \
  --plane-module /path/to/references/plane.py \
  --handoff-module /path/to/references/handoff_lib.py \
  --checkpoint /path/to/checkpoint --out kernel-results.json
```

`prompts.py` must expose `make(kind, length, seed)` returning at least the
requested number of token IDs (`"text"`, 20000, 77 in this test). `plane.py`
must expose `PleDevice(lookup, capacity)` and the ctypes `PleParams` structure
matching the supplied library. `handoff_lib.py` must expose
`pooled_block_keys(raw, knorm, cos, sin, epsilon)` returning the reference
pooled keys. The kernel test additionally requires PyTorch/CUDA and a compatible
Mobius installation providing the Qwen4Exp configuration and CPU PLE lookup.
`--checkpoint` must be a local compatible Hugging Face checkpoint directory
containing `config.json` and its weights. Module paths must identify existing
`.py` files. These files execute as Python code; use only trusted references.
Their parent directories are added to the import path for sibling dependencies.

Prompt validation requires sequence consistency and exactly 48 generated tokens
for every run, matching first tokens, and finite relative-L2 logits error at most
`--logits-rtol` (default `1e-3`, finite and positive). This applies to fast/steps
parity, state leakage, and, with `--long`, chunked/single and repeat comparisons.
Acceptance rates are diagnostic only. A failed comparison sets `all_ok=false`
and exits nonzero; specify a different tolerance explicitly if qualifying a
different numeric configuration.

Benchmark/profiling/evaluation drivers, weights, inherited model builders,
and local task metadata are intentionally excluded.
