# Qwen graph execution through Generator

The `qwen38_flash_graph` model type exposes fixed-shape CUDA graph execution
through the normal Model, Tokenizer, and Generator interfaces. It supports
greedy MTP(1), folded decode, dense chunked prefill, and shared device weights.

## DFlash2 device pipeline

This incremental integration also supports a three-token DFlash2 drafter with
four-row folded verification, through both Generator and the native Engine.
It requires the companion ORT feature branches and a compatible four-row target
package; it is not independently usable with an upstream-main ORT build.

The drafter and target share a CUDA stream. Persistent pinned staging and GPU
lattice selection remove per-round allocations and host lattice readback.
Unused MTP sessions are omitted for DFlash-only packages, prompt hidden storage
is sized to the configured prefill chunk, and prompt argmax stays on device
until full logits are requested. Generator preserves logits for each buffered
emission before scheduling the next verifier pass.

Export the drafter using a local checkpoint and a compatible target graph:

```bash
python export_dflash2.py --checkpoint /models/dflash2-checkpoint \
  --target /models/target/target.onnx --output /models/drafter-fp8 \
  --precision fp8 --drafts 3
```

The exporter uses BF16 activations, page size 128, and the target's exact
embedding and FP8 head. `--precision bf16` disables body-weight FP8
quantization; it does not replace the target head. The output directory must
not already exist. This exports the drafter, not the target, auxiliary feature
capture graphs, shared-weight pool, or complete runtime configuration.
Supply a package configured with `model.dflash2` and four-row folded graphs.
`model.qwen38_graph.prefill_chunk_tokens` defaults to 8192.
`model.qwen38_graph.prefill.unified` is optional and requires prefill enabled.

For the qualified H200 workload, set these before loading the model:

```bash
export ORT_DSA_INT8_GROUPED_MAX_ROWS=4
export ORT_QMOE_INT2_DECODE_MAX_EXPANDED_ROWS=40
export ORT_HC_DECODE_ROWS=1
export ORT_INT2_S4_FUSED_FINALIZE=1
```

When using a custom ORT build from Python, set `ORT_LIB_PATH` to its shared
library; explicit selection takes precedence over the installed Python ORT
package. Invalid explicit paths fail rather than silently selecting another
runtime. Rebuild the package's data-plane library with this runtime.

The combined experimental stack reached 334-342 decode tokens/s on H200 at
2K-32K contexts. These are combined-stack results, not individual-kernel
ablations or measurements of this rebased branch. Paired HumanEval runs scored
154/164 with exact DFlash/control output parity. A five-problem LiveCodeBench
smoke scored 2/5 versus historical MTP 3/5, including a capacity-limited
regression; this is an integration candidate, not an accuracy-qualified default.

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
produces up to four tokens internally; the API returns one per call.

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
