# Qwen graph execution through Generator

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

## Validation

`test_api.py` exercises a supplied compatible model package.
`test_plane_kernels.py` checks a supplied CUDA data-plane library.
`test_prompts.py` checks supplied prompt/reference cases.
`smoke.py` and `smoke_cpp.cpp` provide integration examples.
These tests require compatible model artifacts and native libraries; they
are not standalone CPU unit tests. Unit/static checks do not replace a native
rebuild and GPU/model qualification of this exact main-based branch.

Benchmark/profiling/evaluation drivers, weights, inherited model builders,
and local task metadata are intentionally excluded.
