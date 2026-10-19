# Qwen graph execution through Generator

The `qwen38_flash_graph` model type exposes fixed-shape CUDA graph execution
through the normal `Model`, `Tokenizer`, and `Generator` interfaces. It supports
greedy MTP(1), fold decode, dense chunked prefill, shared weights, and explicit
handoff of recurrent and attention state.

## Build and dependencies

Build GenAI with CUDA enabled, using an ORT build that provides the required
CUDA plugin and contrib operators. Build the `qwen38_plane` target alongside
GenAI. This integration depends on the companion generic CUDA operator stack;
it is not a standalone model exporter.

Runtime configuration uses `model.type = "qwen38_flash_graph"` and the
`model.qwen38_graph` settings declared in `src/config.h`. Configure the target,
MTP head, optional dense-prefill and KV-prefill graph paths, data-plane library,
and PLE table inside the model package. Provider registration and session
options follow the existing GenAI model configuration conventions.

## API

Use the regular Generator lifecycle: create the model and tokenizer, set
greedy search options, append prompt tokens, and call `generate_next_token`
until `is_done`. An accepted draft produces two tokens internally; the API
returns one token per call and buffers the second token.

## Limits

- Batch size one and greedy decoding only.
- One active generator per model.
- Prompt chunks may be appended before generation begins. Appending after
  generation begins is rejected.
- `rewind_to(0)` resets the state. Rewinding to a nonzero position is rejected.
- Graph cache capacities bound prompt plus output lengths.
- The configuration controls stopping tokens; do not change production EOS
  settings to reproduce fixed-length performance measurements.

INT8 prefill/decode caches use their actual element sizes during handoff, and
inconsistent source/destination types raise errors. Session-created allocators
remain owned by the session; shared EP allocators remain owned by the ORT
environment.

## Validation

`test_api.py` and `test_plane_kernels.py` supply API and data-plane tests.
`test_prompts.py` supplies prompt checks, and `smoke.py` and `smoke_cpp.cpp`
provide integration examples. Existing package
qualification checks sampled token IDs against the prior greedy reference and
checks default-prompt output sequences. These checks are model-specific;
they do not establish compatibility with every other model.

Benchmark runners, profiling utilities, evaluation drivers, model weights,
and local task metadata are intentionally excluded from this PR stack.
