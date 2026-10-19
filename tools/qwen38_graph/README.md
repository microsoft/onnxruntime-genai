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

## Package construction and graph optimization

The package tools are:

- `build_package.py`: separate verify/head graph package.
- `build_package_v12.py`: fold-decode package with optional dense prefill.
- `build_package_v12b.py`: shared-weight fold package.
- `optimize_ported_package.py`: convert supported exported graph chains into
  generic ORT contrib operators and construct an isolated optimized package.

Pass explicit source bundles, tokenizer/checkpoint paths, data-plane library,
and runtime/provider library paths for your environment. Legacy research
defaults are not portable. Run each tool with `--help` to inspect its arguments.

The optimizer performs guarded graph rewrites for normalization, rotary
embedding, GDN preparation, gates, hyper-connections, FP8 projections, MoE
residuals, and sparse-attention cache handoff. It preserves external weight
sharing, writes package metadata, and does not mutate a qualified input
package's hard-linked files in place. Runtime schemas must match those used
by the companion ORT operator stack.

## Tests

Run the graph-rewrite unit tests without downloading model weights:

```bash
python -m unittest discover -s tools/qwen38_graph -p test_optimize_ported_package.py
```

`test_api.py` checks the Generator API against a supplied compatible model
package. `test_plane_kernels.py` checks the CUDA data-plane library.
`test_prompts.py` checks supplied prompt/reference cases. `smoke.py` and
`smoke_cpp.cpp` provide minimal integration examples. These integration checks
require compatible model artifacts and runtime libraries; they are not
standalone CPU unit tests.

The local qualified v26 runtime retained exact reference token IDs and showed
small decode gains at 8K, 16K, and 32K context lengths. Those measurements came
from the combined local operator stack, not from independent builds of every
draft PR against upstream main. Each draft remains subject to its own CI,
dependency resolution, and base-specific validation before review.
