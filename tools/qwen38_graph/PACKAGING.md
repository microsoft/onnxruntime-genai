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

The optimizer rewrites supported normalization, rotary, GDN preparation,
gates, hyper-connections, FP8 projection, MoE residual, and sparse-attention
cache handoff chains. It preserves external weight sharing and avoids
mutating qualified input packages through hard-linked graph files.
Generated graph schemas must match the companion ORT operators:
normalization/rotary, GDN/convolution, quantized MoE/FP8, hyper-connections,
packed sparse-attention/indexer, and speculative fold/state operations.

## CPU unit tests

With ONNX, NumPy, and ml-dtypes available, run:

```bash
python -m unittest discover -s tools/qwen38_graph -p test_optimize_ported_package.py
```

Tests construct synthetic graphs and need no model weights or GPU.
Unit tests and Python static/lint checks do not qualify complete generated
packages, native runtime compatibility, model accuracy, or GPU performance.
No benchmark/profiling/evaluation drivers, weights, or task metadata are
included.
