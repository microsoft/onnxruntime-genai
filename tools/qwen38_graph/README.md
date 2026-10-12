# qwen38_flash_graph qualification scripts

Scripts that exercise a package of `model.type = "qwen38_flash_graph"` through the Engine API on a real GPU. The model type,
the package layout, the config schema and the limits are described in [docs/qwen38_flash_graph.md](../../docs/qwen38_flash_graph.md);
the GPU-free tests are `test/cpp/engine/qwen38_graph_*_tests.cpp`.

## Requirements

* Linux, a CUDA GPU, and an `onnxruntime_genai` Python package built with `-DUSE_CUDA=ON -DENABLE_QWEN38_PLANE=ON`
  (`make onnxruntime-genai python qwen38_plane`; the resulting `libqwen38_plane.so` goes to `<package>/libs/`).
* A package with the layout of the documentation (the exporter is not part of this repository), the custom-op libraries it
  lists, and the plugin CUDA execution provider library. Pass the latter with `--provider PATH` or `QWEN38_CUDA_EP_LIBRARY`.
* A UTF-8 text file of natural prose with enough tokens for the longest prompt (for example 8,192 for
  `engine_behaviours.py` and 33,000 for a 32k benchmark). Prompts are cut from it at seed-dependent offsets. Draft acceptance
  depends on the text, so benchmark numbers are only comparable for the same file.

| script | purpose |
| --- | --- |
| `smoke.py` | minimal example: one chat prompt, prints the stream, speed and speculative statistics |
| `engine_gate.py` | token identity: `collect` the Engine's tokens for a fixed prompt set with MTP and with `--no-mtp`, then `compare` the two files (greedy MTP must equal plain greedy decoding). A token file of another greedy stack in the same JSON format can be compared too |
| `engine_behaviours.py` | stop tokens, limits, stop strings, cancel during prefill and decode, a second queued request, continuation, rewind, capability errors, statistics (the GPU counterpart of the C++ executor tests) |
| `bench_engine_mtp.py` | decode throughput at exact prompt lengths: 1 warm-up + `--reps` measured runs on distinct prompts, TPS = (n-1) tokens / time between the first and the last delivered token |

```bash
python engine_gate.py collect $PKG --text-file prose.txt --out mtp.json
python engine_gate.py collect $PKG --text-file prose.txt --out plain.json --no-mtp
python engine_gate.py compare mtp.json plain.json
python bench_engine_mtp.py $PKG --text-file prose.txt --contexts 2048,8192 --out bench.json
```

For fixed-length benchmarks (EOS ignored) use a package copy whose `eos_token_id` holds ids the model does not generate;
configuration overlays append to, rather than replace, the stop-token list. The benchmark reports the finish reason and
whether each run produced `--new` tokens.

`engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'` measures the host cost of one Engine step
with an instantly answering scripted device.
