# ONNX Runtime GenAI Test Folder Structure

This directory contains assets, scripts, and definitions used by the test suite.

## Layout

- `test/audios/*` contains all audio files used for testing.
- `test/images/*` contains all image files used for testing.
- `test/models/model_name/*` contains uploaded assets for `model_name` such as `model.onnx`, `genai_config.json`, `tokenizer.json`, and related files. These are already-created models stored in the repo.
- `test/python/*` contains all Python scripts used for testing.
    - `test/python/builder/*` contains Python scripts used to validate the model builder.
    - `test/python/create/*` contains Python scripts used to create ONNX models. Scripts follow the `create_*.py` naming format.
    - `test/python/integration/*` contains Python scripts used to run integrating testing on Foundry Local models.
    - `test/python/models/*` contains Python scripts used to test ONNX models. Scripts follow the `test_*.py` naming format.
- `test/tool-definitions/*` contains all tool definitions used for testing.

## Parallel preprocessing checks

After building a Release configuration, run the focused C++ suite:

```bash
ctest --test-dir build/Linux/Release -R '^ParallelPreprocessingTests$' --output-on-failure
build/Linux/Release/engine_unit_tests --gtest_filter='Lfm2VlImage*'
```

Timing assertions are excluded from normal CTest correctness runs. On an idle
machine with a known CPU allocation, opt in explicitly:

```bash
build/Linux/Release/parallel_preprocessing_tests --gtest_filter='ParallelPerformanceTests.*'
```

Alternatively configure `ENABLE_PREPROCESSING_PERFORMANCE_TESTS=ON` to register
the separate `ParallelPreprocessingPerformanceTests` CTest entry. These checks
should not be enabled on shared correctness CI workers.

The Qwen, VideoChatFlash, and LFM2 performance tests compare against independent
pre-pool nested loops, not null-pool calls to the implementation being tested.
The reference loops are compiled separately from the timing call sites so that
their dimensions remain runtime inputs. These checks cover both small synchronous
inputs and larger parallel inputs, allow 25% timing noise, and run only in Release
builds on machines reporting at least four hardware threads. Avoid competing CPU
work when running them.

Preprocessing callbacks retain contiguous inner loops and partition patches,
channel planes, or patch rows. Small inputs remain synchronous; LFM2 and
VideoChatFlash's direct small-input paths share
`ThreadPool::kMinimumParallelCost` with the scheduler.
The correctness tests also cover temporal replication, non-square shapes, empty
video inputs, and padded LFM2 images.

Each image performance check also prints a caller-only control for the current
implementation and checks its output against the independent pre-PR loops.
Worker utility tests include inputs above the dispatch threshold, observe a
non-caller thread, propagate an exception raised specifically by a worker, and
reuse that pool successfully. Only worker-backed submissions on the same pool
are serialized; synchronous small jobs may overlap them.

Nemotron streaming mel population retains its pre-pool frame/mel loops for both
FP32 and FP16, including small chunks. Whisper opts out of preprocessing pool
creation and uses the direct sequential tensor-copy path (its FP16 cast is
unchanged). Other processors continue to use the shared pool. Nemotron checks
cover empty, short, and larger chunks and ring-cache wrapping.

Set `ORTGENAI_PREPROCESSING_THREADS` before creating the first multimodal
processor for a model to select 0 through 3 worker threads. `0` disables pool
creation and uses direct sequential copies/loops; the caller is not counted as
a worker. Unset or empty preserves the default of at most three workers.
Invalid nonempty values raise an error. This process-wide setting is sampled
once per model; changing it does not resize already-created pools.
Use `0` or a smaller count for CPU-constrained applications or to avoid
contention with inference. The default still uses the machine-reported hardware
concurrency, not the application's affinity/quota. Isolated preprocessing
speedups do not establish inference throughput under contention.
