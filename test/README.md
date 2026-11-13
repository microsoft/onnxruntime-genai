# ONNX Runtime GenAI Test Folder Structure

This directory contains assets, scripts, and definitions used by the test suite.

## Layout

```text
test/
  audios/                         # Audio fixtures
  images/                         # Image fixtures
  models/                         # Checked-in ONNX models and model configurations
    <model_name>/                 # model.onnx, genai_config.json, tokenizers, etc.
      <variant>/                  # Optional precision/device/case variants
      external-data/              # External model weights, where needed
    engine/<model_name>/          # Engine-specific model fixtures
    hf-internal-testing/<model_name>/ # Tiny model fixtures
  cmake/                          # CMake/build regression checks
    export-isolation/             # Shared-library export fixtures
    telemetry-dependencies/       # Telemetry dependency configuration checks
  cpp/                            # C++ API and runtime tests
    cuda/                         # CUDA sampling/top-k tests and benchmarks
    cuda_batched_sampler_tests/   # Batched CUDA sampler tests
    cuda_kernel/                  # CUDA kernel tests
    engine/                       # Internal Engine component tests
      custom_ops_test_lib/        # Custom-operator library fixture
      qwen38_legacy_plane_lib/     # Legacy-plane library fixture
    lilicorr_drafter_tests/        # LiliCorr drafter tests
    telemetry/                    # Telemetry tests and native fixtures
  csharp/                         # C# binding tests
  platform/
    apple/apple_package_test/     # Apple package test project
      apple_package_test.xcodeproj/ # Xcode project/workspace configuration
      ios_package_test/           # iOS test app (Base.lproj/ resources)
      ios_package_testUITests/    # iOS UI tests
      macos_package_test/         # macOS test app (Base.lproj/ resources)
  python/
    builder/                      # Model-builder/export/quantization tests
    create/                       # ONNX fixture creation scripts (create_*.py)
    engine/                       # All Engine tests and Engine test runners
    integration/                  # Cross-component/runtime/example/packaging tests
    models/                       # Model-specific behavior and loader tests
    cpu/{ort,torch}/              # CPU dependency requirements
    cuda/{ort,ort-nightly,torch}/  # CUDA dependency requirements
    directml/{ort,torch}/          # DirectML dependency requirements
    macos/{ort,torch}/             # macOS dependency requirements
    webgpu/{ort,torch}/            # WebGPU dependency requirements
  tool-definitions/               # Tool-calling fixtures
```

Keep model assets, including configuration-only fixtures such as
`test/models/nemotron-parse/`, under `test/models/`, not a separate configs
folder. Model-specific nested directories hold their variants and assets.

Only `README.md`, `_test_utils.py`, `conftest.py`, `requirements.txt`,
`special_tokens.py`, `test_onnxruntime_genai.py`,
`test_onnxruntime_genai_api.py`, and `test_onnxruntime_genai_e2e.py` belong
directly under `test/python/`. Store all other Python files in the appropriate
subfolder above. Engine tests belong in `engine/` even when they use real-model
integration; non-Engine Foundry Local tests and shared model-resolution helpers
belong in `integration/`. Keep category-specific helpers beside their tests.

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

LFM2's RGB path keeps the three channels explicit for vectorization while
preserving the generic-channel fallback and padded source strides. Keep the
64/128/512-square old-loop, null-pool, and pooled benchmark controls when
changing this kernel.

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
cover empty, short, and larger chunks and ring-cache wrapping. The cache-update
test checks literal frame values after each update, including a write across
the ring boundary, and builds FP32/FP16 expectations independently of the cache.

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

## Telemetry shutdown regression

Native desktop builds with `ENABLE_TESTS=ON` and `ENABLE_TELEMETRY=ON` include
`telemetry_shutdown_tests` when building 1DS from source. When using an installed static
`MSTelemetry` package, also set `TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR` to the exact source tree used
to build that package. Private SDK instrumentation must match the linked archive, including the
SDK version-lifetime fix. Source builds compile a patched PAL with process-lifetime version storage;
installed package providers must supply the equivalent fix.

Run the isolated processes from the configured build directory:

```sh
ctest -C Release -R '^TelemetryShutdown-' --output-on-failure --repeat until-fail:50
```

The cases exercise the actual singleton, initialization on an application thread, explicit
shutdown, shutdown/reinitialization, constructor-only priming, full runtime suppression, and CI
suppression, and injected mutex/allocation failures during priming. Failed priming permanently
suppresses telemetry, including after ordinary shutdown/reinitialization. Test-only SDK objects
observe real mutex initialization and native worker joins, including in the no-worker modes;
the lifetime observer also makes the destruction-order defect deterministic on GCC/MSVC.
The upload-at-exit cases, including after shutdown/reinitialization, pass real events through
decoration, queuing, storage, and encoding. Encoding is held until process exit to exercise late
SDK-version reads; a successful in-process HTTP responder prevents any collector transmission.
Test-only observers detect version-string access after destruction, including with the original
SDK used as a negative control. Each case uses a private cache directory. Missing SDK
initialization fails enabled cases rather than silently passing.

The constructor primes only an empty diagnostic source before singleton destruction is registered.
SDK initialization remains lazy; priming creates no uploader or collector event. A priming
exception latches the existing full suppression state rather than retrying mutex initialization
after singleton construction.

These regressions link GenAI into an executable. They cover ordinary static destruction and explicit
shutdown, not Windows DLL process detach or explicit `FreeLibrary`. Production Windows process
termination skips uploader teardown before taking telemetry locks; normal shutdown and explicit
unload retain cleanup.

Production `MAX_TEARDOWN_TIME` remains zero: exit is not extended to upload pending events.
Events already persisted in the offline cache can be uploaded in a later GenAI session using
that cache. Abrupt process termination does not guarantee persistence of events still in memory.
