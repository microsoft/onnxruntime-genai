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

## Qwen position-input component tests

`RopeRows/QwenPositionValuesTest.*` in `engine_unit_tests` checks exact image-prefill
and successive decode position tensors for three-row and four-row Qwen mRoPE inputs.
The four-row fixture is a tiny identity graph used only for real session metadata,
not language-model inference. Regenerate it from the repository root with:

```sh
python test/python/create/create_qwen_position_inputs.py
```

The existing three-row Qwen3-VL fixture is retained as a control. These tests do not
establish numerical vision-embedding or end-to-end generation parity.

## Qwen partitioned host-vision regression

Run the CPU host-runtime regression from `test/python`:

```sh
python -m pytest -q --test_models ../models models/test_qwen_fara_models.py -k qwen3_5_openvino_partitioned
```

The numerical cases generate small ONNX stages and PNG images in pytest temporary
directories automatically. They use the actual partitioned Qwen3.5 runtime with
independent NumPy references for interpolation, spatial-merge order, rotary values,
frame masks, image-feature concatenation and host injection. A diagnostic merger
preserves its complete inputs in the features so elementwise embedding comparisons
observe each host calculation. Non-square, unequal packed/padded and temporal grids,
real image preprocessing, and image prefill followed by text decoding are covered.
Embedding input IDs are tested with both int32 and int64 graph declarations.
Vision dtype cases cross float32/float16 pixel sources with individually typed
patch and merger feeds and stage outputs, including mixed-type pipelines.
Host math stays float32; numerical expectations account for rounding at each
float16 boundary. Prefill and two text decode steps cover unequal images, with
an additional temporal-mask control for all-float16 boundaries. Diagnostic graphs
cast boundary inputs to float32 internally, so these do not test learned float16 kernels.
The CPU cast suite checks infinity/NaN preservation in both scalar directions,
including the negative infinities used by temporal attention masks.
OpenVINO-partitioned Qwen image grids accept the configured input name when the
canonical `image_grid_thw` key is absent. When both are supplied, the canonical
tensor takes priority for both vision and position inputs, regardless of insertion order.
Profiled dtype cases require non-int64 `image_grid_thw` to be rejected before any
vision stage executes; an int64 control verifies all three stages execute normally.
ONNX and Pillow are required to execute all cases; the existing Qwen3.5 tokenizer
and processor assets supply the fixture metadata.

These are synthetic host-math and wiring tests, not pretrained-model feature parity,
learned attention validation, public video support, or OpenVINO EP/device validation.

Partitioned vision mergers are released after prompt injection; these regressions
check output behavior across decoding, not memory usage or allocator reclamation.

## Gemma3 partitioned image-injection regression

Run the CPU runtime regression from `test/python`:

```sh
python -m pytest -q --test_models ../models models/test_decoder_state_input_ids.py -k gemma3_partitioned
```

The cases generate tiny ONNX vision, text-embedding and decoder graphs in pytest
temporary directories. Direct tensor inputs check that the host merger replaces
only image rows, while a diagnostic decoder exposes all received embeddings and
token-type markers in its raw logits. Image/text-only prefill and two successive
decode forwards cover int32/int64 embedding IDs, all int32/int64 source/decoder
marker combinations, and a configured decoder input name.
Vision input/output dtype cases independently cross float32 and float16 with both
pixel source types, checking injected rows, markers and successive decode steps.
Native `TokenTypeIdsLifecycleTest` cases cover int32/int64 decoder metadata and
batch sizes one/two. Unchanged sequence lengths reuse the bound marker tensor,
but every update still applies staged markers once or clears the buffer to text
zeros; shape changes and invalid pending-marker lengths are also checked.
Image cases also check full-prefill fallback when chunking is requested.
Session profiles verify invalid marker dtypes and out-of-int32-range values are
rejected before vision executes, while valid markers execute vision once.
Missing/empty image markers retain their existing downstream error after vision.
Partitioned Gemma3 requires a single vision graph; nonempty `vision.pipeline`
configurations are rejected during model loading.
ONNX and the existing `multimodal-decoder-no-input-ids` tokenizer assets are required.
Cases skip if the optional model directory is absent; an existing directory with
missing tokenizer assets still fails explicitly.

These are synthetic runtime-wiring tests, not image-preprocessing, pretrained-model
parity, chunked image-prefill support, or OpenVINO EP/device validation.

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
