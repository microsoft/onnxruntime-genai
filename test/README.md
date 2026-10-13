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

## Telemetry shutdown regression

Native desktop builds with `ENABLE_TESTS=ON` and `ENABLE_TELEMETRY=ON` include
`telemetry_shutdown_tests` when building 1DS from source. When using an installed static
`MSTelemetry` package, also set `TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR` to the exact source tree used
to build that package. Private SDK instrumentation must match the linked archive.

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
Collector events are intercepted before queuing, and each case uses a private cache directory.
Missing SDK initialization fails enabled cases rather than silently passing.

The constructor primes only an empty diagnostic source before singleton destruction is registered.
SDK initialization remains lazy; priming creates no uploader or collector event. No process-exit
callback or shutdown-state machinery is added. A priming exception latches the existing full
suppression state rather than retrying mutex initialization after singleton construction.
