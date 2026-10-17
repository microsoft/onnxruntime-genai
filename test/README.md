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
