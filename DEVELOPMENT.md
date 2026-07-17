# Development guide

This guide explains how to build, test, and lint **ONNX Runtime GenAI** from source. It is written to be quick to skim for humans and unambiguous for AI coding agents: every section lists the exact command to run.

For end-user install instructions (pip / NuGet), see the [README](README.md) and the [official docs](https://onnxruntime.ai/docs/genai).

---

## 1. Prerequisites

| Tool | Notes |
| ---- | ----- |
| **Python** ≥ 3.10 | Drives `build.py`; also the runtime for the Python wheel. Official wheels target 3.11–3.14. |
| **CMake** ≥ 3.26 | Build-system generator. macOS builds require CMake ≥ 3.28. |
| **C++ compiler** with C++20 support | MSVC (Visual Studio 2022), GCC ≥ 11, or Clang. |
| **Git** | For cloning the repository. |
| **.NET SDK** ≥ 8 | Only for the C# bindings (`--build_csharp`). |
| **JDK** ≥ 8 | Only for the Java bindings (`--build_java`). The repository includes the Gradle wrapper, so a separate Gradle installation is not required. |
| **CUDA Toolkit** | Only for CUDA builds (`--use_cuda`). |

Clone the repository:

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
```

> This repository has **no git submodules**. CMake-managed native dependencies are fetched automatically at configure time (via `FetchContent`, driven by `cmake/deps.txt`), so no submodule initialization is needed. Python, Rust, and Java dependencies use their respective package managers.

Install the common Python build-time dependencies. `pybind11` is required when building the Python SDK incrementally with `--sdk python`:

```bash
python -m pip install requests wheel pybind11
```

---

## 2. Quick start

`build.py` is the single cross-platform entry point. `build.sh` (Linux/macOS) and `build.bat` (Windows) are thin wrappers around it.

Build the CPU native library, Python wheel, C examples, and native tests with the default `RelWithDebInfo` configuration:

```bash
python build.py
```

Common everyday build (RelWithDebInfo, parallel, skip tests and examples for a fast inner loop):

```bash
python build.py --config RelWithDebInfo --parallel --skip_tests --skip_examples
```

The Python wheel is built by default. The build tree is created under `build/<platform>/<config>/` (for example, `build/Windows/RelWithDebInfo/` or `build/Linux/RelWithDebInfo/`). With a multi-config generator such as Visual Studio, native binaries are in an additional configuration subdirectory such as `build/Windows/RelWithDebInfo/RelWithDebInfo/`.

---

## 3. `build.py` phases and key flags

`build.py` runs three phases. If no phase is specified, it requests all three (`--update --build --test`). For `--arm64` and `--arm64ec` cross-compilation, the test phase is disabled after CMake configuration because the target binaries cannot run on the host. If you pass any phase explicitly, only those phases run.

| Phase flag | What it does |
| ---------- | ------------ |
| `--update` | Run CMake to (re)generate the build system. |
| `--build`  | Compile all targets. |
| `--test`   | Run the unit tests. |

Example — regenerate CMake, then build, without running tests:

```bash
python build.py --update --build
```

Example — rebuild only (skip CMake regeneration and tests), useful after editing source files:

```bash
python build.py --build
```

Frequently used flags:

| Flag | Purpose |
| ---- | ------- |
| `--config {Debug,Release,RelWithDebInfo,MinSizeRel}` | Build configuration. Default `RelWithDebInfo`. |
| `--parallel` | Parallel compilation. |
| `--skip_tests` | Skip the test phase. |
| `--skip_wheel` | Do not build the Python wheel. |
| `--skip_examples` | Do not build the sample executables. |
| `--build_dir <path>` | Override the build directory (default `build/<platform>`). |
| `--use_guidance` | Enable constrained / grammar-based decoding (guidance). |
| `--ort_home <path>` | Use a prebuilt ONNX Runtime at this path instead of auto-downloading. |
| `--cmake_extra_defines K=V` | Pass extra `-DK=V` options to CMake. |

Run `python build.py --help` for the complete list.

---

## 4. Execution-provider (hardware) builds

By default the build targets the **CPU** execution provider. Enable hardware acceleration with these flags:

```bash
# CUDA (reads CUDA_HOME / CUDA_PATH, or pass --cuda_home)
python build.py --use_cuda --config RelWithDebInfo

# DirectML (Windows)
python build.py --use_dml --config RelWithDebInfo

# WinML (Windows; bundles CUDA + DirectML). --winml_sdk_version is optional.
python build.py --use_winml --winml_sdk_version 2.1.1 --config RelWithDebInfo
```

---

## 5. Language bindings

The Python wheel is built by default. Enable the others explicitly:

```bash
# C# (requires the .NET SDK)
python build.py --build_csharp --config RelWithDebInfo

# Java (requires a JDK; uses the repository's Gradle wrapper)
python build.py --build_java --config RelWithDebInfo
```

To skip the Python wheel (e.g. when you only want the native library or C#):

```bash
python build.py --skip_wheel --build_csharp
```

---

## 6. Incremental core + SDK builds

The packaging pipeline builds the native **core** once, installs it, then builds each SDK layer (Python / C# / Java) against that prebuilt core. You can use the same flow locally to iterate on a single binding without rebuilding the core.

Step 1 — build and install only the core to a prefix:

```bash
python build.py --config RelWithDebInfo --parallel --skip_tests --skip_examples --skip_wheel --ort_home /path/to/ort --install_dir /path/to/core-install --cmake_extra_defines ENABLE_PYTHON=OFF ENABLE_TESTS=OFF
```

Step 2 — build a single SDK against the installed core (the core is **not** rebuilt):

```bash
python build.py --sdk python --prebuilt_genai_home /path/to/core-install --ort_home /path/to/ort --config RelWithDebInfo
```

`--sdk` accepts `python`, `java`, or `csharp`. Both `--prebuilt_genai_home` and `--ort_home` are required in SDK mode.

---

## 7. Running tests

### C++ unit tests (CTest)

After a build, run the native tests from the build directory:

```bash
ctest --test-dir build/<platform>/<config> --build-config <config> --output-on-failure
```

Example:

```bash
ctest --test-dir build/Windows/RelWithDebInfo --build-config RelWithDebInfo --output-on-failure
```

You can also let `build.py` run them as part of the `--test` phase:

```bash
python build.py --config RelWithDebInfo --test
```

> Note: `build.py` automatically disables the native test phase for `--arm64` and `--arm64ec`. Android and iOS disable CMake unit-test targets during configuration; Android emulator tests are available only for an x86-64 Android build with `--build_java --android_run_emulator`.

### Python tests (pytest)

Install the Python test dependencies and the wheel produced by your build:

```bash
python -m pip install -r test/python/requirements.txt
python -m pip install --force-reinstall <path-to-built-wheel>
```

The Python tests require model files. From the repository root, point pytest at a models directory with `--test_models`:

```bash
python -m pytest -sv test/python/test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models test/models
```

Drop the `-k` filter to run the whole file. See [`test/python/README.md`](test/python/README.md). Provider-specific tests may require additional dependency files under `test/python/<provider>/`.

---

## 8. Linting and formatting

This project uses [lintrunner](https://github.com/suo/lintrunner). Install and initialize once:

```bash
pip install -r requirements-lintrunner.txt
lintrunner init
```

Then, to auto-format your local (changed) files:

```bash
lintrunner -a
```

To format the entire tree:

```bash
lintrunner -a --all-files
```

> **CI gate:** C/C++ formatting is enforced with **clang-format 20.1.0** (`--dry-run -Werror`). Any unformatted changed C/C++ file fails the PR, so run `lintrunner -a` before pushing.

---

## 9. Cross-compilation

```bash
# Windows ARM64 (cross-compiled on an x64 host)
python build.py --arm64 --config RelWithDebInfo --skip_tests

# Android (requires ANDROID_HOME / ANDROID_NDK_HOME)
python build.py --android --android_abi arm64-v8a --config RelWithDebInfo --skip_tests

# iOS simulator framework (build on macOS)
python build.py --ios --build_apple_framework --apple_sysroot iphonesimulator --osx_arch arm64 --apple_deploy_target 15.4 --cmake_generator Xcode --config RelWithDebInfo --skip_tests --skip_wheel
```

For a physical iOS device, use `--apple_sysroot iphoneos`. Cross-compiled targets cannot run native CTest binaries on the build host, so pass `--skip_tests` explicitly unless you are using the supported Android emulator-test flow.

---

## 10. Repository layout

| Path | Contents |
| ---- | -------- |
| `src/` | Core C++ implementation (models, engine, generators, search/sampling, KV cache). |
| `src/python/` | Python binding (pybind11) + wheel packaging. |
| `src/csharp/` | C# binding. |
| `src/java/` | Java binding (Gradle). |
| `src/objectivec/` | Objective-C binding. |
| `cmake/` | CMake modules (options, ORT resolution, packaging, telemetry) and `deps.txt` (fetched dependencies). |
| `test/` | C++ (`test/*.cpp`) and Python (`test/python/`) tests; test models in `test/models/`. |
| `examples/` | Runnable samples (C, C#, Python). |
| `.pipelines/` | Azure DevOps packaging pipeline (stages, jobs, steps). |
| `build.py` | Cross-platform build driver. |
| `VERSION_INFO` | Package version (e.g. `0.15.0-dev`). |

---

## 11. Troubleshooting

- **Dependency fetch errors during CMake configure** — dependencies are downloaded via `FetchContent` (`cmake/deps.txt`); ensure network access to the declared URLs, then re-run with `--update`.
- **Missing Python build dependency** — run `python -m pip install requests wheel pybind11`.
- **Stale CMake cache after switching flags (e.g. adding `--arm64`)** — delete the build directory (or `CMakeCache.txt` inside it) and re-run with `--update`.
- **`onnxruntime.lib` / ORT not found** — pass `--ort_home <path>` to point at a prebuilt ONNX Runtime, or omit it to let `build.py` auto-download a matching build.
- **Clean an existing build** — `python build.py --clean --skip_examples` runs the CMake `clean` target for the selected configuration; it does not rebuild. Run a normal `python build.py` command afterward, or delete the build directory for a completely fresh configure.
