# Development guide

This guide explains how to build, test, and lint **ONNX Runtime GenAI** from source. It is written to be quick to skim for humans and unambiguous for AI coding agents: every section lists the exact command to run.

For end-user install instructions (pip / NuGet), see the [README](../README.md) and the [official docs](https://onnxruntime.ai/docs/genai).

For the current continuous-batching architecture and transactional paged KV-cache flow, see [Paged Attention Engine](paged_attention_engine.md). The performance and design history behind the batching implementation is documented separately in [Engine Batching Design](paged_attention_batching.md).

### Memory-backed CPU model loading

The native C/C++ API can load flat CPU text and translation models without opening their model assets by path. Create a config with `OgaCreateConfigFromBuffer` (`OgaConfig::CreateFromBuffer` in C++), register each graph with the existing `OgaConfigAddModelData`, register external weight files with `OgaConfigAddExternalInitializerData`, and supply tokenizer config/vocabulary JSON with `OgaConfigSetTokenizerData`. Then use the normal model, tokenizer and generator APIs. Path-based loading is unchanged.

Config JSON is parsed immediately and tokenizer JSON is copied into shared immutable storage. Graph and external-initializer buffers are borrowed: keep them immutable and alive until every dependent model/session is destroyed, including sessions recreated by decoder-pipeline resets during generation. Pipeline initialization and recreation use the same asset-aware session loader. ORT can borrow eligible initializer slices directly; this does not guarantee zero copies for every tensor.

The existing graph-only `OgaConfigAddModelData` path retains its original lifetime contract: without the memory-backed config API, graph buffers generally need only survive model creation, and pipeline recreation continues loading files. Memory-backed configs instead retain their registered asset source throughout recreation.

These APIs consume the supplied JSON, not a directory snapshot: tokenizer-module and chat-template sidecar files are not discovered. Applications must supply complete supported metadata in the JSON inputs.

This initial contract requires external initializer data and covers main-graph initializers with flat relative asset names. Model packages, subgraph initializers, other execution providers, processors and adapters are not supported. File-backed `shared_initializers` descriptors and custom-ops library paths are rejected, including when added through config overlays. All graph and external-data references must be supplied by the caller; enforce filesystem denial separately when using this API as a sandbox boundary.

`MemoryAssetsTest.*` covers registration errors, owned tokenizer-data lifetime, path/memory parity for configuration-only added tokens, shared-initializer and custom-ops rejection, unchanged path-based metadata validation and deterministic CPU generation using external-weight variants of existing test models. Pipeline coverage runs generation after repeated session resets in file, legacy graph-buffer and memory-asset modes. The legacy graph buffer is invalidated after creation; memory-asset source files are removed after initialization.

Encoder–decoder coverage uses the existing synthetic Marian cache model with separate external-data buffers for both graphs. It checks path/memory generation parity after removing the staged files before memory-backed model creation, repeated generation after releasing the config, and rejection when either session's weight buffer is missing.

Regenerate the tiny GPT-2 variant with `python test/models/hf-internal-testing/tiny-random-gpt2-fp32/export_external_data.py` in an environment containing `onnx`. Generate the pipeline fixture from the existing dummy decoder with the same script plus `--model-dir test/models/engine/dummy-decoder --filename decoder.onnx`.
Generate its file-source adapter fixture with the same script plus
`--model-dir test/models/engine/dummy-decoder --filename decoder.onnx --weights-filename adapter-weights.data --with-adapter`
in an environment containing `onnx`, `numpy`, and `onnxruntime`.

For the Marian variant, run the same script twice with `--model-dir test/models/marian-cache/dynamic --float-initializers-only`: once with `--filename encoder.onnx --weights-filename encoder-weights.data`, then with `--filename decoder.onnx --weights-filename decoder-weights.data`. Integer shape/axis constants remain inline for ORT shape inference.

---

### Cross-platform native file-source loading

`OgaConfigSetModelFileHandles` (`OgaConfig::SetModelFileHandles` in C++)
registers a complete flat model directory as named, pre-opened, read-only files
on a JSON-backed config. Supply tokenizer JSON through `OgaConfigSetTokenizerData`.
Windows, Linux, Android, macOS and iOS use the same loading contract. `OgaFileHandle`
is a native Windows `HANDLE` (`void*`), not a CRT descriptor, or a POSIX `int`
descriptor. Windows handles must permit reading and are duplicated with only read
rights; POSIX descriptors must already be read-only. Original handles can be closed
after successful registration. File handles are not interchangeable between processes:
the host must first transfer/duplicate them into the process calling this API.
GenAI resolves graph names and ORT resolves external-weight names; the host does
not need to parse ONNX tensors. CPU/WebGPU provider selection is unchanged.

Descriptors are duplicated and shared across config copies and session recreation.
Original handles may be closed and paths unlinked after registration. Graph-buffer
overrides, such as decrypted translation graphs, are copied into model-owned
storage during model creation. Tokenizers own their JSON and optional module data;
speech/vision processor JSON and adapters can also come from the file sources.
File-source preprocessing currently supports Whisper and Phi-4 multimodal only;
other model types fail explicitly instead of reopening processor paths.
Files must remain immutable. Metadata must contain a JSON object and no embedded
NUL bytes. Nested names, packages and external tensor attributes are outside this
contract.

Build against an ORT SDK that exposes `GetExperimentalFunction` (API version 28
or newer), and use a runtime with the experimental
`OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32` entry.
Older SDKs can still build the library, but descriptor registration fails explicitly.
An unavailable API, missing file or invalid range is an explicit error, not
filesystem or CPU fallback. ORT maps initializer ranges rather than entire weight
files; small or unaligned tensors may be copied. Full ONNX loading with external
initializers is required, including on mobile; minimal ORT builds do not advertise
the entry. Execution-provider selection is unchanged. GPU providers use ordinary
initializer upload from CPU-backed mappings, not Windows/D3D12 accelerated file
loading. Ordinary path-based loading remains unchanged.
File-source graph buffers are consumed during session creation. The ORT-format
options `session.use_ort_model_bytes_directly=1` and
`session.use_ort_model_bytes_for_initializers=1` are rejected in this mode rather
than retaining pointers into a released mapping. Disabled (`0`) values remain
supported. File-backed and registered graph overrides both undergo normal
decoder state-manifest validation.
Driver initialization must still occur before sandbox lockdown when required.

`FileSourceAssetsTest.*` runs the same native-handle contract tests on desktop
platforms and covers closed/unlinked files, graph and weight
sources for both encoder/decoder sessions, failed registration, truncation,
config-independent ownership, tokenizer-module tokens after config release,
oversized metadata, metadata that could be mistaken for a path, and pipeline session recreation. It explicitly skips
when the selected ORT lacks the API; those skips are not file-source validation.

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
python -m pip install -r requirements-dev.txt
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

The core install prefix, including the native CPack archive, uses this layout:

```text
<prefix>/include/                         # Public C and C++ headers
<prefix>/lib/                             # Native GenAI libraries
<prefix>/lib/cmake/onnxruntime-genai/     # CMake package configuration
```

Use the exported CMake package from a native consumer:

```cmake
find_package(onnxruntime-genai CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE onnxruntime-genai::onnxruntime-genai)
```

Configure the consumer with `-Donnxruntime-genai_DIR=<prefix>/lib/cmake/onnxruntime-genai`. ONNX Runtime is an external runtime dependency and is not included in the core archive.

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

### Native shared-library exports

On Linux and Android, `src/exports.map` restricts `onnxruntime-genai` exports to the public `Oga*` C API. Apple builds use `src/exported-symbols.lst` for the same restriction. C++ consumers use the header-only wrappers over that API; GenAI internals and statically linked dependencies, including curl, remain private to prevent symbol collisions with libraries loaded by the host process. This does not hide symbols in separately shipped shared dependencies. Windows retains its explicit `OGA_EXPORT` exports, and CUDA add-on export policies are unchanged.

ELF links also localize definitions from all static archives with `--exclude-libs,ALL`, including GCC's implicitly linked `libstdc++_nonshared.a` compatibility archive on manylinux. Public APIs must remain in direct or OBJECT sources: placing them in STATIC archives would hide them despite the version script.

With `ENABLE_TESTS=ON` on Linux or macOS, check that every public API declared in `src/ort_genai_c.h` is exported and no other symbols are exported. The ELF checker inspects externally visible definitions only; local symbols left in the dynamic symbol table by GCC toolset compatibility objects are not exports:

```bash
ctest --test-dir build/Linux/Release -R '^SharedLibraryExports$' --output-on-failure
```

The Linux x64 CPU and CUDA CI jobs run both `SharedLibraryExports` and `TelemetryHelpersTests` explicitly, alongside `SharedLibraryArchiveExports` and `SharedLibraryArchiveRuntime`, which check strong/weak archive-symbol isolation and runtime calls through the public fixture API. Windows x64 CPU CI runs `TelemetryHelpersTests`, including redaction and environment-read race regressions. Linux and macOS core release jobs also run the export checker on the installed library against its installed public header, even with `ENABLE_TESTS=OFF`; macOS checks the signed library.

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

Drop the `-k` filter to run the whole file. See [`test/python/README.md`](../test/python/README.md). Provider-specific tests may require additional dependency files under `test/python/<provider>/`.

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

### GitHub Actions pinning

The Clang Format Lint workflow also uses [pinact](https://github.com/suzuki-shunsuke/pinact) to check that GitHub Actions and reusable workflows use full-length commit SHAs with version comments. Pull requests, pushes, and manual runs check pinning without GitHub API calls. After installing pinact v4.0.0, run the same read-only check locally:

```sh
pinact run --fix=false --no-api
```

The full workflow also runs nightly at 06:23 UTC. On scheduled runs, pinact uses authenticated GitHub API calls to verify that version comments match their commits. To reproduce this read-only verification locally, provide a GitHub token via `GITHUB_TOKEN` and run:

```sh
pinact run --fix=false --verify
```

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

### Qwen3.8 operator attributes

The text and MTP builders keep coupled operator settings in canonical dictionaries, shared by their expansion emitters rather than duplicated as loose instance fields:

- `sparse_indexer_attrs`: `num_heads` (HF `indexer_n_heads`), `kv_heads`, `head_dim`, `token_budget` (HF `indexer_budget`), `compress_ratio`, `fixed_cache`, and paged `state_capacity`. These configure the sparse indexer operators and their cache bindings.
- `ple_attrs`: zero-based `layer_ids` (HF `ple_layer_ids` is one-based), `embed_dim`, `conv_kernel_size`, `conv_dilation` (HF `ngram_size`), `ngram_size`, `token_pad_id`, `heads_per_ngram`, and `external`. These configure Engram's n-gram mapping, embedding lookup, and stateful convolution.
- `hyper_connection_attrs`: `num_branches` (HF `hc_count`), expanded `hidden_size`, `tile_first_hidden_state`, and `emit_pre_final_hidden_states`. These configure branchwise normalization, mixing, and hidden-state graph outputs.
- `attention_attrs["output_gate_type"]`: the attention output-gate activation, taken from HF `output_gate_type` or falling back to HF `hidden_act`.

HF configuration field names are unchanged. Compact capture bindings for all three state families use the resolved `context_length_attrs["state_update_capacity"]`; paged CUDA `state_window=8` resolves to capacity 7, including PLE and indexer outputs.

## 11. Troubleshooting

- **Dependency fetch errors during CMake configure** — dependencies are downloaded via `FetchContent` (`cmake/deps.txt`); ensure network access to the declared URLs, then re-run with `--update`.
- **Missing Python build dependency** — run `python -m pip install -r requirements-dev.txt`.
- **Stale CMake cache after switching flags (e.g. adding `--arm64`)** — delete the build directory (or `CMakeCache.txt` inside it) and re-run with `--update`.
- **`onnxruntime.lib` / ORT not found** — pass `--ort_home <path>` to point at a prebuilt ONNX Runtime, or omit it to let `build.py` auto-download a matching build.
- **Clean an existing build** — `python build.py --clean --skip_examples` runs the CMake `clean` target for the selected configuration; it does not rebuild. Run a normal `python build.py` command afterward, or delete the build directory for a completely fresh configure.
