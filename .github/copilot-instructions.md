# ONNX Runtime GenAI - Copilot Instructions

## Build, test, and lint

`build.py` is the cross-platform entry point; `build.bat` and `build.sh` are thin wrappers. It configures, builds, and tests by default, builds a Python wheel, and writes to `build/<platform>/<config>/`. Native dependencies are fetched by CMake from `cmake/deps.txt`; this repository has no git submodules.

```bash
# Install common build dependencies.
python -m pip install -r requirements-dev.txt

# Default CPU build: native library, Python wheel, C examples, and native tests.
python build.py

# Fast native/Python inner loop. RelWithDebInfo is the default configuration.
python build.py --config RelWithDebInfo --parallel --skip_tests --skip_examples

# Rebuild an already configured tree without rerunning CMake or tests.
python build.py --build

# Provider and binding builds.
python build.py --use_cuda --config RelWithDebInfo
python build.py --use_dml --config RelWithDebInfo
python build.py --use_winml --winml_sdk_version 2.1.1 --config RelWithDebInfo
python build.py --build_csharp --config RelWithDebInfo
python build.py --build_java --config RelWithDebInfo
```

CPU is the default provider. Use `--ort_home <path>` to build against a prebuilt ONNX Runtime instead of resolving one automatically. Put all CMake options in `cmake/options.cmake`; pass exceptional overrides through `--cmake_extra_defines K=V`. Run `python build.py --help` for platform, cross-compilation, and packaging flags.

After building, use the matching platform/configuration directory:

```bash
# All registered native test executables.
ctest --test-dir build/<platform>/<config> --build-config <config> --output-on-failure

# One registered native suite (UnitTests, ReInitTests, EngineUnitTests, etc.).
ctest --test-dir build/<platform>/<config> --build-config <config> -R "^EngineUnitTests$" --output-on-failure

# Python test dependencies and one Python test selection.
python -m pip install -r test/python/requirements.txt
python -m pytest -sv test/python/test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models test/models
```

Python API tests need a built wheel installed and real model assets supplied through `--test_models`; provider-specific tests can have additional requirements under `test/python/<provider>/`. `build.py` disables native execution for ARM64/ARM64EC cross-builds, and Android/iOS do not use the normal host CTest path.

Linting is driven by `.lintrunner.toml`. Ruff/Ruff Format cover Python except excluded paths (notably `src/python/py/models/**`), and clang-format covers C/C++/CUDA/Objective-C. CI requires clang-format 20.1.0.

```bash
pip install -r requirements-lintrunner.txt
lintrunner init
lintrunner                  # Check changed files against origin/main.
lintrunner -a               # Auto-fix changed files.
lintrunner --all-files      # Check the entire tree.
```

## Architecture

### API and generation stack

- `src/ort_genai_c.h` is the stable ABI boundary: opaque `Oga*` handles, explicit create/destroy functions, and `OgaResult*` errors. `src/ort_genai.h` is a zero-cost C++ RAII layer over that API; it converts failures with `OgaCheckResult()` and owns handles through custom deletion. Keep internal C++ types behind the C ABI.
- A `Config` resolves `genai_config.json`, model/tokenizer paths, execution providers, token IDs, and canonical graph input/output names. These configured tensor names and `SessionInfo` validation are the contract between exported ONNX models and runtime state; do not hard-code model I/O names in generator logic.
- `Model` owns configuration, sessions, provider/device assignments, and creates a model-specific `State`. Decoder-only state composes token/position inputs, logits, KV/recurrent state, and optional hidden-state I/O around an ONNX Runtime session.
- `Generator` creates device-specific `Search` state and model `State`. Generation advances through a strategy (`GenerateNextToken()` delegates to `strategy_->Step`), allowing ordinary, beam, speculative, and other flows to share lifecycle and state machinery.
- `Search` is both the sampling policy and sequence owner: it holds sequences, lengths, next-token/index buffers, logits, EOS state, and checkpoint/rewind hooks. Beam search is selected for `num_beams > 1`; otherwise the selected `DeviceInterface` creates greedy/sampling search.
- Tokenization is model-adjacent but independently constructible from a model, config, or path. It supports scalar/batch encode/decode and streaming decode; do not couple tokenizer lifetime to generation state.

### Execution providers and memory

Provider implementations live behind `DeviceInterface`; generation code should use `DeviceSpan`/`DeviceBuffer` and interface operations rather than call CUDA, DirectML, or CPU implementations directly. A model may assign execution, inputs, logits, scoring, and KV cache to different devices. In particular, CUDA/TRT-RTX can score on-device while other providers score on CPU, and input placement changes for providers such as WebGPU and graph capture.

CUDA sources build into a separately loaded `onnxruntime-genai-cuda` library. The main core is normally compiled as an object target and linked into the public shared library so white-box tests can reuse internal objects; Apple framework/Xcode builds are the exception.

### Continuous-batching Engine

`src/engine/` is a separate request-oriented execution path, not an extension of `Generator`. `EngineDependencies` composes a cache manager, scheduler, model executor, optional draft/MTP components, and sampler state. Static and dynamic schedulers share the same Engine surface; `engine.dynamic_batching` selects continuous batching, paged cache management, and variable-length decoder I/O.

Dynamic steps are transactional before event publication: plan a batch, reserve paged/fixed state, checkpoint request/search/sampler state, pack and execute once, stage samples, then commit state and publish events. Recoverable pre-publication failures restore checkpoints and reservations; unexpected publication failures make the Engine permanently unhealthy rather than exposing partial state. Preserve this boundary when changing scheduling, cache ownership, sampling, or request bookkeeping.

An Engine has no worker thread. One owner thread performs Engine and Request operations and `Run()` makes synchronous progress while returning event records. Each Engine request represents one sequence (`batch_size == 1`, `num_beams == 1`); throughput comes from batching requests. Completed requests remain resident for continuation and consume cache/batch capacity until `Close()`. Keep `docs/paged_attention_engine.md` current when changing admission, scheduling, cache ownership, packed I/O, transactions, request lifecycle, or failure handling.

### Language bindings and model builder

Python, Java, C#, and Objective-C bindings are thin adapters over native handles. Preserve explicit native ownership/destruction and C-API error translation; avoid reimplementing core generation behavior in a binding. Python maps contiguous NumPy memory to native spans, while Java JNI transports handles as `jlong`.

`src/python/py/models/` exports ONNX graphs and `genai_config.json`; it is not runtime inference code. Builder output must preserve the tensor names, shapes, cache metadata, and model configuration consumed by the C++ runtime.

## Repository-specific conventions

- When adding or changing a public operation, keep the C ABI, C++ RAII wrapper, relevant language bindings, and tests synchronized. Preserve opaque-handle ownership and return errors through `OgaResult*`/binding translation.
- Use configured model I/O names and provider/device abstractions. Model-family differences belong in configuration, model/state implementations, decoder I/O, or processors—not scattered conditionals in generic generation code.
- State that participates in speculative decoding or Engine transactions must implement matching checkpoint, commit, and rewind/restore behavior. Do not update only logical sequences while leaving KV, recurrent, sampler, or request bookkeeping state unhandled.
- CMake options are centralized in `cmake/options.cmake`, source collections in `cmake/global_variables.cmake`, ORT resolution in `cmake/ortlib.cmake`, and fetched dependency versions in `cmake/deps.txt`.
- Native `unit_tests` deliberately exercise the public shared-library API. Tests requiring internal symbols belong in dedicated white-box executables linked to `onnxruntime-genai-obj`, following `reinit_tests` and `engine_unit_tests`.
- For changes under `src/python/py/models/**`, first read `.github/instructions/python-model-builder.instructions.md` plus the linked `README.md`, `DESIGN.md`, `loaders/LOADERS.md`, and `quantization/QUANTIZATION.md`. Those scoped rules take precedence, and this subtree is excluded from the repository Ruff/lintrunner configuration.
