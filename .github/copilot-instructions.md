# ONNX Runtime GenAI - AI Coding Agent Instructions

## Architecture Overview

This is **ONNX Runtime GenAI**, a high-performance inference library for generative AI models. The codebase implements the complete generative AI loop including preprocessing, ONNX Runtime inference, logits processing, search/sampling, and KV cache management.

### Core Components

- **`src/models/`** - Model implementations with support for LLMs, VLMs (Vision), ALMs (Audio), and Pipeline models
- **`src/engine/`** - Request batching engine for concurrent model execution with dynamic scheduling  
- **`src/generators.h`** - Central generator logic coordinating the full inference pipeline
- **`src/ort_genai.h`** - Zero-cost C++ wrapper around the C API for automatic resource management
- **Language bindings**: Python (`src/python/`), C# (`src/csharp/`), Java (`src/java/`), Objective-C (`src/objectivec/`)

### Key Abstractions

```cpp
// Core inference flow: Model -> Generator -> Tokenizer
auto model = OgaModel::Create("phi-2");
auto tokenizer = OgaTokenizer::Create(*model);
auto generator = OgaGenerator::Create(*model, params);
```

The `State` class hierarchy in `src/models/model.h` handles device-specific execution, while the `Engine` class in `src/engine/` manages request batching and scheduling.

## Build System & Development Workflow

### Primary Build Commands

```bash
# Cross-platform Python build script (preferred)
python build.py --config Release --use_cuda --build_java --enable_tests

# Platform-specific scripts
build.bat         # Windows batch
build.sh          # Linux/Mac shell
```

### Key Build Options (cmake/options.cmake)

- `USE_CUDA/USE_DML/USE_ROCM` - Hardware acceleration backends
- `USE_WINML` - Windows ML integration requiring `WINML_SDK_VERSION` parameter
- `ENABLE_JAVA/ENABLE_PYTHON` - Language binding compilation
- `USE_GUIDANCE` - Constrained generation support

### WinML Build Pattern

WinML builds require explicit SDK version specification:

```bash
# WinML build - WINML_SDK_VERSION is mandatory
python build.py --use_winml -DWINML_SDK_VERSION=1.8.2084
```

WinML integration downloads `Microsoft.WindowsAppSDK.ML` via NuGet and copies headers/libs to a local `ort/` directory.

### Testing

```bash
# Python tests with test models
python -m pytest -sv test_onnxruntime_genai_api.py -k "test_name" --test_models ..\test_models

# C++ unit tests via CMake/CTest
ctest --build-config Release --output-on-failure
```

## Code Patterns & Conventions

### Device Interface Pattern

Each hardware backend implements `DeviceInterface` (defined in `src/smartptrs.h`):

```cpp
struct CudaInterface : DeviceInterface {
  std::unique_ptr<DeviceBuffer> Allocate(size_t size) override;
  void CopyToDevice(DeviceSpan<T> dst, std::span<const T> src) override;
};
```

### Model State Management

Models follow the `State` pattern where each model type extends the base `State` class:

```cpp
struct State {
  virtual DeviceSpan<float> Run(int total_length, 
                               DeviceSpan<int32_t>& next_tokens) = 0;
  virtual void RewindTo(size_t index) {}  // For session continuation
};
```

### Error Handling Convention

Use `OgaCheckResult()` wrapper for C API error propagation:

```cpp
OgaCheckResult(OgaCreateModel(model_path, &model));  // Throws std::runtime_error
```

### Memory Management

- **DeviceSpan/DeviceBuffer**: Device-agnostic memory abstractions
- **std::unique_ptr with custom deleters**: For C API resource cleanup
- **LeakChecked<T>**: Debug-mode leak detection for core types

## Critical Integration Points

### ONNX Runtime Dependency Management

ADO pipelines obtain ORT lib/headers via three methods:
1. **Explicit `ORT_HOME`** - Pipeline provides pre-built ORT artifacts (preferred)
2. **Auto-download via CMake** - `cmake/ortlib.cmake` fetches from ORT-Nightly feed when `ORT_HOME` unset
3. **Python build driver** - `tools/python/util/dependency_resolver.py` downloads NuGet packages

### Model Loading Pipeline

1. **Config parsing** (`src/config.cpp`) - Reads `genai_config.json` model metadata
2. **ONNX session creation** via `onnxruntime_api.h` wrappers
3. **Device interface selection** based on provider availability
4. **KV cache initialization** (`src/models/kv_cache.cpp`) for transformer models

### Multi-Modal Support

Vision models (Phi-Vision) use separate processor classes:
- `PhiImageProcessor` - Image tokenization and preprocessing
- `MultiModalProcessor` - Coordinates text/image inputs

### Execution Provider Detection

Hardware acceleration auto-detection follows this priority:
1. CUDA (if `USE_CUDA=ON` and CUDA runtime available)
2. DirectML (Windows, if `USE_DML=ON`)
3. CPU fallback

## Project-Specific Gotchas

### Windows-Specific Build Requirements

- **Visual Studio 2022** required for C++20 features
- **WinML integration** requires specific NuGet package versions (see `cmake/nuget.cmake`)
- **Cross-compilation** for ARM64/ARM64EC supported via CMake platform flags

### Model Compatibility Matrix

The repo supports specific model architectures - check `src/models/model_type.h` for the canonical list. New models require:
1. Config template in model directory
2. State implementation extending base `State` class
3. Optional custom processors for multi-modal inputs

### Performance Considerations

- **KV caching** is automatically managed but can be configured via `runtime_settings.cpp`
- **Continuous decoding** (session continuation) requires careful state management
- **Multi-LoRA** adapters use separate weight loading in `src/models/adapters.cpp`

## Testing Strategy

Tests are organized by language binding:
- **C++ tests**: `test/` directory, focused on core API validation
- **Python tests**: `test/python/`, includes end-to-end model testing
- **Platform tests**: Android/iOS tests run via emulator/simulator

Always test with actual model files from `test/test_models/` directory rather than mock data.
