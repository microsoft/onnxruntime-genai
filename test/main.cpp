// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include "ort_genai.h"

// Global variable to store custom model base path
std::string g_custom_model_path;
// Set to true when the WebGPU plugin EP is successfully registered via --ep_dir.
bool g_webgpu_ep_registered = false;
// Set to true when the CUDA plugin EP is successfully registered via --ep_dir.
bool g_cuda_ep_registered = false;

namespace {

namespace fs = std::filesystem;

// Platform-specific file-name prefix/suffix of an ONNX Runtime execution provider plugin library,
// e.g. "onnxruntime_providers_webgpu.dll" or "libonnxruntime_providers_webgpu.so".
#if defined(_WIN32)
constexpr const char* kProviderPrefix = "";
constexpr const char* kProviderSuffix = ".dll";
#elif defined(__APPLE__)
constexpr const char* kProviderPrefix = "lib";
constexpr const char* kProviderSuffix = ".dylib";
#else
constexpr const char* kProviderPrefix = "lib";
constexpr const char* kProviderSuffix = ".so";
#endif

// Execution provider plugin libraries to register at test time. Each entry pairs an (arbitrary)
// registration name passed to OgaRegisterExecutionProviderLibrary with the platform-independent
// stem of the library file. The registration name is just a handle -- it is deliberately NOT an EP
// name (we use a ".GenAI" suffix to make that obvious). The actual EP name(s) are defined by the
// plugin EP implementation (e.g. "WebGpuExecutionProvider", "CudaPluginExecutionProvider") and are
// what appear in OrtEpDevice / what genai matches on. The full file name is built as
// "<prefix><stem><suffix>", e.g. on Windows "onnxruntime_providers_webgpu" ->
// "onnxruntime_providers_webgpu.dll".
// A single EP may ship under more than one library file name across ORT versions; list each
// candidate (same registration name) and RegisterEpLibrariesFromDirectory registers whichever is
// present. CUDA currently ships as either onnxruntime_providers_cuda or
// onnxruntime_providers_cuda_plugin depending on ORT version.
constexpr std::array<std::pair<std::string_view, std::string_view>, 3> kPluginEpLibraries = {{
    {"WebGPU.GenAI", "onnxruntime_providers_webgpu"},
    {"CUDA.GenAI", "onnxruntime_providers_cuda"},
    {"CUDA.GenAI", "onnxruntime_providers_cuda_plugin"},
}};

// Builds the platform-specific plugin library file name for an EP library stem.
std::string EpLibraryFileName(std::string_view stem) {
  return std::string(kProviderPrefix) + std::string(stem) + kProviderSuffix;
}

// Registers an execution provider plugin library with ONNX Runtime, logging the outcome.
// Returns true on success. Registration failures are reported but do not abort the test run.
bool RegisterEpLibrary(const std::string& registration_name, const std::string& library_path) {
  try {
    std::cout << "Registering execution provider library '" << registration_name << "' -> "
              << library_path << std::endl;
    OgaRegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Warning: failed to register execution provider library '" << registration_name
              << "': " << e.what() << std::endl;
    return false;
  }
}

// Registers each execution provider plugin library that is present in `ep_dir`.
void RegisterEpLibrariesFromDirectory(const fs::path& ep_dir) {
  std::error_code ec;
  if (ep_dir.empty()) {
    return;
  }
  if (!fs::is_directory(ep_dir, ec)) {
    std::cerr << "Warning: --ep_dir '" << ep_dir.string() << "' is not a directory." << std::endl;
    return;
  }

  for (const auto& [registration_name, library_stem] : kPluginEpLibraries) {
    // An EP may be listed under multiple candidate library names (see kPluginEpLibraries); once one
    // has registered for a given EP, skip the rest to avoid double-registering the same provider.
    // (registration_name is an arbitrary handle, not the EP name -- see kPluginEpLibraries.)
    if (registration_name == "WebGPU.GenAI" && g_webgpu_ep_registered) continue;
    if (registration_name == "CUDA.GenAI" && g_cuda_ep_registered) continue;

    const fs::path library_path = ep_dir / EpLibraryFileName(library_stem);
    if (!fs::is_regular_file(library_path, ec)) {
      continue;
    }
    bool ok = RegisterEpLibrary(std::string(registration_name), library_path.string());
    if (ok && registration_name == "WebGPU.GenAI") {
      g_webgpu_ep_registered = true;
    } else if (ok && registration_name == "CUDA.GenAI") {
      g_cuda_ep_registered = true;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;

  std::cout << "Initializing OnnxRuntime... ";
  std::cout.flush();
  try {
    std::cout << "done" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);

    // Parse custom args after InitGoogleTest (it strips its own flags).
    //   --ep_dir <dir>  register every EP plugin library found in <dir>
    fs::path ep_dir;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model_path" && i + 1 < argc) {
        g_custom_model_path = argv[++i];
        std::cout << "Using custom model path: " << g_custom_model_path << std::endl;
      } else if (arg == "--ep_dir" && i + 1 < argc) {
        ep_dir = argv[++i];
      }
    }

    RegisterEpLibrariesFromDirectory(ep_dir);

    int result = RUN_ALL_TESTS();
    std::cout << "Shutting down OnnxRuntime... ";
    OgaShutdown();
    std::cout << "done" << std::endl;
    return result;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}