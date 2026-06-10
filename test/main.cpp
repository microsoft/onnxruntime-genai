// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ort_genai.h"

// Global variable to store custom model base path
std::string g_custom_model_path;

namespace {

namespace fs = std::filesystem;

// Platform-specific file name of the WebGPU execution provider plugin library.
std::string WebGpuProviderLibraryName() {
#if defined(_WIN32)
  return "onnxruntime_providers_webgpu.dll";
#elif defined(__APPLE__)
  return "libonnxruntime_providers_webgpu.dylib";
#else
  return "libonnxruntime_providers_webgpu.so";
#endif
}

// Registers an execution provider plugin library with ONNX Runtime, logging the outcome.
// Registration failures are reported but do not abort the test run.
void RegisterEpLibrary(const std::string& registration_name, const std::string& library_path) {
  try {
    std::cout << "Registering execution provider library '" << registration_name << "' -> "
              << library_path << std::endl;
    // The ORT environment is a singleton shared with ONNX Runtime GenAI, so registering the
    // plugin library here (via the C API, against the native onnxruntime) makes the EP available
    // to all models created in this process. No C++ wrapper API is needed for this.
    OgaRegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "Warning: failed to register execution provider library '" << registration_name
              << "': " << e.what() << std::endl;
  }
}

// Locates the WebGPU provider library so it can be registered without explicit configuration.
// Checks, in order: an explicit env var, the test executable's directory, and the cwd.
std::string FindWebGpuProviderLibrary(const fs::path& exe_dir) {
  if (const char* env_path = std::getenv("ORTGENAI_TEST_WEBGPU_EP_LIBRARY"); env_path && *env_path) {
    if (fs::exists(env_path)) {
      return env_path;
    }
    std::cerr << "Warning: ORTGENAI_TEST_WEBGPU_EP_LIBRARY is set to '" << env_path
              << "' but the file does not exist." << std::endl;
  }

  const std::string library_name = WebGpuProviderLibraryName();
  for (const fs::path& dir : {exe_dir, fs::current_path()}) {
    if (dir.empty()) {
      continue;
    }
    const fs::path candidate = dir / library_name;
    if (fs::exists(candidate)) {
      return candidate.string();
    }
  }

  return {};
}

// Registers execution provider plugin libraries requested via:
//   * repeatable CLI args: --register_ep_library <name> <path>
//   * the WebGPU provider library discovered on the filesystem / via env var (as "webgpu").
void RegisterRequestedEpLibraries(const std::vector<std::pair<std::string, std::string>>& explicit_libraries,
                                  const fs::path& exe_dir) {
  for (const auto& [name, path] : explicit_libraries) {
    RegisterEpLibrary(name, path);
  }

  if (const std::string webgpu_path = FindWebGpuProviderLibrary(exe_dir); !webgpu_path.empty()) {
    RegisterEpLibrary("webgpu", webgpu_path);
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

    const fs::path exe_dir = (argc > 0 && argv[0] != nullptr)
                                 ? std::filesystem::absolute(argv[0]).parent_path()
                                 : std::filesystem::path{};

    // Parse custom args after InitGoogleTest (it strips its own flags).
    std::vector<std::pair<std::string, std::string>> ep_libraries;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model_path" && i + 1 < argc) {
        g_custom_model_path = argv[++i];
        std::cout << "Using custom model path: " << g_custom_model_path << std::endl;
      } else if (arg == "--register_ep_library" && i + 2 < argc) {
        const std::string name = argv[++i];
        const std::string path = argv[++i];
        ep_libraries.emplace_back(name, path);
      }
    }

    RegisterRequestedEpLibraries(ep_libraries, exe_dir);

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