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

// Execution providers that can be loaded as plugin libraries at test time, mapped to the
// platform-independent stem of their library file. The full file name is built as
// "<prefix><stem><suffix>", e.g. on Windows "webgpu" -> "onnxruntime_providers_webgpu.dll".
constexpr std::array<std::pair<std::string_view, std::string_view>, 1> kPluginEpLibraries = {{
    {"WebGpuExecutionProvider", "onnxruntime_providers_webgpu"},
}};

// Builds the platform-specific plugin library file name for an EP library stem.
std::string EpLibraryFileName(std::string_view stem) {
  return std::string(kProviderPrefix) + std::string(stem) + kProviderSuffix;
}

// Registers an execution provider plugin library with ONNX Runtime, logging the outcome.
// Registration failures are reported but do not abort the test run.
void RegisterEpLibrary(const std::string& registration_name, const std::string& library_path) {
  try {
    std::cout << "Registering execution provider library '" << registration_name << "' -> "
              << library_path << std::endl;
    OgaRegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());
  } catch (const std::exception& e) {
    std::cerr << "Warning: failed to register execution provider library '" << registration_name
              << "': " << e.what() << std::endl;
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

  for (const auto& [ep_name, library_stem] : kPluginEpLibraries) {
    const fs::path library_path = ep_dir / EpLibraryFileName(library_stem);
    if (!fs::is_regular_file(library_path, ec)) {
      continue;
    }
    RegisterEpLibrary(std::string(ep_name), library_path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;

  // Fully suppress telemetry for the unit-test process before any Oga call (and therefore the
  // telemetry provider) runs, so local non-CI test runs never spin up the 1DS uploader, write a
  // device id, or emit events. Read by GenAiTelemetry::Initialize (shared name with ONNX Runtime).
#if defined(_WIN32)
  _putenv_s("ORT_RUNNING_UNIT_TESTS", "1");
#else
  setenv("ORT_RUNNING_UNIT_TESTS", "1", 1);
#endif

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