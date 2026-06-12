// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ort_genai.h"

// On desktop platforms unit_tests links the ONNX Runtime import library directly, so we can use the
// ORT C API to enumerate the execution provider devices that are actually registered (shared with
// GenAI's native onnxruntime instance) for diagnostics.
#if defined(_WIN32)
#define ORT_GENAI_TEST_HAVE_ORT_C_API 1
#include <onnxruntime_c_api.h>
#endif

// Global variable to store custom model base path
std::string g_custom_model_path;

namespace {

namespace fs = std::filesystem;

// Shared file-name stem of every ONNX Runtime execution provider plugin library, e.g.
// "onnxruntime_providers_webgpu.dll" or "libonnxruntime_providers_cuda.so".
#if defined(_WIN32)
constexpr const char* kProviderPrefix = "onnxruntime_providers_";
constexpr const char* kProviderSuffix = ".dll";
#elif defined(__APPLE__)
constexpr const char* kProviderPrefix = "libonnxruntime_providers_";
constexpr const char* kProviderSuffix = ".dylib";
#else
constexpr const char* kProviderPrefix = "libonnxruntime_providers_";
constexpr const char* kProviderSuffix = ".so";
#endif

// Provider libraries that match the naming convention but are not registrable EP plugins.
bool IsNonPluginProviderLibrary(const std::string& ep_name) {
  return ep_name == "shared";
}

// Derives the execution provider name from a plugin library file name, following the
// "<prefix>onnxruntime_providers_<ep><suffix>" convention. Returns nullopt when the file name does
// not match the convention.
std::optional<std::string> EpNameFromLibraryFile(const fs::path& file) {
  const std::string name = file.filename().string();
  const std::string prefix = kProviderPrefix;
  const std::string suffix = kProviderSuffix;
  if (name.size() <= prefix.size() + suffix.size()) {
    return std::nullopt;
  }
  if (name.compare(0, prefix.size(), prefix) != 0 ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return std::nullopt;
  }
  return name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
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

// Enumerates every execution provider plugin library in `ep_dir` (following the ORT provider
// naming convention) and registers it under the EP name derived from its file name. A pipeline
// only needs to drop the EP plugin(s) it wants to exercise (e.g. the WebGPU, CUDA or
// NvTensorRtRtx provider library) into this single directory.
void RegisterEpLibrariesFromDirectory(const fs::path& ep_dir) {
  std::error_code ec;
  if (ep_dir.empty()) {
    return;
  }
  if (!fs::is_directory(ep_dir, ec)) {
    std::cerr << "Warning: --ep_dir '" << ep_dir.string() << "' is not a directory." << std::endl;
    return;
  }

  for (const auto& entry : fs::directory_iterator(ep_dir, ec)) {
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    const std::optional<std::string> ep_name = EpNameFromLibraryFile(entry.path());
    if (!ep_name) {
      continue;
    }
    if (IsNonPluginProviderLibrary(*ep_name)) {
      std::cout << "Skipping non-plugin provider library: " << entry.path().filename().string()
                << std::endl;
      continue;
    }
    RegisterEpLibrary(*ep_name, entry.path().string());
  }
}

// Enumerates and logs every execution provider device that ONNX Runtime currently has registered.
// Because EP plugin libraries are stored in ONNX Runtime's process-global environment (shared with
// GenAI's native onnxruntime instance), this reflects exactly the set of devices that GenAI's V2
// provider lookup will see when creating a model. This is purely diagnostic and never aborts the run.
void DumpRegisteredEpDevices() {
#if defined(ORT_GENAI_TEST_HAVE_ORT_C_API)
  const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (ort == nullptr) {
    std::cerr << "Warning: could not obtain the ONNX Runtime C API for EP device enumeration."
              << std::endl;
    return;
  }

  OrtEnv* env = nullptr;
  if (OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ep_diag", &env)) {
    std::cerr << "Warning: could not create an ONNX Runtime environment for EP device enumeration: "
              << ort->GetErrorMessage(status) << std::endl;
    ort->ReleaseStatus(status);
    return;
  }

  const OrtEpDevice* const* devices = nullptr;
  size_t num_devices = 0;
  if (OrtStatus* status = ort->GetEpDevices(env, &devices, &num_devices)) {
    std::cerr << "Warning: GetEpDevices failed: " << ort->GetErrorMessage(status) << std::endl;
    ort->ReleaseStatus(status);
  } else {
    std::cout << "Registered execution provider devices (" << num_devices << "):" << std::endl;
    for (size_t i = 0; i < num_devices; ++i) {
      const char* ep_name = ort->EpDevice_EpName(devices[i]);
      const char* ep_vendor = ort->EpDevice_EpVendor(devices[i]);
      std::cout << "  [" << i << "] EP='" << (ep_name ? ep_name : "<null>")
                << "' vendor='" << (ep_vendor ? ep_vendor : "<null>") << "'" << std::endl;
    }
  }

  ort->ReleaseEnv(env);
#else
  std::cout << "EP device enumeration is not available on this platform." << std::endl;
#endif
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
    DumpRegisteredEpDevices();

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