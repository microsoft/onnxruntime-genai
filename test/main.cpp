// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"
#include "ep_registration.h"

namespace fs = std::filesystem;

// Global variable to store custom model base path
std::string g_custom_model_path;
// Set to true when the corresponding plugin EP is successfully registered (via --ep_dir or, with
// --winml_eps, a WinML EP package). Read by the EP-specific tests in c_api_tests.cpp.
bool g_webgpu_ep_registered = false;
bool g_cuda_ep_registered = false;
bool g_openvino_ep_registered = false;

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;

  std::cout << "Initializing OnnxRuntime... ";
  std::cout.flush();
  try {
    std::cout << "done" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);

    // Parse custom args after InitGoogleTest (it strips its own flags).
    //   --ep_dir <dir>   register every known EP plugin library found under <dir> (recursive)
    //   --winml_eps      (Windows) also discover and register WinML-installed EP packages
    fs::path ep_dir;
    bool use_winml_eps = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model_path" && i + 1 < argc) {
        g_custom_model_path = argv[++i];
        std::cout << "Using custom model path: " << g_custom_model_path << std::endl;
      } else if (arg == "--ep_dir" && i + 1 < argc) {
        ep_dir = argv[++i];
      } else if (arg == "--winml_eps") {
        use_winml_eps = true;
      }
    }

    test_ep::EpRegistrar ep_registrar;
    ep_registrar.DiscoverFromDirectory(ep_dir);

#if defined(_WIN32)
    // Opt-in: discover WinML-installed EP packages so WebGPU / CUDA / TensorRT-RTX / OpenVINO tests
    // can run on Windows without an explicit --ep_dir. Kept opt-in so the legacy-path tests (which
    // require the plugin EP to be absent) can still be exercised on a machine that has WinML EPs.
    if (use_winml_eps)
      ep_registrar.DiscoverWinML();
#else
    (void)use_winml_eps;
#endif

    ep_registrar.RegisterAll();
    g_webgpu_ep_registered = ep_registrar.IsRegistered("WebGPU.GenAI");
    g_cuda_ep_registered = ep_registrar.IsRegistered("CUDA.GenAI");
    g_openvino_ep_registered = ep_registrar.IsRegistered("OpenVINO.GenAI");

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