// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "ep_registration.h"
#include "ort_genai.h"
#include "telemetry_test_environment.h"

// Global variable to store custom model base path
std::string g_custom_model_path;

namespace {

namespace fs = std::filesystem;

}  // namespace

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;

  // Fully suppress telemetry for the unit-test process before any Oga call (and therefore the
  // telemetry provider) runs, so local non-CI test runs never spin up the 1DS uploader, write a
  // device id, or emit events. Read by GenAiTelemetry::Initialize (shared name with ONNX Runtime).
  Generators::test::SuppressTelemetryForTests();

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

    test_ep::EpRegistrar ep_registrar;
    ep_registrar.DiscoverFromDirectory(ep_dir);
    ep_registrar.RegisterAll();

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