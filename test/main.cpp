// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "ort_genai.h"

// Global variable to store custom model base path
std::string g_custom_model_path;

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;

  // Parse custom model path argument
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model_path" && i + 1 < argc) {
      g_custom_model_path = argv[++i];
      std::cout << "Using custom model path: " << g_custom_model_path << std::endl;
      break;
    }
  }

  std::cout << "Initializing OnnxRuntime... ";
  std::cout.flush();
  try {
    std::cout << "done" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
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