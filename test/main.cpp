// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include <gtest/gtest.h>

#include "ort_genai.h"

int main(int argc, char** argv) {
  std::cout << "Generators Utility Library" << std::endl;
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