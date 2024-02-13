// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <iostream>
// #include "tests.cpp"
// #include "c_api.cpp"

extern std::unique_ptr<OrtEnv> g_ort_env;

int main(int argc, char **argv) {
  std::cout << "Generators Utility Library" << std::endl;
  std::cout << "Initializing OnnxRuntime... ";
  std::cout.flush();
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  std::cout << "done" << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  std::cout << "Shutting down OnnxRuntime... ";
  g_ort_env.release();
  std::cout << "done" << std::endl;
  return result;
}