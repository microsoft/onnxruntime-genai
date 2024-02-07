// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <generators.h>
#include <iostream>

extern std::unique_ptr<OrtEnv> g_ort_env;

void Test_GreedySearch_Gpt_Fp32_C_API();
void Test_GreedySearch_Gpt_Fp32();
void Test_BeamSearch_Gpt_Fp32();

#if USE_CUDA
void Test_Phi2_Cuda();
void Test_GreedySearch_Gpt_Cuda();
void Test_BeamSearch_Gpt_Cuda();
void Test_Sampling_Cuda();
#endif

int main() {
  std::cout << "Generators Utility Library" << std::endl;

  std::cout << "Initializing OnnxRuntime...";
  std::cout.flush();
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  std::cout << "done" << std::endl;

  try {
    Test_GreedySearch_Gpt_Fp32_C_API();
    Test_GreedySearch_Gpt_Fp32();
    Test_BeamSearch_Gpt_Fp32();

#if USE_CUDA
    Test_GreedySearch_Gpt_Cuda();
    Test_BeamSearch_Gpt_Cuda();
    Test_Phi2_Cuda();
    Test_Sampling_Cuda();
#endif
  } catch (const std::exception& e) {
    std::cout << "Fatal Exception: " << e.what() << std::endl;
  }
  return 0;
}
