// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include <iostream>

extern std::unique_ptr<OrtEnv> g_ort_env;

void Test_Lib_GreedySearchTest_GptGreedySearchFp32();
void Test_GreedySearchTest_GptGreedySearchFp32();

void Test_BeamSearchTest_GptBeamSearchFp32();
void Test_Lib_BeamSearchTest_GptBeamSearchFp32();

#if USE_CUDA
void LaunchTest(float* test, cudaStream_t stream);

void Test_Lib_GreedySearchTest_GptGreedySearchFp32_Cuda();
void Test_Lib_BeamSearchTest_GptBeamSearchFp32_Cuda();
#endif

int main()
{
	std::cout << "Generators Utility Library" << std::endl;

	std::cout << "Initializing OnnxRuntime..."; std::cout.flush();
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  std::cout << "done" << std::endl;

  try {
    Test_Lib_GreedySearchTest_GptGreedySearchFp32();
    Test_GreedySearchTest_GptGreedySearchFp32();

    Test_Lib_BeamSearchTest_GptBeamSearchFp32();
    Test_BeamSearchTest_GptBeamSearchFp32();

#if USE_CUDA
    Test_Lib_GreedySearchTest_GptGreedySearchFp32_Cuda();
    Test_Lib_BeamSearchTest_GptBeamSearchFp32_Cuda();
#endif
  }
  catch (const std::exception& e)
  {
    std::cout << "Fatal Exception:" << e.what();
  }
	return 0;
}
