// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Generators.h"
#include <iostream>
#include "onnxruntime_cxx_api_2.h"
#if USE_CUDA
#include <cuda_runtime.h>
#endif

extern std::unique_ptr<OrtEnv> g_ort_env;

void Test_Lib_GreedySearchTest_GptGreedySearchFp32();
void Test_GreedySearchTest_GptGreedySearchFp32();

void Test_BeamSearchTest_GptBeamSearchFp32();
void Test_Lib_BeamSearchTest_GptBeamSearchFp32();

#if USE_CUDA
void LaunchTest(float* test, cudaStream_t stream);
#endif

int main()
{
#if USE_CUDA
  std::cout << "Test output" << std::endl;

  cudaError_t cuda_status = cudaSetDevice(0);
  assert(cuda_status == cudaSuccess);

  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  float *p_test;
  cudaHostAlloc(&p_test, sizeof(float), cudaHostAllocDefault);
  LaunchTest(p_test, cuda_stream);
  cudaStreamSynchronize(cuda_stream);
#endif

  Ort::InitApi();
  g_ort_env = OrtEnv::Create();

	std::cout << "Generators Utility Library" << std::endl;
  try {
    Test_Lib_GreedySearchTest_GptGreedySearchFp32();
    Test_GreedySearchTest_GptGreedySearchFp32();

    Test_Lib_BeamSearchTest_GptBeamSearchFp32();
    Test_BeamSearchTest_GptBeamSearchFp32();
  }
  catch (const std::exception& e)
  {
    std::cout << "Fatal Exception:" << e.what();
  }
	return 0;
}
