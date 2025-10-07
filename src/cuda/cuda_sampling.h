// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <curand_kernel.h>
#include <memory>
#include "cuda_common.h"
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

constexpr int kFusedSamplingMaxK = 256;  // Threshold to switch from Fused to Multi-Stage sampling

// This struct holds buffers for the SAMPLING stage.
// It inherits the Top-K buffers from the TopkData struct.
struct SamplingData : public TopkData {
  SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream, void* buffer = nullptr, size_t buffer_size = 0);

  static size_t CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream);

  void ReInitCurandStates(unsigned long long random_seed, int batch_size, cudaStream_t stream);

  // Buffers for the sampling logic
  float* prefix_sums;
  float* scores_adjusted;
  float* prefix_sums_adjusted;
  float* thresholds;
  curandState* curand_states;

  unsigned long long random_seed_{};

 private:
  void InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream) override;
};

// Main entry point for the sampling process.
// This function orchestrates the Top-K selection followed by Top-P sampling.
void GetSample(SamplingData* sampling_data, cudaStream_t stream, int32_t* d_next_token, const float* d_scores,
               int vocab_size, int batch_size, int k, float p, float temperature);

// A general-purpose block-wise softmax implementation, needed by beam search.
template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count);

// The following are for macro benchmark.
void LaunchFusedSampleKernel(SamplingData* data, cudaStream_t stream, const float* scores, const int* indices,
                             int32_t* next_token_out, int k, int batch_size, float p, float temperature, int stride);

void LaunchMultiStageSampleKernel(SamplingData* data, cudaStream_t stream, const float* scores, const int* indices,
                                  int32_t* next_token_out, int k, int batch_size, float p, float temperature, int stride);

}  // namespace cuda
}  // namespace Generators
