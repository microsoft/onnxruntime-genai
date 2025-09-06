// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <curand_kernel.h>
#include <memory>
#include "cuda_common.h"
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// This struct holds buffers for the SAMPLING stage.
// It inherits the Top-K buffers from the TopkData struct.
struct SamplingData : public TopkData {
  SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream);

  // Re-initializes the cuRAND states with a new seed.
  void ReInitCurandStates(unsigned long long random_seed, int batch_size, cudaStream_t stream);
  unsigned long long GetRandomSeed() const { return random_seed_; }

  // Buffers for the sampling logic (Top-P, temperature, etc.)
  cuda_unique_ptr<float> prefix_sums;
  cuda_unique_ptr<float> scores_adjusted;
  cuda_unique_ptr<float> prefix_sums_adjusted;
  cuda_unique_ptr<float> thresholds;
  cuda_unique_ptr<curandState> curand_states;

  cuda_unique_ptr<unsigned char> scan_temp_buffer;
  unsigned long long random_seed_{};
};

// Main entry point for the sampling process.
// This function orchestrates the Top-K selection followed by Top-P sampling.
void GetSample(SamplingData* sampling_data, cudaStream_t stream, int32_t* d_next_token, const float* d_scores,
               int vocab_size, int batch_size, int k, float p, float temperature);

// A general-purpose block-wise softmax implementation, needed by beam search.
template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count);

}  // namespace cuda
}  // namespace Generators
