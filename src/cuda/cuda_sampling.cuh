// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include "cuda_common.h"
#include <curand_kernel.h>

namespace Generators {
namespace cuda {

struct SamplingData {
  SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream);
  cuda_unique_ptr<int> indices_sorted;
  cuda_unique_ptr<float> scores_sorted;
  cuda_unique_ptr<float> scores_softmaxed;
  cuda_unique_ptr<float> prefix_sums;
  cuda_unique_ptr<float> thresholds;
  cuda_unique_ptr<int> indices_in;
  cuda_unique_ptr<int> offsets;
  cuda_unique_ptr<float> temp_buffer;
  cuda_unique_ptr<curandState> curand_states;
  size_t temp_storage_bytes = 0;
};

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream);
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* d_next_token, float* d_scores, int vocab_size, int batch_size, int k, float p, float temperature);

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements, int input_stride, int output_stride, int batch_count, float temperature = 1.0);

}  // namespace cuda
}  // namespace Generators