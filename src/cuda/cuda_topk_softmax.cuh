// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <limits>

#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// Softmax for Sorted Input and k <= 256
template <int kBlockSize, bool DoCopyIndices>
__global__ void CopyAndSoftmaxKernel(int* final_indices, float* final_scores, const int* sorted_indices,
                                     const float* sorted_scores, int k, float temperature, int input_stride) {
  const int batch_idx = blockIdx.x;
  const float* batch_sorted_scores = sorted_scores + batch_idx * input_stride;

  // This implementation uses CUB for parallel reduction, which is efficient.
  // The key fix is to correctly broadcast the reduction result (which CUB places in thread 0)
  // to all other threads in the block using shared memory.
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // STEP 1: Find max_val in parallel
  // Each thread (where threadIdx.x < k) loads its score and applies temperature.
  float thread_score =
      (threadIdx.x < k) ? (batch_sorted_scores[threadIdx.x] / temperature) : -std::numeric_limits<float>::max();

  // CUB reduces the values, placing the result in thread 0.
  float max_val_reduced = BlockReduce(temp_storage).Reduce(thread_score, cub::Max());
  if (threadIdx.x == 0) {
    block_max_val = max_val_reduced;
  }
  __syncthreads();  // Ensure block_max_val is visible to all threads.

  // STEP 2: Find sum_exp in parallel
  // Each thread calculates its contribution to the sum using the correct max value.
  float thread_exp = (threadIdx.x < k) ? expf(thread_score - block_max_val) : 0.0f;

  // CUB reduces the contributions, placing the result in thread 0.
  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_exp, cub::Sum());
  if (threadIdx.x == 0) {
    block_sum_exp = sum_exp_reduced;
  }
  __syncthreads();  // Ensure block_sum_exp is visible to all threads.

  // STEP 3: Write final results
  // Each thread (where threadIdx.x < k) writes its final index and calculated probability.
  if (threadIdx.x < k) {
    if constexpr (DoCopyIndices) {
      const int* batch_sorted_indices = sorted_indices + batch_idx * input_stride;
      final_indices[batch_idx * k + threadIdx.x] = batch_sorted_indices[threadIdx.x];
    }
    // Handle case where sum_exp is zero to prevent division by zero (NaN).
    final_scores[batch_idx * k + threadIdx.x] = (block_sum_exp > 0.0f) ? (thread_exp / block_sum_exp) : 0.0f;
  }
}

// Softmax for Sorted Input. No limitation on K.
template <int kBlockSize, bool DoCopyIndices>
__global__ void ProcessSortedTopK(float* final_scores, int* final_indices, const float* sorted_input_scores,
                                  const int* sorted_input_indices, int k, int input_stride, float temperature) {
  const int batch_idx = blockIdx.x;
  const float* batch_scores = sorted_input_scores + batch_idx * input_stride;
  [[maybe_unused]] const int* batch_indices =
      DoCopyIndices ? (sorted_input_indices + batch_idx * input_stride) : nullptr;

  // For sorted input, the max score is always the first element.
  __shared__ float max_val;
  if (threadIdx.x == 0) {
    max_val = batch_scores[0] / temperature;
  }
  __syncthreads();

  // Cooperatively calculate sum_exp in parallel.
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float sum_exp;

  float thread_sum_exp = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_sum_exp += expf((batch_scores[i] / temperature) - max_val);
  }

  float sum_exp_reduced = BlockReduce(temp_storage).Reduce(thread_sum_exp, cub::Sum());
  if (threadIdx.x == 0) {
    sum_exp = sum_exp_reduced;
  }
  __syncthreads();

  // All threads write final results.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if constexpr (DoCopyIndices) {
      final_indices[batch_idx * k + i] = batch_indices[i];
    }
    float scaled_score = batch_scores[i] / temperature;
    float thread_exp = expf(scaled_score - max_val);
    final_scores[batch_idx * k + i] = (sum_exp > 0.0f) ? (thread_exp / sum_exp) : 0.0f;
  }
}

template <bool DoCopyIndices>
void ApplySoftmaxToSortedTopK(cudaStream_t stream, float* final_scores, int* final_indices,
                              const float* sorted_input_scores, const int* sorted_input_indices, int k, int batch_size,
                              int input_stride, float temperature) {
  dim3 grid(batch_size);
  dim3 block(256);

  if (k <= 256) {
    CopyAndSoftmaxKernel<256, DoCopyIndices><<<grid, block, 0, stream>>>(
        final_indices, final_scores, sorted_input_indices, sorted_input_scores, k, temperature, input_stride);
  } else {
    ProcessSortedTopK<256, DoCopyIndices><<<grid, block, 0, stream>>>(
        final_scores, final_indices, sorted_input_scores, sorted_input_indices, k, input_stride, temperature);
  }

  CUDA_CHECK(cudaGetLastError());
}

template void ApplySoftmaxToSortedTopK<true>(cudaStream_t stream, float* final_scores, int* final_indices,
                                             const float* sorted_input_scores, const int* sorted_input_indices, int k,
                                             int batch_size, int input_stride, float temperature);

template void ApplySoftmaxToSortedTopK<false>(cudaStream_t stream, float* final_scores, int* final_indices,
                                              const float* sorted_input_scores, const int* sorted_input_indices, int k,
                                              int batch_size, int input_stride, float temperature);

}  // namespace cuda
}  // namespace Generators
