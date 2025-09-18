// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <stdio.h>
#include "cuda_sampling.h"
#include "cuda_topk.h"
#include "cuda_topk_softmax.cuh"
#include "smartptrs.h"
#include "span.h"

namespace Generators {
namespace cuda {

// Initializes the cuRAND states for each batch item.
__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size) return;
  curand_init(seed, index, 0, &states[index]);
}

void SamplingData::ReInitCurandStates(unsigned long long random_seed, int batch_size, cudaStream_t stream) {
  random_seed_ = random_seed;
  InitCurandStates<<<CeilDiv(batch_size, 128), 128, 0, stream>>>(random_seed, curand_states, batch_size);
  CUDA_CHECK_LAUNCH();
}

size_t SamplingData::CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream) {
  // Get size from base class and add our own buffer sizes
  size_t total_size = TopkData::CalculateTotalSize(batch_size, vocab_size, stream);
  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  total_size += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);  // prefix_sums
  total_size += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);  // scores_adjusted
  total_size += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);  // prefix_sums_adjusted
  total_size += AlignUp(batch_size * sizeof(float), kGpuBufferAlignment);        // thresholds
  total_size += AlignUp(batch_size * sizeof(curandState), kGpuBufferAlignment);  // curand_states

  return total_size;
}

void SamplingData::InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream) {
  // The base class constructor has already called TopkData::InitializeBuffers.
  // This override is only responsible for initializing the members of the derived class.

  // Calculate the starting offset for the derived class members by getting the total size of the base class.
  uint8_t* current_ptr = memory_buffer_span_.data() + TopkData::CalculateTotalSize(batch_size, vocab_size, stream);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  prefix_sums = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);

  scores_adjusted = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);

  prefix_sums_adjusted = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(vocab_batch_size * sizeof(float), kGpuBufferAlignment);

  thresholds = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(batch_size * sizeof(float), kGpuBufferAlignment);

  curand_states = reinterpret_cast<curandState*>(current_ptr);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream, void* buffer, size_t buffer_size)
    : TopkData(batch_size, vocab_size, stream, buffer,
               buffer ? buffer_size : CalculateTotalSize(batch_size, vocab_size, stream)) {
  // The base constructor handles buffer allocation/wrapping.
  // We just need to re-initialize the pointers for the derived class members.
  InitializeBuffers(batch_size, vocab_size, stream);
  ReInitCurandStates(random_seed, batch_size, stream);
}

// A fused kernel that performs all steps of Top-P sampling on a pre-selected set of Top-K candidates.
// This monolithic approach minimizes kernel launch overhead and maximizes data locality by using shared memory.
// It has been empirically shown to be the most performant approach for k <= 256.
template <int kBlockSize>
__global__ void FusedSamplingKernel(int32_t* next_token_out, const float* scores, const int* indices, int k,
                                    float p, float temperature, int stride, curandState* curand_states) {
  const int batch_idx = blockIdx.x;
  const float* batch_scores = scores + batch_idx * stride;
  const int* batch_indices = indices + batch_idx * stride;

  // Allocate shared memory for all intermediate data. This is the key to performance.
  extern __shared__ float smem[];
  float* temp_scaled_logits = smem;
  float* filtered_logits = smem + kBlockSize;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage reduce_temp_storage;
  __shared__ float block_max_val;
  __shared__ float block_sum_exp;

  // --- Stage 1: Initial Softmax with Temperature (for Top-P filtering) ---

  // Apply temperature scaling.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    temp_scaled_logits[i] = batch_scores[i] / temperature;
  }

  // For sorted input, the max score is always the first element.
  if (threadIdx.x == 0) {
    block_max_val = batch_scores[0] / temperature;
  }
  __syncthreads();

  float thread_val = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val += expf(temp_scaled_logits[i] - block_max_val);
  }
  float reduced_sum = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Sum());
  if (threadIdx.x == 0) block_sum_exp = reduced_sum;
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    temp_scaled_logits[i] =
        (block_sum_exp > 0.0f) ? (expf(temp_scaled_logits[i] - block_max_val) / block_sum_exp) : 0.0f;
  }
  __syncthreads();

  // --- Stage 2: Compute Initial CDF (in-place scan on initial probabilities) ---
  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_temp_storage;
  float running_total = 0.0f;
  for (int i = 0; i < k; i += kBlockSize) {
    float score = (threadIdx.x + i < k) ? temp_scaled_logits[threadIdx.x + i] : 0.0f;
    float scanned_score;
    BlockScan(scan_temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();

    if (threadIdx.x + i < k) temp_scaled_logits[threadIdx.x + i] = scanned_score + running_total;
    __syncthreads();

    if (threadIdx.x == kBlockSize - 1) running_total += scanned_score;
    __syncthreads();
  }

  // --- Stage 3: Filter SCALED logits based on the CDF ---
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    const float prev_sum = (i == 0) ? 0.0f : temp_scaled_logits[i - 1];
    // Reread scaled logits from global memory to filter
    float current_scaled_logit = batch_scores[i] / temperature;
    filtered_logits[i] = (prev_sum < p) ? current_scaled_logit : -FLT_MAX;
  }
  __syncthreads();

  // --- Stage 4: Re-normalize filtered logits (temperature=1.0 as it's already baked in) ---
  thread_val = -FLT_MAX;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val = max(thread_val, filtered_logits[i]);
  }
  float reduced_max = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Max());
  if (threadIdx.x == 0) block_max_val = reduced_max;
  __syncthreads();

  thread_val = 0.0f;
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    thread_val += expf(filtered_logits[i] - block_max_val);
  }
  reduced_sum = BlockReduce(reduce_temp_storage).Reduce(thread_val, cub::Sum());
  if (threadIdx.x == 0) block_sum_exp = reduced_sum;
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    filtered_logits[i] = (block_sum_exp > 0.0f) ? (expf(filtered_logits[i] - block_max_val) / block_sum_exp) : 0.0f;
  }
  __syncthreads();

  // --- Stage 5: Compute Final CDF (in-place scan on final probabilities) ---
  running_total = 0.0f;
  for (int i = 0; i < k; i += kBlockSize) {
    float score = (threadIdx.x + i < k) ? filtered_logits[threadIdx.x + i] : 0.0f;
    float scanned_score;
    BlockScan(scan_temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();
    if (threadIdx.x + i < k) filtered_logits[threadIdx.x + i] = scanned_score + running_total;
    __syncthreads();
    if (threadIdx.x == kBlockSize - 1) running_total += scanned_score;
    __syncthreads();
  }

  // --- Stage 6 & 7: Sample via Parallel Search ---
  __shared__ int selected_index_smem;
  __shared__ float threshold_smem;

  if (threadIdx.x == 0) {
    // Use min to prevent multiplying down the random value, which could introduce bias.
    // This robustly handles the case where curand_uniform is exactly 1.0.
    threshold_smem = min(curand_uniform(&curand_states[batch_idx]), 0.9999999f);
    selected_index_smem = k - 1;
  }
  __syncthreads();

  // All threads in the block search in parallel for the first index that meets the threshold.
  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if (filtered_logits[i] >= threshold_smem) {
      atomicMin(&selected_index_smem, i);
      break;  // Early exit for this thread
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    next_token_out[batch_idx] = batch_indices[selected_index_smem];
  }
}

// Kernels for the multi-stage sampling pipeline (used for k > 256)
#pragma region MultiStageKernels

template <int kBlockSize>
__global__ void CorrectPrefixSumKernel(const float* scores, float* prefix_sums, int k) {
  const int batch_idx = blockIdx.x;
  const float* batch_scores = scores + batch_idx * k;
  float* batch_prefix_sums = prefix_sums + batch_idx * k;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ float chunk_total;
  float running_total = 0.0f;

  for (int i = 0; i < k; i += kBlockSize) {
    float score = (threadIdx.x + i < k) ? batch_scores[threadIdx.x + i] : 0.0f;
    float scanned_score;
    BlockScan(temp_storage).InclusiveSum(score, scanned_score);
    __syncthreads();
    if (threadIdx.x + i < k) {
      batch_prefix_sums[threadIdx.x + i] = scanned_score + running_total;
    }
    __syncthreads();
    if (threadIdx.x == kBlockSize - 1) {
      chunk_total = scanned_score;
    }
    __syncthreads();
    running_total += chunk_total;
  }
}

__global__ void FilterOnTopPKernel(float* filtered_logits, const float* original_logits, const float* cdf, int k,
                                   float p, float temperature, int stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float prev_sum = (i == 0) ? 0.0f : cdf[batch_idx * k + i - 1];
    float scaled_logit = original_logits[batch_idx * stride + i] / temperature;
    filtered_logits[batch_idx * k + i] = (prev_sum < p) ? scaled_logit : -FLT_MAX;
  }
}

__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < batch_size) {
    // Use min to prevent multiplying down the random value, which could introduce bias.
    thresholds[i] = min(curand_uniform(&curand_states[i]), 0.9999999f);
  }
}

template <int kBlockSize>
__global__ void SampleKernel(int32_t* next_token_out, const int* indices, const float* cdf, int k, int stride,
                             const float* thresholds) {
  const int batch_idx = blockIdx.x;
  const float threshold = thresholds[batch_idx];
  __shared__ int selected_index_smem;

  if (threadIdx.x == 0) selected_index_smem = k - 1;
  __syncthreads();

  for (int i = threadIdx.x; i < k; i += kBlockSize) {
    if (cdf[batch_idx * k + i] >= threshold) {
      atomicMin(&selected_index_smem, i);
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    next_token_out[batch_idx] = indices[batch_idx * stride + selected_index_smem];
  }
}
#pragma endregion

// A multi-stage sampling pipeline that is more efficient for large k.
void LaunchMultiStageSampleKernel(SamplingData* data, cudaStream_t stream, const float* scores, const int* indices,
                                  int32_t* next_token_out, int k, int batch_size, float p, float temperature,
                                  int stride) {
  dim3 grid(batch_size);
  dim3 block(256);

  // Stage 1: Initial Softmax with Temperature.
  ApplySoftmaxToSortedTopK<false>(stream, data->prefix_sums_adjusted, nullptr, scores, nullptr, k, batch_size,
                                  stride, temperature);

  // Stage 2: Compute Initial CDF.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted, data->prefix_sums, k);

  // Stage 3: Filter scaled logits.
  FilterOnTopPKernel<<<grid, block, 0, stream>>>(data->scores_adjusted, scores, data->prefix_sums, k, p,
                                                 temperature, stride);

  // Stage 4: Re-normalize filtered logits (temperature is already baked in).
  ApplySoftmaxToSortedTopK<false>(stream, data->prefix_sums_adjusted, nullptr, data->scores_adjusted,
                                  nullptr, k, batch_size, k, 1.0f);

  // Stage 5: Compute Final CDF.
  CorrectPrefixSumKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted, data->prefix_sums, k);

  // Stage 6: Generate random thresholds.
  RandomThresholdKernel<<<CeilDiv(batch_size, 256), block, 0, stream>>>(data->curand_states,
                                                                        data->thresholds, batch_size);

  // Stage 7: Sample via Parallel Search.
  SampleKernel<256><<<grid, block, 0, stream>>>(next_token_out, indices, data->prefix_sums, k, stride,
                                                data->thresholds);
}

void LaunchFusedSampleKernel(SamplingData* data, cudaStream_t stream, const float* scores, const int* indices,
                             int32_t* next_token_out, int k, int batch_size, float p, float temperature, int stride) {
  assert(k <= kFusedSamplingMaxK);
  dim3 grid(batch_size);
  constexpr int block_size = 256;
  dim3 block(block_size);

  // Shared memory size is determined by the needs of the fused kernel: two float arrays of size block_size.
  constexpr size_t shared_mem_bytes = 2 * block_size * sizeof(float);

  FusedSamplingKernel<block_size><<<grid, block, shared_mem_bytes, stream>>>(
      next_token_out, scores, indices, k, p, temperature, stride, data->curand_states);
}

void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, const float* scores_in,
               int vocab_size, int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  RunTopK(data, stream, scores_in, vocab_size, batch_size, k);
  const float* topk_scores = data->topk_scores;
  const int* topk_indices = data->topk_indices;
  int topk_stride = data->topk_stride;

  // The fused kernel is the most performant approach for k up to 256.
  if (k <= kFusedSamplingMaxK) {
    LaunchFusedSampleKernel(data, stream, topk_scores, topk_indices, next_token_out, k, batch_size, p,
                            temperature, topk_stride);
  } else {
    // Fall back to multi-stage sampling pipeline. This is not a typical use case.
    LaunchMultiStageSampleKernel(data, stream, topk_scores, topk_indices, next_token_out, k, batch_size, p,
                                 temperature, topk_stride);
  }
  CUDA_CHECK_LAUNCH();
}

// Implementation for the general-purpose block-wise softmax, used by beam search.
template <int kBlockSize, bool is_log_softmax>
__global__ void BlockwiseSoftmaxKernel(float* output, const float* input, int softmax_elements, int input_stride,
                                       int output_stride) {
  const int batch_idx = blockIdx.x;
  const float* batch_input = input + batch_idx * input_stride;
  float* batch_output = output + batch_idx * output_stride;

  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float max_val;
  __shared__ float sum_exp;

  // Step 1: Find max value in parallel for numerical stability.
  float thread_max = -std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
    thread_max = max(thread_max, batch_input[i]);
  }
  float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) {
    max_val = block_max;
  }
  __syncthreads();

  // Step 2: Compute sum of exponents in parallel.
  float thread_sum_exp = 0.0f;
  for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
    thread_sum_exp += expf(batch_input[i] - max_val);
  }
  float block_sum = BlockReduce(temp_storage).Reduce(thread_sum_exp, cub::Sum());
  if (threadIdx.x == 0) {
    sum_exp = block_sum;
  }
  __syncthreads();

  // Step 3: Compute final softmax or log_softmax and write to output.
  if constexpr (is_log_softmax) {
    // Add a small epsilon to prevent log(0) which results in -inf.
    float log_sum_exp = logf(sum_exp + 1e-20f);
    for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
      batch_output[i] = batch_input[i] - max_val - log_sum_exp;
    }
  } else {
    for (int i = threadIdx.x; i < softmax_elements; i += kBlockSize) {
      // Handle case where sum_exp is zero to prevent division by zero (NaN).
      batch_output[i] = (sum_exp > 0.0f) ? (expf(batch_input[i] - max_val) / sum_exp) : 0.0f;
    }
  }
}

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count) {
  // This kernel is efficient for large softmax_elements (like vocab_size) where
  // a single block can cooperatively process one batch item.
  constexpr int kBlockSize = 256;
  dim3 grid(batch_count);
  dim3 block(kBlockSize);

  BlockwiseSoftmaxKernel<kBlockSize, is_log_softmax><<<grid, block, 0, stream>>>(output, input, softmax_elements,
                                                                                 input_stride, output_stride);
  CUDA_CHECK_LAUNCH();
}

// Explicitly instantiate the templates to be linked from other translation units.
template void DispatchBlockwiseSoftmaxForward<true>(cudaStream_t, float*, const float*, int, int, int, int);
template void DispatchBlockwiseSoftmaxForward<false>(cudaStream_t, float*, const float*, int, int, int, int);

}  // namespace cuda
}  // namespace Generators
