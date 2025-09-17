// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace topk_common {

/**
 * @brief Reusable device function for Stage 1 of Top-K sorting algorithms.
 *
 * This function is called by a kernel to find the Top-K candidates within a
 * single partition of the input data. It uses a single, fast, non-stable
 * radix sort. The output candidates will have the correct Top-K scores, but
 * tie-breaking order is not guaranteed. The final stable sort must be handled
 * in subsequent reduction stages or a finalization kernel.
 *
 * @tparam kBlockSize The number of threads in the thread block.
 * @tparam kPartitionSize The size of the data partition to sort.
 * @tparam K The number of top elements to find.
 * @tparam SharedStorage The type of the shared memory storage from the calling kernel.
 * @param scores_in Pointer to the input scores in global memory.
 * @param intermediate_indices Pointer to the intermediate buffer for indices.
 * @param intermediate_scores Pointer to the intermediate buffer for scores.
 * @param vocab_size The total vocabulary size.
 * @param num_partitions The total number of partitions.
 * @param smem A reference to the shared memory from the calling kernel.
 */
template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_Stable(const float* __restrict__ scores_in,
                                         int* __restrict__ intermediate_indices,
                                         float* __restrict__ intermediate_scores,
                                         int vocab_size,
                                         int num_partitions,
                                         TempStorage& smem) {
  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  // Use 64-bit composite key: upper 32 bits for score, lower 32 bits for inverted index
  using CompositeKey = uint64_t;
  using BlockRadixSort = cub::BlockRadixSort<CompositeKey, kBlockSize, ItemsPerThread>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  CompositeKey thread_keys[ItemsPerThread];

  // Create composite keys: score (transformed for sorting) + index for tie-breaking
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    float score;
    int index;

    if (global_idx < vocab_size) {
      score = batch_scores_in[global_idx];
      index = global_idx;
    } else {
      score = -FLT_MAX;
      index = INT_MAX;
    }

    // Transform float to sortable uint32 (for descending order)
    uint32_t score_bits = __float_as_uint(score);
    uint32_t sortable_score = (score_bits & 0x80000000) ? (~score_bits) : (score_bits | 0x80000000);

    // Create composite key: score in upper 32 bits, inverted index in lower 32 bits
    // Inverted index ensures smaller indices come first for ties
    uint32_t inverted_index = UINT32_MAX - index;
    thread_keys[i] = (static_cast<uint64_t>(sortable_score) << 32) | inverted_index;
  }

  // Sort in descending order by composite key
  BlockRadixSort(smem).SortDescending(thread_keys);

  // Extract top K results from sorted data
  for (int i = 0; i < ItemsPerThread; ++i) {
    int rank = threadIdx.x * ItemsPerThread + i;
    if (rank < K) {
      CompositeKey key = thread_keys[i];
      uint32_t sortable_score = static_cast<uint32_t>(key >> 32);
      uint32_t inverted_index = static_cast<uint32_t>(key & 0xFFFFFFFF);

      // Reverse the score transformation
      uint32_t score_bits;
      if (sortable_score & 0x80000000) {
        // Was originally a positive float
        score_bits = sortable_score ^ 0x80000000;
      } else {
        // Was originally a negative float
        score_bits = ~sortable_score;
      }
      float score = __uint_as_float(score_bits);
      int index = UINT32_MAX - inverted_index;

      size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + rank;
      intermediate_scores[offset] = score;
      intermediate_indices[offset] = index;
    }
  }
}

/**
 * @brief Kernel to perform a final, stable sort on K candidates when there is
 * only one partition.
 *
 * @tparam kBlockSize The number of threads in the thread block.
 * @tparam K_padded The padded size of K, which must be a power of two for bitonic sort.
 * @param scores_in The unsorted Top-K candidate scores.
 * @param indices_in The unsorted Top-K candidate indices.
 * @param scores_out The final, sorted Top-K scores.
 * @param indices_out The final, sorted Top-K indices.
 * @param k_final The actual value of K (the number of elements to write).
 */
template <int kBlockSize, int K_padded>
__global__ void SinglePartitionBitonicSort(const float* __restrict__ scores_in,
                                           const int* __restrict__ indices_in,
                                           float* __restrict__ scores_out,
                                           int* __restrict__ indices_out,
                                           int k_final) {
  __shared__ float smem_scores[K_padded];
  __shared__ int smem_indices[K_padded];

  const int batch_idx = blockIdx.y;
  const size_t in_offset = static_cast<size_t>(batch_idx) * K_padded;
  const size_t out_offset = static_cast<size_t>(batch_idx) * k_final;

  // Load the K_padded candidates into shared memory
  for (int i = threadIdx.x; i < K_padded; i += kBlockSize) {
    smem_scores[i] = scores_in[in_offset + i];
    smem_indices[i] = indices_in[in_offset + i];
  }
  __syncthreads();

  // Sort the K_padded candidates in shared memory
  bitonic_sort::SharedMemBitonicSort<kBlockSize, K_padded>(smem_scores, smem_indices);

  // Write out the final top-k_final results
  if (threadIdx.x < k_final) {
    scores_out[out_offset + threadIdx.x] = smem_scores[threadIdx.x];
    indices_out[out_offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators
