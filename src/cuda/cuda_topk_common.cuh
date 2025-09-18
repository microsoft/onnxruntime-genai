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

#ifdef STABLE_TOPK
#define Stage1TempStorage cub::BlockRadixSort<uint64_t, kBlockSize, kPartitionSize / kBlockSize>::TempStorage
#define FindPartitionTopK FindPartitionTopK_StableSort
#else
#define Stage1TempStorage cub::BlockRadixSort<float, kBlockSize, kPartitionSize / kBlockSize, int>::TempStorage
#define FindPartitionTopK FindPartitionTopK_UnstableSort
#endif

/**
 * @brief Performs a stable sort to find the Top-K candidates within a partition.
 * It uses a 64-bit composite key (score + index) to ensure stable sorting for tie-breaking.
 */
template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_StableSort(const float* __restrict__ scores_in,
                                             int* __restrict__ intermediate_indices,
                                             float* __restrict__ intermediate_scores,
                                             int vocab_size,
                                             int num_partitions,
                                             TempStorage& temp_storage) {
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

  // Sort keys from a blocked arrangement to a striped arrangement across threads.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys);

  // The first K threads now hold the top K elements.
  // This is highly efficient due to minimal thread divergence.
  if (threadIdx.x < K) {
    CompositeKey key = thread_keys[0];  // Top K keys are in the first item of the first K threads

    // Unpack the composite key to get the original score and index
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

    // Write the result to global memory
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = score;
    intermediate_indices[offset] = index;
  }
}

/**
 * @brief Performs a faster, unstable sort to find the Top-K candidates within a partition.
 * It sorts directly on 32-bit float scores without tie-breaking logic.
 */
template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_UnstableSort(const float* __restrict__ scores_in,
                                               int* __restrict__ intermediate_indices,
                                               float* __restrict__ intermediate_scores,
                                               int vocab_size,
                                               int num_partitions,
                                               TempStorage& temp_storage) {
  static_assert(kPartitionSize % kBlockSize == 0, "kPartitionSize must be a multiple of kBlockSize");
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;

  using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;

  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size) {
      thread_keys[i] = batch_scores_in[global_idx];
      thread_values[i] = global_idx;
    } else {
      thread_keys[i] = -FLT_MAX;
      thread_values[i] = INT_MAX;
    }
  }

  // Sort the keys and values held in registers across the entire block.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  // The first K threads now hold the top K elements for this partition. Write them out.
  if (threadIdx.x < K) {
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = thread_keys[0];
    intermediate_indices[offset] = thread_values[0];
  }
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators
