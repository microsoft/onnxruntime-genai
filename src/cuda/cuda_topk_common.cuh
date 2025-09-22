// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <float.h>
#include "cuda_topk.h"
#include "cuda_topk_stable_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace topk_common {

/**
 * @brief Finds the Top-K candidates within a single data partition using `cub::BlockRadixSort`.
 * This is the core workhorse for Stage 1 of all partition-based algorithms.
 * It dispatches to a stable or unstable version at compile time based on the `kStableTopK` flag.
 *
 * @param scores_in Pointer to the start of the input scores for the entire batch.
 * @param intermediate_indices Output buffer for the indices of the top candidates.
 * @param intermediate_scores Output buffer for the scores of the top candidates.
 * @param vocab_size The total vocabulary size.
 * @param num_partitions The total number of partitions the vocabulary is divided into.
 * @param temp_storage Shared memory allocated for CUB's internal use.
 */
template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK(const float* __restrict__ scores_in,
                                  int* __restrict__ intermediate_indices,
                                  float* __restrict__ intermediate_scores,
                                  int vocab_size,
                                  int num_partitions,
                                  TempStorage& temp_storage);

/**
 * @brief Performs a stable sort to find the Top-K candidates within a partition.
 * It uses a 64-bit composite key, packing the float score and its integer index together.
 * This ensures that when scores are tied, the element with the original smaller index is ranked higher.
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

    thread_keys[i] = topk_common::PackStableSortKey(score, index);
  }

  // Sort keys from a blocked arrangement to a striped arrangement across threads.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys);

  // The first K threads now hold the top K elements.
  // This is highly efficient due to minimal thread divergence.
  if (threadIdx.x < K) {
    uint64_t key = thread_keys[0];  // Top K keys are in the first item of the first K threads

    // Unpack the composite key to get the original score and index
    float score = topk_common::UnpackStableSortScore(key);
    int index = topk_common::UnpackStableSortIndex(key);

    // Write the result to global memory
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = score;
    intermediate_indices[offset] = index;
  }
}

/**
 * @brief Performs a faster, unstable sort to find the Top-K candidates within a partition.
 * It sorts directly on 32-bit float scores and their associated indices as key-value pairs.
 * This is generally faster as it involves less data manipulation than the stable version.
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

// --- Template metaprogramming to select CUB storage type based on stability ---

// Type alias for stage1 temporary storage depending on stable/unstable
template <int kBlockSize, int kPartitionSize, bool Stable = kStableTopK>
struct Stage1StorageSelector;

template <int kBlockSize, int kPartitionSize>
struct Stage1StorageSelector<kBlockSize, kPartitionSize, true> {
  using type = typename cub::BlockRadixSort<
      uint64_t, kBlockSize, kPartitionSize / kBlockSize>::TempStorage;
};

template <int kBlockSize, int kPartitionSize>
struct Stage1StorageSelector<kBlockSize, kPartitionSize, false> {
  using type = typename cub::BlockRadixSort<
      float, kBlockSize, kPartitionSize / kBlockSize, int>::TempStorage;
};

template <int kBlockSize, int kPartitionSize>
using Stage1TempStorage =
    typename Stage1StorageSelector<kBlockSize, kPartitionSize>::type;

// Unified API wrapper that dispatches to the correct sort implementation at compile time.
template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK(const float* __restrict__ scores_in,
                                  int* __restrict__ intermediate_indices,
                                  float* __restrict__ intermediate_scores,
                                  int vocab_size,
                                  int num_partitions,
                                  TempStorage& temp_storage) {
  if constexpr (kStableTopK) {
    topk_common::FindPartitionTopK_StableSort<kBlockSize, kPartitionSize, K>(
        scores_in, intermediate_indices, intermediate_scores,
        vocab_size, num_partitions, temp_storage);
  } else {
    topk_common::FindPartitionTopK_UnstableSort<kBlockSize, kPartitionSize, K>(
        scores_in, intermediate_indices, intermediate_scores,
        vocab_size, num_partitions, temp_storage);
  }
}

/**
 * @brief Host/device helper to compute the next power of two for a given integer.
 * This is useful for bitonic sort, which requires a power-of-two input size.
 */
__device__ __host__ __forceinline__ constexpr int NextPowerOfTwo(int n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators
