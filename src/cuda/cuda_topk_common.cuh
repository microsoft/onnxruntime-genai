// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cuda_runtime.h>
#include <float.h>
#include "cuda_topk.h"
#include "cuda_topk_stable_sort_helper.cuh"
#include "cuda_topk_warp_sort_helper.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace topk_common {

// A simple greater-than comparator for descending sort
struct DescendingOp {
  template <typename T>
  __device__ __host__ bool operator()(const T& a, const T& b) const {
    return a > b;
  }
};

/**
 * @brief Finds the Top-K candidates within a single data partition.
 * This is the core workhorse for Stage 1 of all partition-based algorithms.
 * It is templated on `UseMergeSort` to allow the host to select the best
 * internal sorting algorithm based on benchmark data.
 */
template <int kBlockSize, int kPartitionSize, int K, bool UseMergeSort, typename TempStorage>
__device__ void FindPartitionTopK(const float* __restrict__ scores_in,
                                  int* __restrict__ intermediate_indices,
                                  float* __restrict__ intermediate_scores,
                                  int vocab_size,
                                  int num_partitions,
                                  TempStorage& temp_storage);

// --- Implementations using Merge Sort (for smaller partitions) ---

template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_StableSort_Merge(const float* __restrict__ scores_in,
                                                   int* __restrict__ intermediate_indices,
                                                   float* __restrict__ intermediate_scores,
                                                   int vocab_size,
                                                   int num_partitions,
                                                   TempStorage& temp_storage) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using SortKeyT = uint64_t;
  using SortValueT = cub::NullType;
  using BlockMergeSort = cub::BlockMergeSort<SortKeyT, kBlockSize, ItemsPerThread, SortValueT>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  SortKeyT thread_keys[ItemsPerThread];
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x * ItemsPerThread + i;
    if (global_idx < vocab_size) {
      thread_keys[i] = topk_common::PackStableSortKey(batch_scores_in[global_idx], global_idx);
    } else {
      thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
    }
  }

  BlockMergeSort(temp_storage.merge_storage).Sort(thread_keys, DescendingOp());

  // Unpack and store
  float thread_scores_out[ItemsPerThread];
  int thread_indices_out[ItemsPerThread];
  for (int i = 0; i < ItemsPerThread; ++i) {
    thread_scores_out[i] = topk_common::UnpackStableSortScore(thread_keys[i]);
    thread_indices_out[i] = topk_common::UnpackStableSortIndex(thread_keys[i]);
  }
  size_t base_offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K;
  cub::StoreDirectBlocked(threadIdx.x, intermediate_scores + base_offset, thread_scores_out, K);
  cub::StoreDirectBlocked(threadIdx.x, intermediate_indices + base_offset, thread_indices_out, K);
}

template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_UnstableSort_Merge(const float* __restrict__ scores_in,
                                                     int* __restrict__ intermediate_indices,
                                                     float* __restrict__ intermediate_scores,
                                                     int vocab_size,
                                                     int num_partitions,
                                                     TempStorage& temp_storage) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using BlockMergeSort = cub::BlockMergeSort<float, kBlockSize, ItemsPerThread, int>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x * ItemsPerThread + i;
    if (global_idx < vocab_size) {
      thread_keys[i] = batch_scores_in[global_idx];
      thread_values[i] = global_idx;
    } else {
      thread_keys[i] = -FLT_MAX;
      thread_values[i] = INT_MAX;
    }
  }

  BlockMergeSort(temp_storage.merge_storage).Sort(thread_keys, thread_values, DescendingOp());

  size_t base_offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K;
  cub::StoreDirectBlocked(threadIdx.x, intermediate_scores + base_offset, thread_keys, K);
  cub::StoreDirectBlocked(threadIdx.x, intermediate_indices + base_offset, thread_values, K);
}

// --- Implementations using Radix Sort (for larger partitions) ---

template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_StableSort_Radix(const float* __restrict__ scores_in,
                                                   int* __restrict__ intermediate_indices,
                                                   float* __restrict__ intermediate_scores,
                                                   int vocab_size,
                                                   int num_partitions,
                                                   TempStorage& temp_storage) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using CompositeKey = uint64_t;
  using BlockRadixSort = cub::BlockRadixSort<CompositeKey, kBlockSize, ItemsPerThread>;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;
  CompositeKey thread_keys[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size) {
      thread_keys[i] = topk_common::PackStableSortKey(batch_scores_in[global_idx], global_idx);
    } else {
      thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
    }
  }

  BlockRadixSort(temp_storage.radix_storage).SortDescendingBlockedToStriped(thread_keys);

  if (threadIdx.x < K) {
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = topk_common::UnpackStableSortScore(thread_keys[0]);
    intermediate_indices[offset] = topk_common::UnpackStableSortIndex(thread_keys[0]);
  }
}

template <int kBlockSize, int kPartitionSize, int K, typename TempStorage>
__device__ void FindPartitionTopK_UnstableSort_Radix(const float* __restrict__ scores_in,
                                                     int* __restrict__ intermediate_indices,
                                                     float* __restrict__ intermediate_scores,
                                                     int vocab_size,
                                                     int num_partitions,
                                                     TempStorage& temp_storage) {
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

  BlockRadixSort(temp_storage.radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  if (threadIdx.x < K) {
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K + threadIdx.x;
    intermediate_scores[offset] = thread_keys[0];
    intermediate_indices[offset] = thread_values[0];
  }
}

// --- Metaprogramming for Storage and Dispatch ---

template <int kBlockSize, int kPartitionSize, bool Stable = kStableTopK>
struct Stage1StorageSelector;

template <int kBlockSize, int kPartitionSize>
struct Stage1StorageSelector<kBlockSize, kPartitionSize, true> {
  using RadixStorage = typename cub::BlockRadixSort<uint64_t, kBlockSize, kPartitionSize / kBlockSize>::TempStorage;
  using MergeStorage = typename cub::BlockMergeSort<uint64_t, kBlockSize, kPartitionSize / kBlockSize, cub::NullType>::TempStorage;
  union type {
    RadixStorage radix_storage;
    MergeStorage merge_storage;
  };
};

template <int kBlockSize, int kPartitionSize>
struct Stage1StorageSelector<kBlockSize, kPartitionSize, false> {
  using RadixStorage = typename cub::BlockRadixSort<float, kBlockSize, kPartitionSize / kBlockSize, int>::TempStorage;
  using MergeStorage = typename cub::BlockMergeSort<float, kBlockSize, kPartitionSize / kBlockSize, int>::TempStorage;
  union type {
    RadixStorage radix_storage;
    MergeStorage merge_storage;
  };
};

template <int kBlockSize, int kPartitionSize>
using Stage1TempStorage = typename Stage1StorageSelector<kBlockSize, kPartitionSize>::type;

template <int kBlockSize, int kPartitionSize, int K, bool UseMergeSort, typename TempStorage>
__device__ void FindPartitionTopK(const float* __restrict__ scores_in,
                                  int* __restrict__ intermediate_indices,
                                  float* __restrict__ intermediate_scores,
                                  int vocab_size,
                                  int num_partitions,
                                  TempStorage& temp_storage) {
  if constexpr (UseMergeSort) {
    if constexpr (kStableTopK) {
      FindPartitionTopK_StableSort_Merge<kBlockSize, kPartitionSize, K>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, temp_storage);
    } else {
      FindPartitionTopK_UnstableSort_Merge<kBlockSize, kPartitionSize, K>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, temp_storage);
    }
  } else {
    if constexpr (kStableTopK) {
      FindPartitionTopK_StableSort_Radix<kBlockSize, kPartitionSize, K>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, temp_storage);
    } else {
      FindPartitionTopK_UnstableSort_Radix<kBlockSize, kPartitionSize, K>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, temp_storage);
    }
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

__device__ __host__ __forceinline__ constexpr int Log2NextPowerOfTwo(int n) {
  if (n <= 1) return 0;
#if defined(__CUDA_ARCH__)
  int x = NextPowerOfTwo(n);
  return 31 - __clz(x);
#else
  int x = NextPowerOfTwo(n);
  int log2 = 0;
  while (x >>= 1) ++log2;
  return log2;
#endif
}

/**
 * @brief A unified, benchmark-driven helper for performing a reduction (merge) step.
 * It loads candidate data from N partitions, sorts them, and writes the top K back to global memory.
 * The internal sorting algorithm is selected at compile time based on kSortSize.
 */
template <int kBlockSize, int kSortSize, int K_PADDED, int kItemsPerThread, typename TempStorage>
__device__ void BlockReduceTopK(const float* scores_in_batch,
                                const int* indices_in_batch,
                                float* scores_out_batch,
                                int* indices_out_batch,
                                int num_elements_to_sort,
                                int first_child_partition,
                                int partition_idx,
                                TempStorage& smem) {
  // constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  constexpr int kSortSizePo2 = NextPowerOfTwo(kSortSize);

  // This unified helper selects the sort algorithm based on the compile-time kSortSize,
  // consistent with the constraints applied in hybrid_sort and the micro-benchmark.
  if constexpr (kSortSize <= BestAlgoThresholds::kWarpBitonic_MaxSize) {
    // --- 1. Warp Bitonic Sort ---
    // Load to shared memory first
    for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
      if (i < num_elements_to_sort) {
        int part_idx = i / K_PADDED;
        int element_idx = i % K_PADDED;
        size_t local_offset = static_cast<size_t>(first_child_partition + part_idx) * K_PADDED + element_idx;
        smem.stage2_storage.scores[i] = scores_in_batch[local_offset];
        smem.stage2_storage.indices[i] = indices_in_batch[local_offset];
      }
    }
    __syncthreads();

    // Have the first warp sort from shared memory using registers
    if (threadIdx.x < warpSize) {
      float my_score = (threadIdx.x < num_elements_to_sort) ? smem.stage2_storage.scores[threadIdx.x] : -FLT_MAX;
      int my_index = (threadIdx.x < num_elements_to_sort) ? smem.stage2_storage.indices[threadIdx.x] : INT_MAX;
      topk_common::WarpBitonicSort(my_score, my_index);
      if (threadIdx.x < K_PADDED) {
        smem.stage2_storage.scores[threadIdx.x] = my_score;
        smem.stage2_storage.indices[threadIdx.x] = my_index;
      }
    }
  } else if constexpr (kSortSize <= BestAlgoThresholds::kCubWarpMerge_MaxSize) {
    // --- 2. CUB Warp Merge Sort ---
    for (int i = threadIdx.x; i < kSortSizePo2; i += kBlockSize) {
      if (i < num_elements_to_sort) {
        int part_idx = i / K_PADDED;
        int element_idx = i % K_PADDED;
        size_t local_offset = static_cast<size_t>(first_child_partition + part_idx) * K_PADDED + element_idx;
        smem.stage2_storage.scores[i] = scores_in_batch[local_offset];
        smem.stage2_storage.indices[i] = indices_in_batch[local_offset];
      } else {
        smem.stage2_storage.scores[i] = -FLT_MAX;
        smem.stage2_storage.indices[i] = INT_MAX;
      }
    }
    __syncthreads();
    topk_common::WarpMergeSort<kSortSizePo2>(smem.stage2_storage.scores, smem.stage2_storage.indices, &smem.cub_warp_storage, num_elements_to_sort);
  } else {
    // --- 3. CUB Block Merge Sort ---
#ifdef STABLE_TOPK
    using SortKeyT = uint64_t;
    SortKeyT thread_keys[kItemsPerThread];
    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int part_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t local_offset = static_cast<size_t>(first_child_partition + part_idx) * K_PADDED + element_idx;
        thread_keys[i] = topk_common::PackStableSortKey(scores_in_batch[local_offset], indices_in_batch[local_offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, cub::NullType>(smem.cub_block_merge_storage).Sort(thread_keys, topk_common::DescendingOp());

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + item_idx;
        scores_out_batch[out_offset] = topk_common::UnpackStableSortScore(thread_keys[i]);
        indices_out_batch[out_offset] = topk_common::UnpackStableSortIndex(thread_keys[i]);
      }
    }
    return;  // Early exit to avoid double-write
#else
    float thread_keys[kItemsPerThread];
    int thread_values[kItemsPerThread];
    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < num_elements_to_sort) {
        int part_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t local_offset = static_cast<size_t>(first_child_partition + part_idx) * K_PADDED + element_idx;
        thread_keys[i] = scores_in_batch[local_offset];
        thread_values[i] = indices_in_batch[local_offset];
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = INT_MAX;
      }
    }
    cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>(smem.cub_block_merge_storage).Sort(thread_keys, thread_values, topk_common::DescendingOp());
    cub::StoreDirectBlocked(threadIdx.x, scores_out_batch + static_cast<size_t>(partition_idx) * K_PADDED, thread_keys, K_PADDED);
    cub::StoreDirectBlocked(threadIdx.x, indices_out_batch + static_cast<size_t>(partition_idx) * K_PADDED, thread_values, K_PADDED);
    return;  // Early exit to avoid double-write
#endif
  }
  __syncthreads();

  // Final write to global memory for warp-based and smem-based sorts
  if constexpr (kSortSize <= BestAlgoThresholds::kCubWarpMerge_MaxSize) {
    if (threadIdx.x < K_PADDED) {
      size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      scores_out_batch[out_offset] = smem.stage2_storage.scores[threadIdx.x];
      indices_out_batch[out_offset] = smem.stage2_storage.indices[threadIdx.x];
    }
  }
}

inline bool IsSupportedCooperative(void* kernel, int total_blocks, int block_size=256, int device_id = -1) {
  if (device_id < 0) {
    CUDA_CHECK(cudaGetDevice(&device_id));
  }

  int cooperative_launch_support = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, device_id));
  if (!cooperative_launch_support) {
    return false;
  }

  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_id));
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));
  int max_active_blocks = num_sm * max_blocks_per_sm;

  if (total_blocks > max_active_blocks) {
    return false;
  }

  return true;
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators
