// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_warp_sort_helper.cuh"
#include <cooperative_groups.h>
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace flash_convergent {

/**
 * @brief A two-stage cooperative algorithm specialized for **small to medium k**.
 *
 * Algorithm Overview:
 * This kernel is designed to have very low overhead by performing the entire reduction
 * in a single step, making it dominant where the number of candidates is manageable.
 *
 * 1.  **Stage 1 (Partition Top-K)**: All thread blocks find top candidates in parallel.
 *
 * 2.  **Grid-Wide Sync**.
 *
 * 3.  **Stage 2 (Single-Step Reduction)**: A single block (`blockIdx.x == 0`) performs the final merge.
 * It loads all candidates and sorts them in one go using the fastest CUB block-level
 * algorithm (Merge or Radix) as determined by pre-computed benchmarks for the total sort size.
 *
 * Performance Characteristics:
 * -   **Strengths**: High performance for small and medium `k` where the total number of
 * candidates (`k * num_partitions`) is less than ~1024, as it avoids iterative overhead.
 * -   **Weaknesses**: Requires cooperative launch. Performance degrades for very large `k`
 * as the final sort becomes very large and slow.
 */
namespace cg = cooperative_groups;

constexpr int kMaxPartitions = 64;
constexpr std::array<int, 4> kPartitionSizes = {2816, 3328, 4096, 4864};

// --- Metaprogramming to select the correct SharedStorage type ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel, SortAlgo kSortAlgo>
struct SharedStorageSelector;

// Specialization for CUB_BLOCK_MERGE
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
struct SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, SortAlgo::CUB_BLOCK_MERGE> {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  static constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  static constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  union type {
    Stage1TempStorageType stage1_storage;
#ifdef STABLE_TOPK
    typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage merge_storage;
#else
    typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage merge_storage;
#endif
  };
};

// Specialization for CUB_BLOCK_RADIX
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
struct SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, SortAlgo::CUB_BLOCK_RADIX> {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  static constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;
  static constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
  union type {
    Stage1TempStorageType stage1_storage;
#ifdef STABLE_TOPK
    typename cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThread>::TempStorage radix_storage;
#else
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>::TempStorage radix_storage;
#endif
  };
};

// --- Unified Convergent Kernel ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void FlashConvergentKernel(const float* __restrict__ scores_in,
                                      float* __restrict__ intermediate_scores,
                                      int* __restrict__ intermediate_indices,
                                      float* __restrict__ scores_out,
                                      int* __restrict__ indices_out,
                                      int vocab_size,
                                      int num_partitions,
                                      int k_actual) {
  cg::grid_group grid = cg::this_grid();
  constexpr int kSortSize = K_PADDED * kMaxPartitionsForKernel;

  // For the sort sizes handled by this kernel's final reduction, the best algorithm
  // will always be a block-level one, not a warp-level one.
  constexpr SortAlgo kSortAlgo = (GetBestAlgo(kSortSize) == SortAlgo::WARP_BITONIC || GetBestAlgo(kSortSize) == SortAlgo::CUB_WARP_MERGE)
                                     ? SortAlgo::CUB_BLOCK_MERGE
                                     : GetBestAlgo(kSortSize);

  using SharedStorage = typename SharedStorageSelector<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel, kSortAlgo>::type;
  __shared__ SharedStorage smem;

  // --- Stage 1: Parallel Partition Sort ---
  constexpr bool kUseMergeSortInStage1 = kPartitionSize <= BestAlgoThresholds::kCubBlockMerge_MaxSize;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, kUseMergeSortInStage1>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2: One block performs the final merge ---
  if (blockIdx.x == 0) {
    const int batch_idx = blockIdx.y;
    const int num_elements_to_sort = num_partitions * K_PADDED;
    constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);

#ifdef STABLE_TOPK
    using SortKeyT = uint64_t;
    SortKeyT thread_keys[kItemsPerThread];
    for (int i = 0; i < kItemsPerThread; ++i) {
      int load_idx = threadIdx.x + i * kBlockSize;
      if (load_idx < num_elements_to_sort) {
        size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + load_idx;
        thread_keys[i] = topk_common::PackStableSortKey(intermediate_scores[offset], intermediate_indices[offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    if constexpr (kSortAlgo == SortAlgo::CUB_BLOCK_RADIX) {
      cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread>(smem.radix_storage).SortDescendingBlockedToStriped(thread_keys);
      if (threadIdx.x < k_actual) {
        size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
        scores_out[out_offset] = topk_common::UnpackStableSortScore(thread_keys[0]);
        indices_out[out_offset] = topk_common::UnpackStableSortIndex(thread_keys[0]);
      }
    } else {  // CUB_BLOCK_MERGE
      cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, cub::NullType>(smem.merge_storage).Sort(thread_keys, topk_common::DescendingOp());
      float thread_scores_out[kItemsPerThread];
      int thread_indices_out[kItemsPerThread];
      for (int i = 0; i < kItemsPerThread; ++i) {
        thread_scores_out[i] = topk_common::UnpackStableSortScore(thread_keys[i]);
        thread_indices_out[i] = topk_common::UnpackStableSortIndex(thread_keys[i]);
      }
      cub::StoreDirectBlocked(threadIdx.x, scores_out + (size_t)batch_idx * k_actual, thread_scores_out, k_actual);
      cub::StoreDirectBlocked(threadIdx.x, indices_out + (size_t)batch_idx * k_actual, thread_indices_out, k_actual);
    }
#else
    using SortKeyT = float;
    using SortValueT = int;
    SortKeyT thread_keys[kItemsPerThread];
    SortValueT thread_values[kItemsPerThread];
    for (int i = 0; i < kItemsPerThread; ++i) {
      int load_idx = threadIdx.x + i * kBlockSize;
      if (load_idx < num_elements_to_sort) {
        size_t offset = (size_t)batch_idx * num_partitions * K_PADDED + load_idx;
        thread_keys[i] = intermediate_scores[offset];
        thread_values[i] = intermediate_indices[offset];
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = INT_MAX;
      }
    }
    if constexpr (kSortAlgo == SortAlgo::CUB_BLOCK_RADIX) {
      cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.radix_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
      if (threadIdx.x < k_actual) {
        size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
        scores_out[out_offset] = thread_keys[0];
        indices_out[out_offset] = thread_values[0];
      }
    } else {  // CUB_BLOCK_MERGE
      cub::BlockMergeSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem.merge_storage).Sort(thread_keys, thread_values, topk_common::DescendingOp());
      cub::StoreDirectBlocked(threadIdx.x, scores_out + (size_t)batch_idx * k_actual, thread_keys, k_actual);
      cub::StoreDirectBlocked(threadIdx.x, indices_out + (size_t)batch_idx * k_actual, thread_values, k_actual);
    }
#endif
  }
}

// --- Host-side Launcher ---

inline int EstimateBestPartitionSize(int vocab_size, int k) {
  const auto& benchmarks = GetSortBenchmarkResults();
  double min_total_latency = std::numeric_limits<double>::max();
  int best_partition_size = 0;

  for (int p_size : kPartitionSizes) {
    const int num_partitions = CeilDiv(vocab_size, p_size);
    if (num_partitions > kMaxPartitions) {
      continue;
    }

    SortAlgo best_algo_s1 = GetBestAlgo(p_size);
    float latency_s1 = benchmarks.GetLatency(best_algo_s1, p_size);
    int sort_size_s2 = kConvergentSortMaxK * num_partitions;
    SortAlgo best_algo_s2 = GetBestAlgo(sort_size_s2);
    float latency_s2 = benchmarks.GetLatency(best_algo_s2, sort_size_s2);
    float total_latency = latency_s1 + latency_s2;

    if (total_latency < min_total_latency) {
      min_total_latency = total_latency;
      best_partition_size = p_size;
    }
  }
  return (best_partition_size == 0) ? kPartitionSizes[0] : best_partition_size;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
void* GetKernel() {
  return (void*)FlashConvergentKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED>
void* GetKernelForNumPartitions(int num_partitions) {
  if (num_partitions <= 8) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 8>();
  if (num_partitions <= 16) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 16>();
  if (num_partitions <= 32) return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 32>();
  return GetKernel<kBlockSize, kPartitionSize, K_PADDED, 64>();
}

template <int P_SIZE, int K_PADDED>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  int num_partitions = CeilDiv(vocab_size, P_SIZE);
  dim3 grid(num_partitions, batch_size);
  dim3 block(kBlockSize);
  void* kernel = GetKernelForNumPartitions<kBlockSize, P_SIZE, K_PADDED>(num_partitions);
  void* kernel_args[] = {(void*)&scores_in, (void*)&data->intermediate_scores_1, (void*)&data->intermediate_indices_1,
                         (void*)&data->intermediate_scores_2, (void*)&data->intermediate_indices_2,
                         (void*)&vocab_size, (void*)&num_partitions, (void*)&k};
  CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, kernel_args, 0, stream));
}

template <int K_PADDED>
void LaunchKernelByPartitionSize(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size) {
  if (partition_size == kPartitionSizes[0]) LaunchKernel<kPartitionSizes[0], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kPartitionSizes[1]) LaunchKernel<kPartitionSizes[1], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kPartitionSizes[2]) LaunchKernel<kPartitionSizes[2], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else LaunchKernel<kPartitionSizes[3], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));
  if (data->flash_convergent_partition_size_k != k) {
    data->flash_convergent_partition_size_k = k;
    data->flash_convergent_partition_size = EstimateBestPartitionSize(vocab_size, k);
  }
  const int partition_size = data->flash_convergent_partition_size;
  if (k <= 4) LaunchKernelByPartitionSize<4>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 8) LaunchKernelByPartitionSize<8>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 16) LaunchKernelByPartitionSize<16>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 32) LaunchKernelByPartitionSize<32>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 52) LaunchKernelByPartitionSize<52>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else LaunchKernelByPartitionSize<64>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  CUDA_CHECK_LAUNCH();
  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k;
}

template <int K_PADDED, int kMaxPartitionsForKernel>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;
  void* kernel;
  if (partition_size == kPartitionSizes[0]) kernel = GetKernel<kBlockSize, kPartitionSizes[0], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kPartitionSizes[1]) kernel = GetKernel<kBlockSize, kPartitionSizes[1], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kPartitionSizes[2]) kernel = GetKernel<kBlockSize, kPartitionSizes[2], K_PADDED, kMaxPartitionsForKernel>();
  else kernel = GetKernel<kBlockSize, kPartitionSizes[3], K_PADDED, kMaxPartitionsForKernel>();
  return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

template <int K_PADDED>
bool IsSupportedDispatch(int batch_size, int partition_size, int num_partitions) {
  if (num_partitions <= 8) return CheckSupport<K_PADDED, 8>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 16) return CheckSupport<K_PADDED, 16>(batch_size, num_partitions, partition_size);
  if (num_partitions <= 32) return CheckSupport<K_PADDED, 32>(batch_size, num_partitions, partition_size);
  return CheckSupport<K_PADDED, 64>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kConvergentSortMaxK) return false;
  const int partition_size = EstimateBestPartitionSize(vocab_size, k);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) return false;
  if (k <= 4) return IsSupportedDispatch<4>(batch_size, partition_size, num_partitions);
  if (k <= 8) return IsSupportedDispatch<8>(batch_size, partition_size, num_partitions);
  if (k <= 16) return IsSupportedDispatch<16>(batch_size, partition_size, num_partitions);
  if (k <= 32) return IsSupportedDispatch<32>(batch_size, partition_size, num_partitions);
  if (k <= 52) return IsSupportedDispatch<52>(batch_size, partition_size, num_partitions);
  return IsSupportedDispatch<64>(batch_size, partition_size, num_partitions);
}

}  // namespace flash_convergent
}  // namespace cuda
}  // namespace Generators

