// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>
#include <float.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_stable_sort_helper.cuh"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include <cooperative_groups.h>
#include <array>
#include <limits>
#include <algorithm>

namespace Generators {
namespace cuda {
namespace radix_partition_sort {

namespace cg = cooperative_groups;

// The primary reason for the kMaxPartitions = 64 limit is the use of cooperative groups, which requires that
// the entire grid of thread blocks can be resident on the GPU's Streaming Multiprocessors at the same time.
// Also, the number of registers each thread needs is directly proportional to this threshold,
// Current setting need CeilDiv(K_PADDED * kMaxPartitions, kBlockSize) = 16 registers per thread.
constexpr int kMaxPartitions = 64;

// Threshold of sort size to choose small kernel (bitonic sort in stage 2).
constexpr int kMaxSmallSortSize = 256;

// This partition sizes select as {11, 13, 16, 19} * 256.
constexpr std::array<int, 4> kSupportedPartitionSizes = {2816, 3328, 4096, 4864};

// --- Kernel for small sort size in stage 2 ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void PartitionSortSmallKernel(const float* __restrict__ scores_in,
                                         float* __restrict__ intermediate_scores,
                                         int* __restrict__ intermediate_indices,
                                         float* __restrict__ scores_out,
                                         int* __restrict__ indices_out,
                                         int vocab_size,
                                         int num_partitions,
                                         int k_actual) {  // Pass the real k
  cg::grid_group grid = cg::this_grid();

  constexpr int kSortSizeStage2 = K_PADDED * kMaxPartitionsForKernel;

  static_assert(kSortSizeStage2 <= kMaxSmallSortSize, "kSortSizeStage2 must be <= kMaxSmallSortSize");

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;
    struct {
      float scores[kMaxSmallSortSize];
      int indices[kMaxSmallSortSize];
    } stage2_bitonic_storage;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Parallel Partition Sort ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, Stage1TempStorageType>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2: Shared Memory Bitonic Sort for Smaller K ---
  if (blockIdx.x == 0) {
    const int batch_idx = blockIdx.y;
    const int num_elements_to_sort = num_partitions * K_PADDED;
    constexpr int kSortSize = topk_common::NextPowerOfTwo(kMaxPartitionsForKernel * K_PADDED);

    for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
      if (i < num_elements_to_sort) {
        smem.stage2_bitonic_storage.scores[i] = intermediate_scores[(size_t)batch_idx * num_elements_to_sort + i];
        smem.stage2_bitonic_storage.indices[i] = intermediate_indices[(size_t)batch_idx * num_elements_to_sort + i];
      } else {
        smem.stage2_bitonic_storage.scores[i] = -FLT_MAX;
        smem.stage2_bitonic_storage.indices[i] = INT_MAX;
      }
    }
    __syncthreads();

    bitonic_sort::SharedMemBitonicSort<kBlockSize, kSortSize>(smem.stage2_bitonic_storage.scores, smem.stage2_bitonic_storage.indices);

    if (threadIdx.x < k_actual) {
      size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
      scores_out[out_offset] = smem.stage2_bitonic_storage.scores[threadIdx.x];
      indices_out[out_offset] = smem.stage2_bitonic_storage.indices[threadIdx.x];
    }
  }
}

// --- Kernel for large sort size in stage 2 ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void PartitionSortLargeKernel(const float* __restrict__ scores_in,
                                         float* __restrict__ intermediate_scores,
                                         int* __restrict__ intermediate_indices,
                                         float* __restrict__ scores_out,
                                         int* __restrict__ indices_out,
                                         int vocab_size,
                                         int num_partitions,
                                         int k_actual) {  // Pass the real k
  cg::grid_group grid = cg::this_grid();

  constexpr int kSortSizeStage2 = K_PADDED * kMaxPartitionsForKernel;
  constexpr int kItemsPerThreadStage2 = CeilDiv(kSortSizeStage2, kBlockSize);

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;
#ifdef STABLE_TOPK
    typename cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThreadStage2>::TempStorage stage2_radix_storage;
#else
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThreadStage2, int>::TempStorage stage2_radix_storage;
#endif
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Parallel Partition Sort ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, Stage1TempStorageType>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2: CUB Block Radix Sort for Larger K  ---
  if (blockIdx.x == 0) {
    const int batch_idx = blockIdx.y;
    const int num_elements_to_sort = num_partitions * K_PADDED;

#ifdef STABLE_TOPK
    uint64_t thread_keys[kItemsPerThreadStage2];
    for (int i = 0; i < kItemsPerThreadStage2; ++i) {
      int load_idx = threadIdx.x * kItemsPerThreadStage2 + i;
      if (load_idx < num_elements_to_sort) {
        size_t offset = (size_t)batch_idx * num_elements_to_sort + load_idx;
        thread_keys[i] = topk_common::PackStableSortKey(intermediate_scores[offset], intermediate_indices[offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThreadStage2>(smem.stage2_radix_storage).SortDescendingBlockedToStriped(thread_keys);
    if (threadIdx.x < k_actual) {
      size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
      scores_out[out_offset] = topk_common::UnpackStableSortScore(thread_keys[0]);
      indices_out[out_offset] = topk_common::UnpackStableSortIndex(thread_keys[0]);
    }
#else
    float thread_scores[kItemsPerThreadStage2];
    int thread_indices[kItemsPerThreadStage2];
    for (int i = 0; i < kItemsPerThreadStage2; ++i) {
      int load_idx = threadIdx.x * kItemsPerThreadStage2 + i;
      if (load_idx < num_elements_to_sort) {
        size_t offset = (size_t)batch_idx * num_elements_to_sort + load_idx;
        thread_scores[i] = intermediate_scores[offset];
        thread_indices[i] = intermediate_indices[offset];
      } else {
        thread_scores[i] = -FLT_MAX;
        thread_indices[i] = INT_MAX;
      }
    }
    cub::BlockRadixSort<float, kBlockSize, kItemsPerThreadStage2, int>(smem.stage2_radix_storage).SortDescendingBlockedToStriped(thread_scores, thread_indices);
    if (threadIdx.x < k_actual) {
      size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
      scores_out[out_offset] = thread_scores[0];
      indices_out[out_offset] = thread_indices[0];
    }
#endif
  }
}

// --- Host-side Launcher ---

inline int EstimateBestPartitionSize(int vocab_size) {
  constexpr std::array<int, 4> kPartitionTargets = {8, 16, 32, 64};
  int best_partition_size = 0;
  // Use a large initial cost. Cost is a measure of how far away the number of partitions is
  // from an ideal target (16, 32, 64), plus a penalty for a less balanced workload.
  double min_cost = std::numeric_limits<double>::max();

  for (int p_size : kSupportedPartitionSizes) {
    const int num_partitions = CeilDiv(vocab_size, p_size);
    if (num_partitions > kMaxPartitions) {
      continue;
    }

    // Find the smallest target that is >= num_partitions
    auto it = std::lower_bound(kPartitionTargets.begin(), kPartitionTargets.end(), num_partitions);
    if (it != kPartitionTargets.end()) {
      int target_partitions = *it;
      // Cost is the distance to the target. A perfect match has a cost of 0.
      double cost = target_partitions - num_partitions;

      // Add a small penalty for larger partition sizes to favor more balanced workloads
      // where Stage 1 and Stage 2 sizes are closer.
      // The following optimizes most common use case of k = 50.
      // Note that we can use online benchmark to find best parition size for small k value in the future.
      constexpr int default_padded_k = 52;
      int stage2_size = target_partitions * default_padded_k;
      if (stage2_size > p_size) {
        cost += static_cast<double>((stage2_size - p_size)) / p_size;
      }

      if (cost < min_cost) {
        min_cost = cost;
        best_partition_size = p_size;
      }
    }
  }

  if (best_partition_size == 0) {
    // This vocab_size is too large. Returns a dummy value here. Note that IsSupported will reject it later.
    best_partition_size = kSupportedPartitionSizes[0];
  }

  return best_partition_size;
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kPartitionSortMaxK;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
void* ChoosePartitionSortKernel() {
  void* kernel;
  if constexpr (K_PADDED * kMaxPartitionsForKernel <= kMaxSmallSortSize)
    kernel = (void*)PartitionSortSmallKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
  else
    kernel = (void*)PartitionSortLargeKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
  return kernel;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED>
void* GetPartitionSortKernel(int num_partitions) {
  void* kernel;
  if (num_partitions <= 8) {
    kernel = ChoosePartitionSortKernel<kBlockSize, kPartitionSize, K_PADDED, 8>();
  } else if (num_partitions <= 16) {
    kernel = ChoosePartitionSortKernel<kBlockSize, kPartitionSize, K_PADDED, 16>();
  } else if (num_partitions <= 32) {
    kernel = ChoosePartitionSortKernel<kBlockSize, kPartitionSize, K_PADDED, 32>();
  } else {
    kernel = ChoosePartitionSortKernel<kBlockSize, kPartitionSize, K_PADDED, 64>();
  }
  return kernel;
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  const int partition_size = data->partition_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  auto launch_cooperative_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int kBlockSize = 256;
    dim3 grid(num_partitions, batch_size);
    dim3 block(kBlockSize);

    auto launch_with_partition_size_args = [&](auto p_size) {
      constexpr int P_SIZE = decltype(p_size)::value;
      void* kernel = GetPartitionSortKernel<kBlockSize, P_SIZE, K_PADDED>(num_partitions);
      void* kernel_args[] = {(void*)&scores_in, (void*)&data->intermediate_scores_1, (void*)&data->intermediate_indices_1,
                             (void*)&data->intermediate_scores_2, (void*)&data->intermediate_indices_2,
                             (void*)&vocab_size, (void*)&num_partitions, (void*)&k};
      CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, kernel_args, 0, stream));
    };

    if (partition_size == kSupportedPartitionSizes[0])
      launch_with_partition_size_args(std::integral_constant<int, kSupportedPartitionSizes[0]>());
    else if (partition_size == kSupportedPartitionSizes[1])
      launch_with_partition_size_args(std::integral_constant<int, kSupportedPartitionSizes[1]>());
    else if (partition_size == kSupportedPartitionSizes[2])
      launch_with_partition_size_args(std::integral_constant<int, kSupportedPartitionSizes[2]>());
    else
      launch_with_partition_size_args(std::integral_constant<int, kSupportedPartitionSizes[3]>());
  };

  if (k <= 4)
    launch_cooperative_kernel(std::integral_constant<int, 4>());
  else if (k <= 8)
    launch_cooperative_kernel(std::integral_constant<int, 8>());
  else if (k <= 16)
    launch_cooperative_kernel(std::integral_constant<int, 16>());
  else if (k <= 32)
    launch_cooperative_kernel(std::integral_constant<int, 32>());
  else if (k <= 52)
    launch_cooperative_kernel(std::integral_constant<int, 52>());
  else
    launch_cooperative_kernel(std::integral_constant<int, 64>());

  CUDA_CHECK_LAUNCH();

  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k;
}

// --- The following implements IsSupported ---
template <int K_PADDED, int kMaxPartitionsForKernel>
bool CheckPartitionSortSupport(int batch_size, int num_partitions, int partition_size) {
  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;
  void* kernel;
  if (partition_size == kSupportedPartitionSizes[0])
    kernel = ChoosePartitionSortKernel<kBlockSize, kSupportedPartitionSizes[0], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kSupportedPartitionSizes[1])
    kernel = ChoosePartitionSortKernel<kBlockSize, kSupportedPartitionSizes[1], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kSupportedPartitionSizes[2])
    kernel = ChoosePartitionSortKernel<kBlockSize, kSupportedPartitionSizes[2], K_PADDED, kMaxPartitionsForKernel>();
  else
    kernel = ChoosePartitionSortKernel<kBlockSize, kSupportedPartitionSizes[3], K_PADDED, kMaxPartitionsForKernel>();

  int device, num_sm, max_blocks_per_sm;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kBlockSize, 0));
  return total_blocks <= (num_sm * max_blocks_per_sm);
}

template <int K_PADDED>
bool IsSupportedDispatch(int batch_size, int vocab_size, int k, int partition_size, int num_partitions) {
  if (num_partitions <= 8) {
    return CheckPartitionSortSupport<K_PADDED, 8>(batch_size, num_partitions, partition_size);
  } else if (num_partitions <= 16) {
    return CheckPartitionSortSupport<K_PADDED, 16>(batch_size, num_partitions, partition_size);
  } else if (num_partitions <= 32) {
    return CheckPartitionSortSupport<K_PADDED, 32>(batch_size, num_partitions, partition_size);
  }
  return CheckPartitionSortSupport<K_PADDED, 64>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kPartitionSortMaxK) return false;
  int coop_support = 0;
  cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, 0);
  if (!coop_support) return false;
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  if (partition_size == 0) return false;
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) return false;

  if (k <= 4) return IsSupportedDispatch<4>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k <= 8) return IsSupportedDispatch<8>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k <= 16) return IsSupportedDispatch<16>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k <= 32) return IsSupportedDispatch<32>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k <= 52) return IsSupportedDispatch<52>(batch_size, vocab_size, k, partition_size, num_partitions);
  return IsSupportedDispatch<64>(batch_size, vocab_size, k, partition_size, num_partitions);
}

}  // namespace radix_partition_sort
}  // namespace cuda
}  // namespace Generators
