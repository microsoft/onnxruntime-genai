// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include <cooperative_groups.h>

namespace Generators {
namespace cuda {
namespace flash_convergent {

/**
 * @brief A two-stage cooperative algorithm with a single-step reduction phase.
 *
 * Algorithm Overview:
 * This algorithm uses cooperative groups to perform the entire Top-K operation in
 * a single kernel launch, but with a different reduction strategy than `iterative_sort`.
 *
 * 1.  **Stage 1 (Partition Top-K)**: All thread blocks work in parallel to find the
 * top `K_PADDED` candidates from their assigned vocabulary partitions using
 * `topk_common::FindPartitionTopK`. The results are written to a global
 * intermediate buffer.
 *
 * 2.  **Grid-Wide Sync**: A `cg::grid_group::sync()` ensures all partitions are processed.
 *
 * 3.  **Stage 2 (Single-Step Reduction)**: A single, specialized thread block (`blockIdx.x == 0`)
 * is responsible for the final merge. It loads all candidates from the intermediate
 * buffer and performs a final, large sort to find the global Top-K.
 *
 * Performance Characteristics:
 * -   **Strengths**: By performing the final reduction in a single step, it avoids the overhead of
 * iterative loops and multiple grid-wide synchronizations found in other cooperative methods.
 * It intelligently switches its internal sorting method: for smaller total candidate sets
 * (`<= kMaxSmallSortSize`), it uses a fast shared-memory bitonic sort; for larger sets,
 * it uses the more powerful `cub::BlockRadixSort`.
 * -   **Weaknesses**: Requires cooperative launch support. Its primary limitation is the total
 * number of partitions (`kMaxPartitions`), as a single block must be able to load and sort
 * all candidates. This makes it less scalable for extremely large vocabularies that would
 * require many partitions.
 */
namespace cg = cooperative_groups;

// The limit on partitions is due to cooperative group residency requirements and the
// fact that a single block must sort all `k * num_partitions` candidates in Stage 2.
constexpr int kMaxPartitions = 64;

// Threshold for switching between bitonic sort and CUB radix sort in Stage 2.
constexpr int kMaxSmallSortSize = 256;

// This partition sizes select as {11, 13, 16, 19} * 256.
constexpr std::array<int, 4> kSupportedPartitionSizes = {2816, 3328, 4096, 4864};

// --- Kernel for small Stage 2 sort size (using Bitonic Sort) ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void FlashSortSmallKernel(const float* __restrict__ scores_in,
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

  // --- Stage 2: One block performs the final merge ---
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

    // Write the final top-k results to global memory.
    if (threadIdx.x < k_actual) {
      size_t out_offset = static_cast<size_t>(batch_idx) * k_actual + threadIdx.x;
      scores_out[out_offset] = smem.stage2_bitonic_storage.scores[threadIdx.x];
      indices_out[out_offset] = smem.stage2_bitonic_storage.indices[threadIdx.x];
    }
  }
}

// --- Kernel for large Stage 2 sort size (using CUB Radix Sort) ---
template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
__global__ void FlashSortLargeKernel(const float* __restrict__ scores_in,
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

  // --- Stage 2: One block performs the final merge ---
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

      // TODO: Run experiment to add a small penalty for larger partition sizes to favor more balanced workloads

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
  return static_cast<size_t>(batch_size) * num_partitions * kConvergentSortMaxK;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED, int kMaxPartitionsForKernel>
void* ChooseKernel() {
  void* kernel;
  if constexpr (K_PADDED * kMaxPartitionsForKernel <= kMaxSmallSortSize)
    kernel = (void*)FlashSortSmallKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
  else
    kernel = (void*)FlashSortLargeKernel<kBlockSize, kPartitionSize, K_PADDED, kMaxPartitionsForKernel>;
  return kernel;
}

template <int kBlockSize, int kPartitionSize, int K_PADDED>
void* GetKernel(int num_partitions) {
  void* kernel;
  if (num_partitions <= 8) {
    kernel = ChooseKernel<kBlockSize, kPartitionSize, K_PADDED, 8>();
  } else if (num_partitions <= 16) {
    kernel = ChooseKernel<kBlockSize, kPartitionSize, K_PADDED, 16>();
  } else if (num_partitions <= 32) {
    kernel = ChooseKernel<kBlockSize, kPartitionSize, K_PADDED, 32>();
  } else {
    kernel = ChooseKernel<kBlockSize, kPartitionSize, K_PADDED, 64>();
  }
  return kernel;
}

template <int P_SIZE, int K_PADDED>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  int num_partitions = CeilDiv(vocab_size, P_SIZE);

  dim3 grid(num_partitions, batch_size);
  dim3 block(kBlockSize);

  void* kernel = GetKernel<kBlockSize, P_SIZE, K_PADDED>(num_partitions);
  void* kernel_args[] = {(void*)&scores_in, (void*)&data->intermediate_scores_1, (void*)&data->intermediate_indices_1,
                         (void*)&data->intermediate_scores_2, (void*)&data->intermediate_indices_2,
                         (void*)&vocab_size, (void*)&num_partitions, (void*)&k};
  CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, kernel_args, 0, stream));
}

template <int K_PADDED>
void LaunchKernelByPartitionSize(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size) {
  if (partition_size == kSupportedPartitionSizes[0])
    LaunchKernel<kSupportedPartitionSizes[0], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kSupportedPartitionSizes[1])
    LaunchKernel<kSupportedPartitionSizes[1], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (partition_size == kSupportedPartitionSizes[2])
    LaunchKernel<kSupportedPartitionSizes[2], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
  else
    LaunchKernel<kSupportedPartitionSizes[3], K_PADDED>(data, stream, scores_in, vocab_size, batch_size, k);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  const int partition_size = data->flash_convergent_partition_size;
  if (k <= 4)
    LaunchKernelByPartitionSize<4>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 8)
    LaunchKernelByPartitionSize<8>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 16)
    LaunchKernelByPartitionSize<16>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 32)
    LaunchKernelByPartitionSize<32>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else if (k <= 52)
    LaunchKernelByPartitionSize<52>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  else
    LaunchKernelByPartitionSize<64>(data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  CUDA_CHECK_LAUNCH();

  data->topk_scores = data->intermediate_scores_2;
  data->topk_indices = data->intermediate_indices_2;
  data->topk_stride = k;
}

// --- The following implements IsSupported ---
template <int K_PADDED, int kMaxPartitionsForKernel>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
  constexpr int kBlockSize = 256;
  const int total_blocks = num_partitions * batch_size;
  void* kernel;
  if (partition_size == kSupportedPartitionSizes[0])
    kernel = ChooseKernel<kBlockSize, kSupportedPartitionSizes[0], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kSupportedPartitionSizes[1])
    kernel = ChooseKernel<kBlockSize, kSupportedPartitionSizes[1], K_PADDED, kMaxPartitionsForKernel>();
  else if (partition_size == kSupportedPartitionSizes[2])
    kernel = ChooseKernel<kBlockSize, kSupportedPartitionSizes[2], K_PADDED, kMaxPartitionsForKernel>();
  else
    kernel = ChooseKernel<kBlockSize, kSupportedPartitionSizes[3], K_PADDED, kMaxPartitionsForKernel>();

  int device, num_sm, max_blocks_per_sm;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kBlockSize, 0));
  return total_blocks <= (num_sm * max_blocks_per_sm);
}

template <int K_PADDED>
bool IsSupportedDispatch(int batch_size, int partition_size, int num_partitions) {
  if (num_partitions <= 8) {
    return CheckSupport<K_PADDED, 8>(batch_size, num_partitions, partition_size);
  } else if (num_partitions <= 16) {
    return CheckSupport<K_PADDED, 16>(batch_size, num_partitions, partition_size);
  } else if (num_partitions <= 32) {
    return CheckSupport<K_PADDED, 32>(batch_size, num_partitions, partition_size);
  }
  return CheckSupport<K_PADDED, 64>(batch_size, num_partitions, partition_size);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kConvergentSortMaxK) return false;

  int coop_support = 0;
  cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, 0);
  if (!coop_support) return false;

  const int partition_size = EstimateBestPartitionSize(vocab_size);
  if (partition_size == 0) return false;
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
