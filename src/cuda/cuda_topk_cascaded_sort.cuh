// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <type_traits>
#include "cuda_topk.h"
#include "cuda_topk_warp_sort_helper.cuh"
#include "cuda_topk_common.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace cascaded_sort {

/**
 * @brief A high-performance, single-kernel cooperative "cascaded" sort, specifically
 * optimized for common Large Language Model (LLM) workloads.
 *
 * Algorithm Overview:
 * This algorithm is an evolution of `iterative_sort`, designed to be more adaptive.
 *
 * 1.  **Stage 1 (Partition Top-K)**: The input is partitioned, and `topk_common::FindPartitionTopK`
 * finds the top `K_PADDED` candidates within each partition.
 *
 * 2.  **Stage 2 (Cascaded Reduction)**: Instead of a fixed reduction factor, this algorithm uses a
 * host-side planner (`GetReductionFactors`) to determine an optimal sequence of up to two
 * reduction factors (e.g., merge 8 sets, then merge 4 sets). This "cascaded" approach,
 * where each reduction step is a separate, grid-synchronized phase within the *same kernel*,
 * allows it to adapt more effectively to the number of partitions.
 *
 * Performance Characteristics:
 * -   **Strengths**: Offers the highest performance for its target workloads by combining the low
 * launch overhead of a single cooperative kernel with a more intelligent, adaptive reduction
 * strategy than `iterative_sort`. The planner is tuned for vocabulary sizes and partition counts
 * commonly found in LLMs.
 * -   **Weaknesses**: Requires a GPU that supports `cudaLaunchCooperativeKernel`. Its performance
 * gains are most pronounced in the specific scenarios it was tuned for.
 */

namespace cg = cooperative_groups;

// The limit on partitions is due to cooperative group residency requirements and the
// fact that a single block must sort all `k * num_partitions` candidates in Stage 2.
constexpr int kMaxPartitions = 64;

// Parition sizes are optimized for common vocab_size (padded to multiple of 256) used in open source LLM:
//    32256, 32512, 128256, 128512, 152064, 152320, 200192, 200448, 201216, 201472, 262400, 262656.
// Constraints: partition_size are multiple of 256, partition_size <= 4096.
// Goal: mimize average waste ratio to get total partitions be one of 2, 4, 8, 16, 32 and 64.
// For example, vocab_size=201088, ideal partition size is 3142 for 64 partitions. The waste ratio is 1 - 3142/3328 = 0.055.
// The maximum vocab_size that this kernel can support is decided by below choices (i.e. 4096 * 64 = 262,144).
constexpr std::array<int, 4> kAllowedPartitionSizes = {1792, 2048, 2560, 3328};

// Helper to compute the max of two values at compile time.
template <typename T>
__host__ __device__ __forceinline__ constexpr T Max(T a, T b) {
  return a > b ? a : b;
}

struct ReductionFactors {
  int factor1 = 1;
  int factor2 = 1;
  int num_reduction_steps = 0;
};

/**
 * @brief Computes the optimal reduction factors based on a comprehensive performance analysis
 * of H200 benchmark data. The optimal strategy is consistently determined by the number of
 * partitions, independent of the `k` value. A maximum of two reduction steps was found
 * to be sufficient for all supported cases (up to 64 partitions).
 */
constexpr ReductionFactors GetReductionFactors(int num_partitions) {
  if (num_partitions <= 1) {
    return {1, 1, 0};
  }
  if (num_partitions <= 4) {
    return {4, 1, 1};
  }
  if (num_partitions <= 8) {
    return {8, 1, 1};
  }
  if (num_partitions <= 16) {
    return {4, 4, 2};
  }
  if (num_partitions <= 32) {
    return {8, 4, 2};
  }
  // Handles 33-64 partitions
  return {8, 8, 2};
}

/**
 * @brief The main kernel for Cascaded Sort. It performs the initial partition sort
 * followed by up to two cascaded reduction steps, all within a single launch.
 */
template <int K_PADDED, int kBlockSize, int kPartitionSize, int Factor1, int Factor2, bool UseMergeS1>
__global__ void CascadedSortKernel(const float* __restrict__ input_scores,
                                   int* __restrict__ intermediate_indices_1,
                                   float* __restrict__ intermediate_scores_1,
                                   int* __restrict__ intermediate_indices_2,
                                   float* __restrict__ intermediate_scores_2,
                                   int vocab_size) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  // --- Shared Memory Union for efficiency ---
  constexpr int kSortSize1 = K_PADDED * Factor1;
  constexpr int kSortSize2 = K_PADDED * Factor2;
  constexpr int kMaxSortSize = Max(kSortSize1, kSortSize2);
  constexpr int kItemsPerThread = CeilDiv(kMaxSortSize, kBlockSize);

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;
    typename cub::WarpMergeSort<uint64_t, (kMaxSortSize + 31) / 32, 32>::TempStorage cub_warp_storage;
#ifdef STABLE_TOPK
    typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage cub_block_merge_storage;
#else
    typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage cub_block_merge_storage;
#endif
    struct {
      __align__(128) float scores[kMaxSortSize];
      __align__(128) int indices[kMaxSortSize];
    } stage2_storage;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Find Top-K within each partition ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, UseMergeS1>(
      input_scores, intermediate_indices_1, intermediate_scores_1, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2, Step 1: First Reduction ---
  int partitions_after_step1 = num_partitions;
  if (Factor1 > 1) {
    partitions_after_step1 = CeilDiv(num_partitions, Factor1);
    if (partition_idx < partitions_after_step1) {
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;

      int first_child = partition_idx * Factor1;
      int num_to_process = min(Factor1, num_partitions - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize1, K_PADDED, kItemsPerThread>(
          scores_in_batch, indices_in_batch, scores_out_batch, indices_out_batch,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
    grid.sync();
  }

  // --- Stage 2, Step 2: Second Reduction ---
  if (Factor2 > 1) {
    int partitions_after_step2 = CeilDiv(partitions_after_step1, Factor2);
    if (partition_idx < partitions_after_step2) {
      const float* scores_in_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      float* scores_out_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      int* indices_out_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;

      int first_child = partition_idx * Factor2;
      int num_to_process = min(Factor2, partitions_after_step1 - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize2, K_PADDED, kItemsPerThread>(
          scores_in_batch, indices_in_batch, scores_out_batch, indices_out_batch,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
  }
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kCascadedSortMaxK;
}

constexpr std::array<int, 7> kTargetPartitionCounts = {1, 2, 4, 8, 16, 32, 64};

inline int EstimateBestPartitionSize(int vocab_size) {
  double min_cost_ratio = std::numeric_limits<double>::infinity();
  int best_partition_size = 0;

  for (int partition_size : kAllowedPartitionSizes) {
    int partitions_needed = CeilDiv(vocab_size, partition_size);

    if (partitions_needed <= kMaxPartitions) {  // Max target count constraint
      // Find smallest target count >= partitions_needed
      auto target_it = std::lower_bound(kTargetPartitionCounts.begin(), kTargetPartitionCounts.end(), partitions_needed);
      if (target_it != kTargetPartitionCounts.end()) {
        int target = *target_it;
        double cost_ratio = static_cast<double>(partition_size * target - vocab_size) / vocab_size;
        if (cost_ratio < min_cost_ratio) {
          min_cost_ratio = cost_ratio;
          best_partition_size = partition_size;
        }
      }
    }
  }
  return best_partition_size;
}

// Heuristic to select an optimal block size based on the padded k value.
// Smaller k values benefit from smaller block sizes, which can improve occupancy.
template <int K_PADDED>
constexpr int GetOptimalBlockSize() {
  return (K_PADDED <= 16) ? 128 : 256;
}

// Templated helper to launch the kernel with a constexpr block size and reduction factors.
template <int K_PADDED, int Factor1, int Factor2, bool UseMergeS1>
void LaunchKernelWithFactors(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size) {
  constexpr int kBlockSize = GetOptimalBlockSize<K_PADDED>();

  const int partition_size = data->cascaded_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  dim3 grid(num_partitions, batch_size);
  dim3 block(kBlockSize);

  switch (partition_size) {
    case kAllowedPartitionSizes[0]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], Factor1, Factor2, UseMergeS1>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[1]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], Factor1, Factor2, UseMergeS1>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[2]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], Factor1, Factor2, UseMergeS1>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[3]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], Factor1, Factor2, UseMergeS1>, grid, block, kernel_args, 0, stream));
      break;
    default:
      assert(false);
      break;
  }
}

// Templated helper to dispatch to the correct kernel based on reduction factors.
template <int K_PADDED>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  const int num_partitions = CeilDiv(vocab_size, data->cascaded_sort_partition_size);
  const auto factors = GetReductionFactors(num_partitions);

  // When sort size is larger than 1024, CUB's block radix sort is better than block merge sort (See cuda_topk_sort_benchmark_cache.h).
  constexpr bool kUseMergeSortS1 = false;

#define LAUNCH_WITH_FACTORS(F1, F2) LaunchKernelWithFactors<K_PADDED, F1, F2, kUseMergeSortS1>(data, stream, scores_in, vocab_size, batch_size)

  if (factors.factor1 == 8 && factors.factor2 == 8)
    LAUNCH_WITH_FACTORS(8, 8);
  else if (factors.factor1 == 8 && factors.factor2 == 4)
    LAUNCH_WITH_FACTORS(8, 4);
  else if (factors.factor1 == 8)
    LAUNCH_WITH_FACTORS(8, 1);
  else if (factors.factor1 == 4 && factors.factor2 == 4)
    LAUNCH_WITH_FACTORS(4, 4);
  else if (factors.factor1 == 4)
    LAUNCH_WITH_FACTORS(4, 1);
  else
    LAUNCH_WITH_FACTORS(1, 1);

#undef LAUNCH_WITH_FACTORS
}

// --- Unified Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));  // caller shall check IsSupported before calling this function.

  if (data->cascaded_sort_partition_size == 0) {
    data->cascaded_sort_partition_size = EstimateBestPartitionSize(vocab_size);
  }

  const int partition_size = data->cascaded_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // This kernel could support up to K=256 in theory, but in practice we limit it to reduce build time.
  static_assert(kCascadedSortMaxK == 64 || kCascadedSortMaxK == 128);
  int k_padded_val = kCascadedSortMaxK;
  if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 64)
    k_padded_val = 64;

  // Dispatch to the correct templated launch helper based on k_padded_val
  if (k_padded_val == 4)
    LaunchKernel<4>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (k_padded_val == 8)
    LaunchKernel<8>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (k_padded_val == 16)
    LaunchKernel<16>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (k_padded_val == 32)
    LaunchKernel<32>(data, stream, scores_in, vocab_size, batch_size, k);
  else if (k_padded_val == 64)
    LaunchKernel<64>(data, stream, scores_in, vocab_size, batch_size, k);
  else if constexpr (kCascadedSortMaxK > 64)
    LaunchKernel<kCascadedSortMaxK>(data, stream, scores_in, vocab_size, batch_size, k);

  CUDA_CHECK_LAUNCH();

  const auto factors = GetReductionFactors(num_partitions);
  const int num_reduction_steps = factors.num_reduction_steps;

  if (num_reduction_steps == 1) { // After 1 step, results are in buffer 2
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else { // After 0 or 2 steps, results are in buffer 1
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }

  int num_partitions_out = num_partitions;
  if (num_reduction_steps > 0) num_partitions_out = CeilDiv(num_partitions, factors.factor1);
  if (num_reduction_steps > 1) num_partitions_out = CeilDiv(num_partitions_out, factors.factor2);
  data->topk_stride = k_padded_val * num_partitions_out;
}

template <int K_PADDED, int Factor1, int Factor2>
bool CheckSupportWithFactors(int batch_size, int partition_size, int num_partitions) {
  constexpr int kBlockSize = GetOptimalBlockSize<K_PADDED>();
  const int total_blocks = num_partitions * batch_size;

  // When sort size is larger than 1024, CUB's block radix sort is better than block merge sort (See cuda_topk_sort_benchmark_cache.h).
  constexpr bool kUseMergeSortS1 = false;

  void* kernel = nullptr;

  switch (partition_size) {
    case kAllowedPartitionSizes[0]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], Factor1, Factor2, kUseMergeSortS1>;
      break;
    case kAllowedPartitionSizes[1]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], Factor1, Factor2, kUseMergeSortS1>;
      break;
    case kAllowedPartitionSizes[2]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], Factor1, Factor2, kUseMergeSortS1>;
      break;
    case kAllowedPartitionSizes[3]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], Factor1, Factor2, kUseMergeSortS1>;
      break;
    default:
      return false;
  }

  return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

// Templated helper to check for support with a constexpr block size.
template <int K_PADDED>
bool CheckSupport(int batch_size, int vocab_size, int k, int partition_size, int num_partitions) {
  const auto factors = GetReductionFactors(num_partitions);

  if (factors.factor1 == 8 && factors.factor2 == 8) return CheckSupportWithFactors<K_PADDED, 8, 8>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 8 && factors.factor2 == 4) return CheckSupportWithFactors<K_PADDED, 8, 4>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 8) return CheckSupportWithFactors<K_PADDED, 8, 1>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 4 && factors.factor2 == 4) return CheckSupportWithFactors<K_PADDED, 4, 4>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 4) return CheckSupportWithFactors<K_PADDED, 4, 1>(batch_size, partition_size, num_partitions);
  return CheckSupportWithFactors<K_PADDED, 1, 1>(batch_size, partition_size, num_partitions);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kCascadedSortMaxK) {
    return false;
  }
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  if (partition_size == 0) {
    return false;
  }
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > 64) {
    return false;
  }

  int k_padded_val;
  if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 64)
    k_padded_val = 64;
  else
    k_padded_val = kCascadedSortMaxK;

  // Dispatch to the correct templated support checker based on k_padded_val
  if (k_padded_val == 4)
    return CheckSupport<4>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k_padded_val == 8)
    return CheckSupport<8>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k_padded_val == 16)
    return CheckSupport<16>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k_padded_val == 32)
    return CheckSupport<32>(batch_size, vocab_size, k, partition_size, num_partitions);
  if (k_padded_val == 64)
    return CheckSupport<64>(batch_size, vocab_size, k, partition_size, num_partitions);

  if constexpr (kCascadedSortMaxK > 64) {
    return CheckSupport<kCascadedSortMaxK>(batch_size, vocab_size, k, partition_size, num_partitions);
  } else {
    return false;
  }
}

}  // namespace cascaded_sort
}  // namespace cuda
}  // namespace Generators

