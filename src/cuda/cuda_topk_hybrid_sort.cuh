// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <vector>
#include <numeric>
#include <algorithm>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace hybrid_sort {

/**
 * @brief A high-performance, two-kernel hybrid Top-K algorithm combining a scalable
 * Stage 1 with a low-overhead cooperative Stage 2 reduction.
 *
 * Algorithm Overview:
 * This algorithm is an evolution of the original multi-kernel hybrid sort, redesigned
 * to minimize kernel launch overhead while retaining scalability.
 *
 * 1.  **Host-Side Planning (`GetReductionFactors`)**: Based on H200 benchmark data, the host
 * selects an optimal, fixed multi-step reduction plan (up to 3 steps) determined
 * solely by the number of partitions.
 *
 * 2.  **Stage 1 (Partition Top-K)**: A standard, non-cooperative kernel (`Stage1_FindPartitionsTopK`)
 * is launched. Its grid size can be large, making it suitable for finding top candidates
 * across a high number of partitions (up to 256) and for large K.
 *
 * 3.  **Stage 2 (Cooperative Reduction)**: A single cooperative kernel (`Stage2_CooperativeReduce`)
 * is launched to perform the entire reduction cascade. It uses grid-wide synchronization
 * (`grid.sync()`) between steps, eliminating the high overhead of launching multiple
 * separate reduction kernels.
 *
 * Performance Characteristics:
 * -   **Strengths**: Combines the scalability of the original hybrid sort's Stage 1 with the
 * high performance and low overhead of cascaded_sort's single-launch reduction. More robust
 * and scalable than cascaded_sort, and faster than the original hybrid_sort.
 * -   **Weaknesses**: Requires a GPU that supports `cudaLaunchCooperativeKernel` for Stage 2.
 */

namespace cg = cooperative_groups;

constexpr int kMaxPartitions = 256;

// A set of well-spaced candidate partition sizes.
constexpr std::array<int, 4> kCandidatePartitionSizes = {1792, 2304, 2816, 3328};

// --- Host-Side Planning Logic ---

// Helper to compute the max of three values at compile time.
template <typename T>
__host__ __device__ __forceinline__ constexpr T Max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__host__ __device__ __forceinline__ constexpr T Max(T a, T b, T c) {
  return Max(a, Max(b, c));
}

struct ReductionFactors {
  int factor1 = 1;
  int factor2 = 1;
  int factor3 = 1;
  int num_reduction_steps = 0;
};

/**
 * @brief Computes the optimal reduction factors based on a comprehensive performance analysis
 * of H200 benchmark data. The optimal strategy is determined by the number of partitions.
 */
constexpr ReductionFactors GetReductionFactors(int num_partitions) {
  if (num_partitions <= 1) return {1, 1, 1, 0};
  if (num_partitions <= 4) return {4, 1, 1, 1};
  if (num_partitions <= 8) return {8, 1, 1, 1};
  if (num_partitions <= 16) return {16, 1, 1, 1};
  if (num_partitions <= 32) return {8, 4, 1, 2};
  if (num_partitions <= 64) return {8, 8, 1, 2};
  if (num_partitions <= 128) return {8, 8, 2, 3};
  // Handles 129-256 partitions
  return {8, 8, 4, 3};
}

inline int EstimateBestPartitionSize(int vocab_size) {
  constexpr std::array<int, 8> kPowerOfTwoTargets = {2, 4, 8, 16, 32, 64, 128, 256};
  int best_partition_size = 0;
  double min_waste_ratio = std::numeric_limits<double>::infinity();

  for (int p_size : kCandidatePartitionSizes) {
    int num_partitions = CeilDiv(vocab_size, p_size);
    if (num_partitions > kMaxPartitions) continue;

    auto it = std::lower_bound(kPowerOfTwoTargets.begin(), kPowerOfTwoTargets.end(), num_partitions);
    if (it != kPowerOfTwoTargets.end()) {
      int target_partitions = *it;
      double waste_ratio = static_cast<double>(target_partitions - num_partitions) / target_partitions;
      if (waste_ratio < min_waste_ratio) {
        min_waste_ratio = waste_ratio;
        best_partition_size = p_size;
      }
    }
  }
  return (best_partition_size == 0) ? kCandidatePartitionSizes[0] : best_partition_size;
}

// Heuristic to select an optimal block size based on the padded k value.
template <int K_PADDED>
constexpr int GetOptimalBlockSize() {
  return (K_PADDED <= 16) ? 128 : 256;
}

// --- Kernels ---

// Stage 1: Standard kernel to find Top-K within each partition.
template <int kBlockSize, int kPartitionSize, int K, bool UseMergeSort>
__global__ void Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                          int* __restrict__ intermediate_indices,
                                          float* __restrict__ intermediate_scores,
                                          int vocab_size, int num_partitions) {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  __shared__ Stage1TempStorageType smem;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K, UseMergeSort>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

// Stage 2: Cooperative kernel to perform the entire reduction cascade.
template <int K_PADDED, int kBlockSize, int Factor1, int Factor2, int Factor3>
__global__ void Stage2_CooperativeReduce(int* __restrict__ intermediate_indices_1,
                                         float* __restrict__ intermediate_scores_1,
                                         int* __restrict__ intermediate_indices_2,
                                         float* __restrict__ intermediate_scores_2,
                                         int num_partitions) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;

  // --- Shared Memory Union ---
  constexpr int kSortSize1 = K_PADDED * Factor1;
  constexpr int kSortSize2 = K_PADDED * Factor2;
  constexpr int kSortSize3 = K_PADDED * Factor3;
  constexpr int kMaxSortSize = Max(kSortSize1, kSortSize2, kSortSize3);
  constexpr int kItemsPerThread = CeilDiv(kMaxSortSize, kBlockSize);

  union SharedStorage {
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

  // --- Step 1: First Reduction ---
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

  // --- Step 2: Second Reduction ---
  int partitions_after_step2 = partitions_after_step1;
  if (Factor2 > 1) {
    partitions_after_step2 = CeilDiv(partitions_after_step1, Factor2);
    if (partition_idx < partitions_after_step2) {
      const float* scores_in_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      // Note: The output of the second step (if it's not the last) goes back to buffer 1 to be read by the third step.
      float* scores_out_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      int* indices_out_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;

      int first_child = partition_idx * Factor2;
      int num_to_process = min(Factor2, partitions_after_step1 - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize2, K_PADDED, kItemsPerThread>(
          scores_in_batch, indices_in_batch, scores_out_batch, indices_out_batch,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
    grid.sync();
  }

  // --- Step 3: Third Reduction ---
  if (Factor3 > 1) {
    int partitions_after_step3 = CeilDiv(partitions_after_step2, Factor3);
    if (partition_idx < partitions_after_step3) {
      // The input comes from buffer 1, where step 2 wrote its output.
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      // Final output goes to buffer 2.
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;

      int first_child = partition_idx * Factor3;
      int num_to_process = min(Factor3, partitions_after_step2 - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize3, K_PADDED, kItemsPerThread>(
          scores_in_batch, indices_in_batch, scores_out_batch, indices_out_batch,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
  }
}

// --- Main Entry Point ---

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));
  if (data->hybrid_sort_partition_size == 0) {
    data->hybrid_sort_partition_size = EstimateBestPartitionSize(vocab_size);
  }

  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  int k_padded;
  if (k <= 4) k_padded = 4;
  else if (k <= 8) k_padded = 8;
  else if (k <= 16) k_padded = 16;
  else if (k <= 32) k_padded = 32;
  else if (k <= 64) k_padded = 64;
  else if (k <= 128) k_padded = 128;
  else k_padded = 256;

  const auto factors = GetReductionFactors(num_partitions);

  auto launch = [&](auto k_padded_const) {
    constexpr int K_PADDED = k_padded_const.value;
    constexpr int kBlockSize = GetOptimalBlockSize<K_PADDED>();
    constexpr bool kUseMergeSortS1 = false; // Radix is generally better for large partitions

    // --- Stage 1 Launch ---
    dim3 grid_stage1(num_partitions, batch_size);
    dim3 block_stage1(256);
#define LAUNCH_STAGE1_KERNEL(P_SIZE) \
    Stage1_FindPartitionsTopK<256, P_SIZE, K_PADDED, kUseMergeSortS1><<<grid_stage1, block_stage1, 0, stream>>>( \
        scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions)

    if (partition_size == kCandidatePartitionSizes[0]) LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[0]);
    else if (partition_size == kCandidatePartitionSizes[1]) LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[1]);
    else if (partition_size == kCandidatePartitionSizes[2]) LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[2]);
    else LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[3]);
    CUDA_CHECK_LAUNCH();
#undef LAUNCH_STAGE1_KERNEL

    // --- Stage 2 Launch ---
    if (factors.num_reduction_steps > 0) {
      // The grid for the cooperative kernel is sized for the first reduction step.
      int partitions_after_step1 = CeilDiv(num_partitions, factors.factor1);
      dim3 grid_stage2(partitions_after_step1, batch_size);
      dim3 block_stage2(kBlockSize);

      void* kernel_args[5];
      kernel_args[0] = (void*)&data->intermediate_indices_1;
      kernel_args[1] = (void*)&data->intermediate_scores_1;
      kernel_args[2] = (void*)&data->intermediate_indices_2;
      kernel_args[3] = (void*)&data->intermediate_scores_2;
      kernel_args[4] = (void*)&num_partitions;

#define LAUNCH_STAGE2_KERNEL(F1, F2, F3) \
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, F1, F2, F3>, \
        grid_stage2, block_stage2, kernel_args, 0, stream))

      if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 4) LAUNCH_STAGE2_KERNEL(8, 8, 4);
      else if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 2) LAUNCH_STAGE2_KERNEL(8, 8, 2);
      else if (factors.factor1 == 8 && factors.factor2 == 8) LAUNCH_STAGE2_KERNEL(8, 8, 1);
      else if (factors.factor1 == 8 && factors.factor2 == 4) LAUNCH_STAGE2_KERNEL(8, 4, 1);
      else if (factors.factor1 == 8) LAUNCH_STAGE2_KERNEL(8, 1, 1);
      else if (factors.factor1 == 16) LAUNCH_STAGE2_KERNEL(16, 1, 1);
      else if (factors.factor1 == 4) LAUNCH_STAGE2_KERNEL(4, 1, 1);
      else LAUNCH_STAGE2_KERNEL(1, 1, 1);
      CUDA_CHECK_LAUNCH();
#undef LAUNCH_STAGE2_KERNEL
    }
  };

  // Dispatch based on k_padded
  if (k_padded == 4) launch(std::integral_constant<int, 4>());
  else if (k_padded == 8) launch(std::integral_constant<int, 8>());
  else if (k_padded == 16) launch(std::integral_constant<int, 16>());
  else if (k_padded == 32) launch(std::integral_constant<int, 32>());
  else if (k_padded == 64) launch(std::integral_constant<int, 64>());
  else if (k_padded == 128) launch(std::integral_constant<int, 128>());
  else launch(std::integral_constant<int, 256>());

  // Determine final output pointers
  if (factors.num_reduction_steps == 1 || factors.num_reduction_steps == 3) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }
  data->topk_stride = k_padded;
}

// Check if the cooperative kernel can be launched with the required grid size.
template<int K_PADDED>
bool CheckSupportCooperative(int batch_size, int num_partitions) {
    const auto factors = GetReductionFactors(num_partitions);
    constexpr int kBlockSize = GetOptimalBlockSize<K_PADDED>();
    int grid_x = CeilDiv(num_partitions, factors.factor1);
    int total_blocks = grid_x * batch_size;

    void* kernel = nullptr;
    if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 4) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 8, 8, 4>;
    else if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 2) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 8, 8, 2>;
    else if (factors.factor1 == 8 && factors.factor2 == 8) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 8, 8, 1>;
    else if (factors.factor1 == 8 && factors.factor2 == 4) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 8, 4, 1>;
    else if (factors.factor1 == 8) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 8, 1, 1>;
    else if (factors.factor1 == 16) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 16, 1, 1>;
    else if (factors.factor1 == 4) kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 4, 1, 1>;
    else kernel = (void*)Stage2_CooperativeReduce<K_PADDED, kBlockSize, 1, 1, 1>;

    return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kHybridSortMaxK) return false;
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) return false;

  int k_padded;
  if (k <= 4) k_padded = 4;
  else if (k <= 8) k_padded = 8;
  else if (k <= 16) k_padded = 16;
  else if (k <= 32) k_padded = 32;
  else if (k <= 64) k_padded = 64;
  else if (k <= 128) k_padded = 128;
  else k_padded = 256;

  if (k_padded == 4) return CheckSupportCooperative<4>(batch_size, num_partitions);
  if (k_padded == 8) return CheckSupportCooperative<8>(batch_size, num_partitions);
  if (k_padded == 16) return CheckSupportCooperative<16>(batch_size, num_partitions);
  if (k_padded == 32) return CheckSupportCooperative<32>(batch_size, num_partitions);
  if (k_padded == 64) return CheckSupportCooperative<64>(batch_size, num_partitions);
  if (k_padded == 128) return CheckSupportCooperative<128>(batch_size, num_partitions);
  if (k_padded == 256) return CheckSupportCooperative<256>(batch_size, num_partitions);

  return false;
}

}  // namespace hybrid_sort
}  // namespace cuda
}  // namespace Generators

