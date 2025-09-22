// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <type_traits>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_common.cuh"

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
 * host-side planner (`GetReductionFactors`) to determine an optimal sequence of up to three
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

struct ReductionFactors {
  int factor1 = 1;
  int factor2 = 1;
  int factor3 = 1;
  int num_reduction_steps = 0;
};

/**
 * @brief Computes the optimal reduction factors based on the number of partitions and `k`.
 * It uses a more complex 3-step reduction for larger `k` values and falls back to
 * an aggressive 1 or 2-step plan for smaller `k`.
 */
constexpr ReductionFactors GetReductionFactors(int num_partitions, int k) {
  constexpr int k_large_threshold = 32;

  // For large k, use a 3-step strategy with smaller, more efficient factors.
  if (k > k_large_threshold) {
    if (num_partitions > 32) {  // 33-64 partitions
      return {4, 4, 4, 3};
    }
    if (num_partitions > 16) {  // 17-32 partitions
      return {4, 4, 2, 3};
    }
  }

  // Otherwise, use the original high-performance 1 or 2-step logic for smaller k.
  if (num_partitions <= 1) {
    return {1, 1, 1, 0};
  }
  if (num_partitions <= 8) {
    int f1 = (num_partitions <= 2) ? 2 : ((num_partitions <= 4) ? 4 : 8);
    return {f1, 1, 1, 1};
  }
  if (num_partitions <= 16) {
    return {4, 4, 1, 2};
  }
  if (num_partitions <= 32) {
    return {8, 4, 1, 2};
  }
  // 33-64 partitions
  return {8, 8, 1, 2};
}

/**
 * @brief The main kernel for Cascaded Sort. It performs the initial partition sort
 * followed by up to three cascaded reduction steps, all within a single launch.
 */
template <int K_PADDED, int kBlockSize, int kPartitionSize, int Factor1, int Factor2, int Factor3>
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
  constexpr int kSortSize3 = K_PADDED * Factor3;

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;

    struct {
      __align__(128) float scores[kSortSize1];
      __align__(128) int indices[kSortSize1];
    } step1_storage;
    struct {
      __align__(128) float scores[kSortSize2];
      __align__(128) int indices[kSortSize2];
    } step2_storage;
    struct {
      __align__(128) float scores[kSortSize3];
      __align__(128) int indices[kSortSize3];
    } step3_storage;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Find Top-K within each partition ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, Stage1TempStorageType>(
      input_scores, intermediate_indices_1, intermediate_scores_1, vocab_size, num_partitions, smem.stage1_storage);

  grid.sync();

  // --- Stage 2, Step 1: First Reduction ---
  // This block executes if the plan includes at least one reduction step.
  // It reads from buffer 1, merges `Factor1` partitions, and writes the result to buffer 2.
  int partitions_after_step1 = num_partitions;
  if (Factor1 > 1) {
    partitions_after_step1 = CeilDiv(num_partitions, Factor1);
    if (partition_idx < partitions_after_step1) {
      // Reads from buffer 1, writes to buffer 2
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;

      int first_child = partition_idx * Factor1;
      int num_to_process = min(Factor1, num_partitions - first_child);
      for (int i = threadIdx.x; i < kSortSize1; i += kBlockSize) {
        if (i < K_PADDED * num_to_process) {
          size_t offset = (static_cast<size_t>(first_child) + i / K_PADDED) * K_PADDED + (i % K_PADDED);
          smem.step1_storage.scores[i] = scores_in_batch[offset];
          smem.step1_storage.indices[i] = indices_in_batch[offset];
        } else {
          smem.step1_storage.scores[i] = -FLT_MAX;
          smem.step1_storage.indices[i] = INT_MAX;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort<kBlockSize, kSortSize1>(smem.step1_storage.scores, smem.step1_storage.indices);
      if (threadIdx.x < K_PADDED) {
        scores_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step1_storage.scores[threadIdx.x];
        indices_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step1_storage.indices[threadIdx.x];
      }
    }
    grid.sync();
  }

  // --- Stage 2, Step 2: Second Reduction ---
  // This block executes if the plan includes a second reduction step.
  // It reads from buffer 2, merges `Factor2` partitions, and writes the result to buffer 1.
  int partitions_after_step2 = partitions_after_step1;
  if (Factor2 > 1) {
    partitions_after_step2 = CeilDiv(partitions_after_step1, Factor2);
    if (partition_idx < partitions_after_step2) {
      // Reads from buffer 2, writes to buffer 1
      const float* scores_in_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step1 * K_PADDED;
      float* scores_out_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      int* indices_out_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;

      int first_child = partition_idx * Factor2;
      int num_to_process = min(Factor2, partitions_after_step1 - first_child);
      for (int i = threadIdx.x; i < kSortSize2; i += kBlockSize) {
        if (i < K_PADDED * num_to_process) {
          size_t offset = (static_cast<size_t>(first_child) + i / K_PADDED) * K_PADDED + (i % K_PADDED);
          smem.step2_storage.scores[i] = scores_in_batch[offset];
          smem.step2_storage.indices[i] = indices_in_batch[offset];
        } else {
          smem.step2_storage.scores[i] = -FLT_MAX;
          smem.step2_storage.indices[i] = INT_MAX;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort<kBlockSize, kSortSize2>(smem.step2_storage.scores, smem.step2_storage.indices);
      if (threadIdx.x < K_PADDED) {
        scores_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step2_storage.scores[threadIdx.x];
        indices_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step2_storage.indices[threadIdx.x];
      }
    }
    grid.sync();
  }

  // --- Stage 2, Step 3: Third Reduction ---
  // This block executes if the plan includes a third reduction step.
  // It reads from buffer 1, merges `Factor3` partitions, and writes the result to buffer 2.
  if (Factor3 > 1) {
    int partitions_after_step3 = CeilDiv(partitions_after_step2, Factor3);
    if (partition_idx < partitions_after_step3) {
      // Reads from buffer 1, writes to buffer 2
      const float* scores_in_batch = intermediate_scores_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      const int* indices_in_batch = intermediate_indices_1 + static_cast<size_t>(batch_idx) * partitions_after_step2 * K_PADDED;
      float* scores_out_batch = intermediate_scores_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;
      int* indices_out_batch = intermediate_indices_2 + static_cast<size_t>(batch_idx) * partitions_after_step3 * K_PADDED;

      int first_child = partition_idx * Factor3;
      int num_to_process = min(Factor3, partitions_after_step2 - first_child);
      for (int i = threadIdx.x; i < kSortSize3; i += kBlockSize) {
        if (i < K_PADDED * num_to_process) {
          size_t offset = (static_cast<size_t>(first_child) + i / K_PADDED) * K_PADDED + (i % K_PADDED);
          smem.step3_storage.scores[i] = scores_in_batch[offset];
          smem.step3_storage.indices[i] = indices_in_batch[offset];
        } else {
          smem.step3_storage.scores[i] = -FLT_MAX;
          smem.step3_storage.indices[i] = INT_MAX;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort<kBlockSize, kSortSize3>(smem.step3_storage.scores, smem.step3_storage.indices);
      if (threadIdx.x < K_PADDED) {
        scores_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step3_storage.scores[threadIdx.x];
        indices_out_batch[static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x] = smem.step3_storage.indices[threadIdx.x];
      }
    }
  }
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kCascadedSortMaxK;
}

// Parition sizes are optimized for common vocab_size (padded to multiple of 256) used in open source LLM:
//    32256, 32512, 128256, 128512, 152064, 152320, 200192, 200448, 201216, 201472, 262400, 262656.
// Constraints: partition_size are multiple of 256, partition_size <= 8192.
// Goal: mimize average waste ratio to get total partitions be one of 2, 4, 8, 16, 32 and 64.
// For example, vocab_size=201088, ideal partition size is 3142 for 64 partitions. The waste ratio is 1 - 3142/3328 = 0.055.
// The maximum vocab_size that this kernel can support is decided by below choices (i.e. 4864 * 64 = 311296).
constexpr std::array<int, 4> kAllowedPartitionSizes = {2048, 3328, 4352, 4864};

constexpr std::array<int, 7> kTargetPartitionCounts = {1, 2, 4, 8, 16, 32, 64};

inline int EstimateBestPartitionSize(int vocab_size) {
  double min_cost_ratio = std::numeric_limits<double>::infinity();
  int best_partition_size = 0;

  for (int partition_size : kAllowedPartitionSizes) {
    int partitions_needed = CeilDiv(vocab_size, partition_size);

    if (partitions_needed <= 64) {  // Max target count constraint
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
template <int K_PADDED, int Factor1, int Factor2, int Factor3>
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
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], Factor1, Factor2, Factor3>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[1]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], Factor1, Factor2, Factor3>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[2]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], Factor1, Factor2, Factor3>, grid, block, kernel_args, 0, stream));
      break;
    case kAllowedPartitionSizes[3]:
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], Factor1, Factor2, Factor3>, grid, block, kernel_args, 0, stream));
      break;
    default:
      assert(false);  // Should be unreachable
      break;
  }
}

// Templated helper to dispatch to the correct kernel based on reduction factors.
template <int K_PADDED>
void LaunchKernel(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  const int num_partitions = CeilDiv(vocab_size, data->cascaded_sort_partition_size);
  const auto factors = GetReductionFactors(num_partitions, k);

  // This dispatch logic is verbose, but it ensures only the necessary kernel variants are instantiated,
  // balancing performance with build time.
  if (factors.factor1 == 8 && factors.factor2 == 8)
    LaunchKernelWithFactors<K_PADDED, 8, 8, 1>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 8 && factors.factor2 == 4)
    LaunchKernelWithFactors<K_PADDED, 8, 4, 1>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 8)
    LaunchKernelWithFactors<K_PADDED, 8, 1, 1>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 4 && factors.factor2 == 4 && factors.factor3 == 4)
    LaunchKernelWithFactors<K_PADDED, 4, 4, 4>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 4 && factors.factor2 == 4 && factors.factor3 == 2)
    LaunchKernelWithFactors<K_PADDED, 4, 4, 2>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 4 && factors.factor2 == 4)
    LaunchKernelWithFactors<K_PADDED, 4, 4, 1>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 4)
    LaunchKernelWithFactors<K_PADDED, 4, 1, 1>(data, stream, scores_in, vocab_size, batch_size);
  else if (factors.factor1 == 2)
    LaunchKernelWithFactors<K_PADDED, 2, 1, 1>(data, stream, scores_in, vocab_size, batch_size);
  else
    LaunchKernelWithFactors<K_PADDED, 1, 1, 1>(data, stream, scores_in, vocab_size, batch_size);
}

// --- Unified Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));  // caller shall check IsSupported before calling this function.

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

  const auto factors = GetReductionFactors(num_partitions, k);
  const int num_reduction_steps = factors.num_reduction_steps;

  if (num_reduction_steps % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }

  int num_partitions_out = num_partitions;
  if (num_reduction_steps > 0) num_partitions_out = CeilDiv(num_partitions, factors.factor1);
  if (num_reduction_steps > 1) num_partitions_out = CeilDiv(num_partitions_out, factors.factor2);
  if (num_reduction_steps > 2) num_partitions_out = CeilDiv(num_partitions_out, factors.factor3);
  data->topk_stride = k_padded_val * num_partitions_out;
}

template <int K_PADDED, int Factor1, int Factor2, int Factor3>
bool CheckSupportWithFactors(int batch_size, int partition_size, int num_partitions) {
  constexpr int kBlockSize = GetOptimalBlockSize<K_PADDED>();
  const int total_blocks = num_partitions * batch_size;
  void* kernel = nullptr;

  switch (partition_size) {
    case kAllowedPartitionSizes[0]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[0], Factor1, Factor2, Factor3>;
      break;
    case kAllowedPartitionSizes[1]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[1], Factor1, Factor2, Factor3>;
      break;
    case kAllowedPartitionSizes[2]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[2], Factor1, Factor2, Factor3>;
      break;
    case kAllowedPartitionSizes[3]:
      kernel = (void*)CascadedSortKernel<K_PADDED, kBlockSize, kAllowedPartitionSizes[3], Factor1, Factor2, Factor3>;
      break;
    default:
      return false;  // Should be unreachable
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kBlockSize, 0));
  int max_active_blocks = num_sm * max_blocks_per_sm;

  return total_blocks <= max_active_blocks;
}

// Templated helper to check for support with a constexpr block size.
template <int K_PADDED>
bool CheckSupport(int batch_size, int vocab_size, int k, int partition_size, int num_partitions) {
  const auto factors = GetReductionFactors(num_partitions, k);
  if (factors.factor1 == 8 && factors.factor2 == 8) return CheckSupportWithFactors<K_PADDED, 8, 8, 1>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 8 && factors.factor2 == 4) return CheckSupportWithFactors<K_PADDED, 8, 4, 1>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 8) return CheckSupportWithFactors<K_PADDED, 8, 1, 1>(batch_size, vocab_size, num_partitions);
  if (factors.factor1 == 4 && factors.factor2 == 4 && factors.factor3 == 4) return CheckSupportWithFactors<K_PADDED, 4, 4, 4>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 4 && factors.factor2 == 4 && factors.factor3 == 2) return CheckSupportWithFactors<K_PADDED, 4, 4, 2>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 4 && factors.factor2 == 4) return CheckSupportWithFactors<K_PADDED, 4, 4, 1>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 4) return CheckSupportWithFactors<K_PADDED, 4, 1, 1>(batch_size, partition_size, num_partitions);
  if (factors.factor1 == 2) return CheckSupportWithFactors<K_PADDED, 2, 1, 1>(batch_size, partition_size, num_partitions);
  return CheckSupportWithFactors<K_PADDED, 1, 1, 1>(batch_size, partition_size, num_partitions);
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

  int cooperative_launch_support = 0;
  cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, 0);
  if (!cooperative_launch_support) {
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
    return false;  // Should be unreachable
  }
}

}  // namespace cascaded_sort
}  // namespace cuda
}  // namespace Generators
