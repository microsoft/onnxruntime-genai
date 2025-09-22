// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_common.cuh"

namespace Generators {
namespace cuda {
namespace hybrid_sort {

/**
 * @brief A portable, multi-kernel, host-planned hybrid Top-K algorithm.
 *
 * Algorithm Overview:
 * This algorithm is designed for portability and does not require cooperative launch.
 * It uses a host-side planner to orchestrate a multi-stage reduction process.
 *
 * 1.  **Host-Side Planning (`GetReductionPlan`)**: Before launching any kernels, the host
 * determines an optimal multi-step reduction plan. It greedily decides how many partitions
 * to merge in each step, aiming to minimize the number of kernel launches. It also selects
 * the best internal sorting algorithm for each step based on the number of items to sort.
 *
 * 2.  **Stage 1 (Partition Top-K)**: A standard kernel (`Stage1_FindPartitionsTopK`) is launched to
 * find the top `K_PADDED` candidates from each vocabulary partition, similar to other algorithms.
 *
 * 3.  **Stage 2 (Planned Reduction)**: A series of reduction kernels (`AdvancedBlockReduceTopK`)
 * are launched according to the plan from step 1. Each launch merges a group of candidate
 * sets from the previous stage.
 *
 * Performance Characteristics:
 * -   **Strengths**: Its main advantage is portability, as it runs on GPUs that do not support
 * cooperative kernels. The host-side planner can make intelligent decisions to create an
 * efficient reduction strategy.
 * -   **"Hybrid" Nature**: The `AdvancedBlockReduceTopK` kernel is a "hybrid" because it uses
 * a compile-time dispatch (`ReductionAlgorithm` enum) to select the most efficient internal
 * sorting method (Warp Bitonic, Block Bitonic, or CUB Radix Sort) based on the number of
 * elements being sorted in that specific reduction step. This provides fine-grained optimization.
 * -   **Weaknesses**: The use of multiple kernel launches can introduce higher overhead compared to
 * single-kernel cooperative approaches like `iterative_sort` or `cascaded_sort`.
 */

// The internal sorting algorithm to be used inside the reduction kernel, chosen by the host-side planner.
enum class ReductionAlgorithm {
  WARP_BITONIC,   // For very small sorts (<= 32 items), uses a register-based warp sort.
  BLOCK_BITONIC,  // For small sorts (33-256 items), uses a shared memory bitonic sort.
  CUB_RADIX_SORT  // For larger sorts (> 256 items), uses the powerful CUB block-wide radix sort.
};

// Contains the parameters for a single reduction kernel launch.
struct ReductionStep {
  int partitions_per_block;      // How many candidate sets to merge in this step.
  int block_size;                // The thread block size for the kernel launch.
  ReductionAlgorithm algorithm;  // The internal sort algorithm to use.
};

using ReductionPlan = std::vector<ReductionStep>;
constexpr int kMaxPartitions = 256;
constexpr int kMaxItemsToSortPerBlock = 4096;  // Budget based on ~64KB of shared memory.

// A set of well-spaced candidate partition sizes.
constexpr std::array<int, 4> kCandidatePartitionSizes = {2816, 3328, 4096, 4864};

// --- Host-Side Planning Logic ---

/**
 * @brief Estimates the best partition size for the given vocabulary size.
 * The goal is to select a size that results in a total number of partitions
 * close to a power of two, which is ideal for reduction stages.
 */
inline int EstimateBestPartitionSize(int vocab_size) {
  constexpr std::array<int, 8> kPowerOfTwoTargets = {2, 4, 8, 16, 32, 64, 128, 256};

  int best_partition_size = 0;
  double min_waste_ratio = std::numeric_limits<double>::infinity();

  for (int p_size : kCandidatePartitionSizes) {
    int num_partitions = CeilDiv(vocab_size, p_size);

    if (num_partitions > kMaxPartitions) {
      continue;
    }

    // Find the smallest power-of-two target that is >= num_partitions
    auto it = std::lower_bound(kPowerOfTwoTargets.begin(), kPowerOfTwoTargets.end(), num_partitions);
    if (it != kPowerOfTwoTargets.end()) {
      int target_partitions = *it;
      // The "waste" is how many "empty" partitions we'd need to add to the target.
      // A lower waste ratio is better.
      double waste_ratio = static_cast<double>(target_partitions - num_partitions) / target_partitions;

      if (waste_ratio < min_waste_ratio) {
        min_waste_ratio = waste_ratio;
        best_partition_size = p_size;
      }
    }
  }

  // Fallback in case no suitable partition size was found.
  return (best_partition_size == 0) ? kCandidatePartitionSizes[0] : best_partition_size;
}

/**
 * @brief Creates a deterministic, multi-step reduction plan.
 * This function determines how many partitions to merge in each step to minimize
 * kernel launches, and which internal sort algorithm is best for each step.
 */
inline ReductionPlan GetReductionPlan(int num_partitions, int k_padded) {
  ReductionPlan plan;
  int current_partitions = num_partitions;

  while (current_partitions > 1) {
    // Determine the max number of partitions we can merge in one block based on shared memory budget.
    int max_p_per_block = (k_padded > 0) ? (kMaxItemsToSortPerBlock / k_padded) : 0;
    if (max_p_per_block <= 1) break;  // Cannot reduce further

    int p_per_block = 1;
    // Greedily select the largest reduction factor possible from our available kernels.
    constexpr std::array<int, 4> kReductionFactors = {16, 8, 4, 2};
    for (int factor : kReductionFactors) {
      if (factor <= current_partitions && factor <= max_p_per_block) {
        p_per_block = factor;
        break;
      }
    }

    if (p_per_block <= 1) {
      break;  // Stuck, cannot make progress
    }

    ReductionStep step;
    step.partitions_per_block = p_per_block;
    step.block_size = (k_padded <= 16) ? 128 : 256;

    int sort_size = k_padded * step.partitions_per_block;
    if (sort_size <= 32) {
      step.algorithm = ReductionAlgorithm::WARP_BITONIC;
    } else if (sort_size <= 256) {
      step.algorithm = ReductionAlgorithm::BLOCK_BITONIC;
    } else {
      step.algorithm = ReductionAlgorithm::CUB_RADIX_SORT;
    }
    plan.push_back(step);
    current_partitions = CeilDiv(current_partitions, step.partitions_per_block);
  }

  return plan;
}

// --- Stage 1 Kernel ---
// Finds the Top-K within each partition of the vocabulary.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                          int* __restrict__ intermediate_indices,
                                          float* __restrict__ intermediate_scores,
                                          int vocab_size, int num_partitions) {
  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  __shared__ Stage1TempStorageType smem;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K, Stage1TempStorageType>(scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

/**
 * @brief The unified reduction kernel. This single kernel can perform any of the reduction
 * algorithms by using compile-time template parameters.
 */
template <int kBlockSize, int K_PADDED, int PartitionsPerBlock, ReductionAlgorithm Algorithm>
__global__ void AdvancedBlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                        float* __restrict__ scores_out, int* __restrict__ indices_out, int num_partitions_in) {
  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions_in * K_PADDED;
  const size_t out_base_offset = (static_cast<size_t>(batch_idx) * gridDim.x + blockIdx.x) * K_PADDED;

  constexpr int kSortSize = K_PADDED * PartitionsPerBlock;

  // --- Sorting Stage (Compile-Time Dispatch) ---
  if constexpr (Algorithm == ReductionAlgorithm::CUB_RADIX_SORT) {
    // --- CUB Radix Sort Path ---
    // This path loads data directly from global memory to registers, using shared memory
    // exclusively for CUB's internal temporary storage to avoid memory hazards.
    constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);
#ifdef STABLE_TOPK
    using SortKeyT = uint64_t;
    __shared__ typename cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread>::TempStorage smem_cub;
    SortKeyT thread_keys[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < K_PADDED * num_partitions_to_process) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = topk_common::PackStableSortKey(scores_in[offset], indices_in[offset]);
      } else {
        thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
      }
    }
    cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread>(smem_cub).SortDescendingBlockedToStriped(thread_keys);
    if (threadIdx.x < K_PADDED) {
      scores_out[out_base_offset + threadIdx.x] = topk_common::UnpackStableSortScore(thread_keys[0]);
      indices_out[out_base_offset + threadIdx.x] = topk_common::UnpackStableSortIndex(thread_keys[0]);
    }
#else
    using SortKeyT = float;
    using SortValueT = int;
    __shared__ typename cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>::TempStorage smem_cub;
    SortKeyT thread_keys[kItemsPerThread];
    SortValueT thread_values[kItemsPerThread];

    for (int i = 0; i < kItemsPerThread; ++i) {
      int item_idx = threadIdx.x + i * kBlockSize;
      if (item_idx < K_PADDED * num_partitions_to_process) {
        int partition_idx = item_idx / K_PADDED;
        int element_idx = item_idx % K_PADDED;
        size_t offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        thread_keys[i] = scores_in[offset];
        thread_values[i] = indices_in[offset];
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = INT_MAX;
      }
    }
    cub::BlockRadixSort<SortKeyT, kBlockSize, kItemsPerThread, SortValueT>(smem_cub).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      scores_out[out_base_offset + threadIdx.x] = thread_keys[0];
      indices_out[out_base_offset + threadIdx.x] = thread_values[0];
    }
#endif
  } else {
    // --- Bitonic Sort Path (Warp or Block) ---
    // This path loads data into shared memory first, then performs an in-place sort.
    constexpr int kSortSizePo2 = topk_common::NextPowerOfTwo(kSortSize > 1 ? kSortSize - 1 : 0);
    __shared__ struct {
      __align__(128) float scores[kSortSizePo2];
      __align__(128) int indices[kSortSizePo2];
    } smem_bitonic;

    for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
      if (i < K_PADDED * num_partitions_to_process) {
        int partition_idx = i / K_PADDED;
        int element_idx = i % K_PADDED;
        size_t global_offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K_PADDED + element_idx;
        smem_bitonic.scores[i] = scores_in[global_offset];
        smem_bitonic.indices[i] = indices_in[global_offset];
      } else {
        smem_bitonic.scores[i] = -FLT_MAX;
        smem_bitonic.indices[i] = INT_MAX;
      }
    }
    for (int i = kSortSize + threadIdx.x; i < kSortSizePo2; i += kBlockSize) {
      smem_bitonic.scores[i] = -FLT_MAX;
      smem_bitonic.indices[i] = INT_MAX;
    }
    __syncthreads();

    if constexpr (Algorithm == ReductionAlgorithm::WARP_BITONIC) {
      if (threadIdx.x < warpSize) {
        float my_score = (threadIdx.x < kSortSize) ? smem_bitonic.scores[threadIdx.x] : -FLT_MAX;
        int my_index = (threadIdx.x < kSortSize) ? smem_bitonic.indices[threadIdx.x] : INT_MAX;
        bitonic_sort::WarpBitonicSort(my_score, my_index);
        if (threadIdx.x < K_PADDED) {
          smem_bitonic.scores[threadIdx.x] = my_score;
          smem_bitonic.indices[threadIdx.x] = my_index;
        }
      }
      __syncthreads();
    } else if constexpr (Algorithm == ReductionAlgorithm::BLOCK_BITONIC) {
      if constexpr (kSortSizePo2 <= 32)
        bitonic_sort::SharedMemBitonicSort<kBlockSize, 32>(smem_bitonic.scores, smem_bitonic.indices);
      else if constexpr (kSortSizePo2 <= 64)
        bitonic_sort::SharedMemBitonicSort<kBlockSize, 64>(smem_bitonic.scores, smem_bitonic.indices);
      else if constexpr (kSortSizePo2 <= 128)
        bitonic_sort::SharedMemBitonicSort<kBlockSize, 128>(smem_bitonic.scores, smem_bitonic.indices);
      else if constexpr (kSortSizePo2 <= 256)
        bitonic_sort::SharedMemBitonicSort<kBlockSize, 256>(smem_bitonic.scores, smem_bitonic.indices);
    }

    if (threadIdx.x < K_PADDED) {
      indices_out[out_base_offset + threadIdx.x] = smem_bitonic.indices[threadIdx.x];
      scores_out[out_base_offset + threadIdx.x] = smem_bitonic.scores[threadIdx.x];
    }
  }
}

// --- Kernel Launcher ---
// Helper to launch the reduction kernel with the correct template parameters from the plan.
template <int K_PADDED>
void LaunchReductionStep(const ReductionStep& step, cudaStream_t stream,
                         const float* scores_in, const int* indices_in,
                         float* scores_out, int* indices_out,
                         int num_partitions_in, int batch_size) {
  dim3 grid(CeilDiv(num_partitions_in, step.partitions_per_block), batch_size);
  dim3 block(step.block_size);

#define LAUNCH_KERNEL(B_SIZE, P_PER_BLOCK, ALGO)                                            \
  AdvancedBlockReduceTopK<B_SIZE, K_PADDED, P_PER_BLOCK, ALGO><<<grid, block, 0, stream>>>( \
      scores_in, indices_in, scores_out, indices_out, num_partitions_in)

  auto dispatch_kernels = [&](auto block_size_const) {
    constexpr int B_SIZE = block_size_const.value;
    // Dispatch to the correct kernel based on the partitions_per_block chosen by the planner.
    switch (step.partitions_per_block) {
      case 16:
        if (step.algorithm == ReductionAlgorithm::CUB_RADIX_SORT)
          LAUNCH_KERNEL(B_SIZE, 16, ReductionAlgorithm::CUB_RADIX_SORT);
        else
          LAUNCH_KERNEL(B_SIZE, 16, ReductionAlgorithm::BLOCK_BITONIC);
        break;
      case 8:
        if (step.algorithm == ReductionAlgorithm::CUB_RADIX_SORT)
          LAUNCH_KERNEL(B_SIZE, 8, ReductionAlgorithm::CUB_RADIX_SORT);
        else
          LAUNCH_KERNEL(B_SIZE, 8, ReductionAlgorithm::BLOCK_BITONIC);
        break;
      case 4:
        if (step.algorithm == ReductionAlgorithm::CUB_RADIX_SORT)
          LAUNCH_KERNEL(B_SIZE, 4, ReductionAlgorithm::CUB_RADIX_SORT);
        else if (step.algorithm == ReductionAlgorithm::BLOCK_BITONIC)
          LAUNCH_KERNEL(B_SIZE, 4, ReductionAlgorithm::BLOCK_BITONIC);
        else
          LAUNCH_KERNEL(B_SIZE, 4, ReductionAlgorithm::WARP_BITONIC);
        break;
      case 2:
        if (step.algorithm == ReductionAlgorithm::BLOCK_BITONIC)
          LAUNCH_KERNEL(B_SIZE, 2, ReductionAlgorithm::BLOCK_BITONIC);
        else
          LAUNCH_KERNEL(B_SIZE, 2, ReductionAlgorithm::WARP_BITONIC);
        break;
      default:
        // This case indicates a planning logic issue.
        break;
    }
  };

  if (step.block_size == 128) {
    dispatch_kernels(std::integral_constant<int, 128>());
  } else {  // 256
    dispatch_kernels(std::integral_constant<int, 256>());
  }
#undef LAUNCH_KERNEL
}

// Helper to launch Stage 1 with the correct K_TEMPLATE compile-time constant.
template <int K_TEMPLATE>
void LaunchStage1(cudaStream_t stream, int partition_size, int num_partitions, int batch_size,
                  const float* scores_in, TopkData* data, int vocab_size) {
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(256);

#define LAUNCH_STAGE1_KERNEL(P_SIZE)                                                            \
  Stage1_FindPartitionsTopK<256, P_SIZE, K_TEMPLATE><<<grid_stage1, block_stage1, 0, stream>>>( \
      scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions)

  switch (partition_size) {
    case kCandidatePartitionSizes[0]:
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[0]);
      break;
    case kCandidatePartitionSizes[1]:
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[1]);
      break;
    case kCandidatePartitionSizes[2]:
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[2]);
      break;
    case kCandidatePartitionSizes[3]:
      LAUNCH_STAGE1_KERNEL(kCandidatePartitionSizes[3]);
      break;
    default:
      break;  // Should not happen with current planner
  }
#undef LAUNCH_STAGE1_KERNEL
}

// --- Main Entry Point ---

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));  // caller shall check IsSupported before calling this function.

  // --- Stage 0: Planning ---
  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // Pad k to a supported value for template instantiation.
  int k_padded;
  if (k <= 4)
    k_padded = 4;
  else if (k <= 8)
    k_padded = 8;
  else if (k <= 16)
    k_padded = 16;
  else if (k <= 32)
    k_padded = 32;
  else if (k <= 64)
    k_padded = 64;
  else if (k <= 128)
    k_padded = 128;
  else
    k_padded = 256;

  // Create the reduction plan based on the padded k value.
  const ReductionPlan plan = GetReductionPlan(num_partitions, k_padded);

  // This lambda captures the launch logic to be called by the k-dispatch below.
  auto launch_kernels = [&](auto k_padded_const) {
    constexpr int K_PADDED = k_padded_const.value;
    // --- Stage 1: Find Partition Top-K ---
    LaunchStage1<K_PADDED>(stream, partition_size, num_partitions, batch_size, scores_in, data, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    // --- Stage 2: Planned Reduction ---
    if (num_partitions > 1 && !plan.empty()) {
      int current_num_partitions = num_partitions;
      // Setup ping-pong pointers for reduction inputs/outputs.
      float* scores_in_ptr = data->intermediate_scores_1;
      int* indices_in_ptr = data->intermediate_indices_1;
      float* scores_out_ptr = data->intermediate_scores_2;
      int* indices_out_ptr = data->intermediate_indices_2;

      // Execute each step in the reduction plan.
      for (const auto& step : plan) {
        LaunchReductionStep<K_PADDED>(step, stream, scores_in_ptr, indices_in_ptr, scores_out_ptr, indices_out_ptr, current_num_partitions, batch_size);
        CUDA_CHECK(cudaGetLastError());
        current_num_partitions = CeilDiv(current_num_partitions, step.partitions_per_block);
        std::swap(scores_in_ptr, scores_out_ptr);
        std::swap(indices_in_ptr, indices_out_ptr);
      }
      // The final result is in the last "in" pointer after the swaps.
      data->topk_scores = scores_in_ptr;
      data->topk_indices = indices_in_ptr;
    } else {
      // No reduction was needed.
      data->topk_scores = data->intermediate_scores_1;
      data->topk_indices = data->intermediate_indices_1;
    }
    data->topk_stride = K_PADDED;
  };

  if (k_padded == 4)
    launch_kernels(std::integral_constant<int, 4>());
  else if (k_padded == 8)
    launch_kernels(std::integral_constant<int, 8>());
  else if (k_padded == 16)
    launch_kernels(std::integral_constant<int, 16>());
  else if (k_padded == 32)
    launch_kernels(std::integral_constant<int, 32>());
  else if (k_padded == 64)
    launch_kernels(std::integral_constant<int, 64>());
  else if (k_padded == 128)
    launch_kernels(std::integral_constant<int, 128>());
  else if (k_padded == 256)
    launch_kernels(std::integral_constant<int, 256>());
}

bool IsSupported(int /*batch_size*/, int vocab_size, int k) {
  if (k > kHybridSortMaxK) {
    return false;
  }

  const int partition_size = EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  if (num_partitions > kMaxPartitions) {
    return false;
  }

  return true;
}

}  // namespace hybrid_sort
}  // namespace cuda
}  // namespace Generators
