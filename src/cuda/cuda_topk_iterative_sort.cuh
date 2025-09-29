// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <type_traits>
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {
namespace iterative_sort {

/**
 * @brief A single-kernel cooperative sort, specialized for **mid-to-large k**.
 *
 * Algorithm Overview:
 * This is an evolution of the original iterative sort, now featuring an adaptive reduction factor.
 *
 * 1.  **Host-Side Planning**: A smart host-side planner (`EstimateBestPartitionSize`)
 * considers a wide range of partition sizes and co-designs the partition count with an
 * optimal reduction factor to minimize workload imbalance and overall cost.
 *
 * 2.  **Stage 1 (Partition Top-K)**: All blocks find top candidates in parallel.
 *
 * 3.  **Stage 2 (Adaptive Iterative Reduction)**: The kernel enters a loop, repeatedly
 * merging candidates using the single, pre-calculated reduction factor. This maintains the
 * low overhead of a single kernel launch while being more efficient than a globally fixed
 * factor for partition counts that are not powers of 4.
 *
 * Performance Characteristics:
 * -   **Strengths**: Fast for mid-to-large `k` where a consistent reduction strategy is
 * beneficial and the overhead of more complex planning is not justified.
 * -   **Weaknesses**: Requires cooperative launch. May be less optimal than fully adaptive
 * (multi-factor) plans for certain partition counts.
 */

namespace cg = cooperative_groups;

__host__ __device__ inline void swap_ptr(float*& a, float*& b) { float* tmp = a; a = b; b = tmp; }
__host__ __device__ inline void swap_ptr(int*& a, int*& b) { int* tmp = a; a = b; b = tmp; }

template <int K_PADDED, int kBlockSize, int kPartitionSize, int kReductionFactor>
__global__ void AdaptiveIterativeSortKernel(const float* __restrict__ input_scores,
                                            int* __restrict__ intermediate_indices_1,
                                            float* __restrict__ intermediate_scores_1,
                                            int* __restrict__ intermediate_indices_2,
                                            float* __restrict__ intermediate_scores_2,
                                            int vocab_size) {
  constexpr bool UseCubMergeSort = (kPartitionSize <= 1024);
  cg::grid_group grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  constexpr int kSortSize = K_PADDED * kReductionFactor;
  constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
      Stage1TempStorageType stage1_storage;
      typename cub::WarpMergeSort<uint64_t, (kSortSize + 31) / 32, 32>::TempStorage cub_warp_storage;
#ifdef STABLE_TOPK
      typename cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>::TempStorage cub_block_merge_storage;
#else
      typename cub::BlockMergeSort<float, kBlockSize, kItemsPerThread, int>::TempStorage cub_block_merge_storage;
#endif
      struct {
          __align__(128) float scores[kSortSize];
          __align__(128) int indices[kSortSize];
      } stage2_storage;
  };
  __shared__ SharedStorage smem;

  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, UseCubMergeSort>(
      input_scores, intermediate_indices_1, intermediate_scores_1, vocab_size, num_partitions, smem.stage1_storage);
  grid.sync();

  int* p_indices_in = intermediate_indices_1;
  float* p_scores_in = intermediate_scores_1;
  int* p_indices_out = intermediate_indices_2;
  float* p_scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = CeilDiv(partitions_remaining, kReductionFactor);
    if (partition_idx < num_active_blocks) {
      const size_t in_batch_offset = static_cast<size_t>(batch_idx) * partitions_remaining * K_PADDED;
      const size_t out_batch_offset = static_cast<size_t>(batch_idx) * num_active_blocks * K_PADDED;
      
      int first_child = partition_idx * kReductionFactor;
      int num_to_process = min(kReductionFactor, partitions_remaining - first_child);
      const int num_elements_to_sort = K_PADDED * num_to_process;

      topk_common::BlockReduceTopK<kBlockSize, kSortSize, K_PADDED, kItemsPerThread>(
          p_scores_in + in_batch_offset, p_indices_in + in_batch_offset,
          p_scores_out + out_batch_offset, p_indices_out + out_batch_offset,
          num_elements_to_sort, first_child, partition_idx, smem);
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(p_scores_in, p_scores_out);
    swap_ptr(p_indices_in, p_indices_out);
    grid.sync();
  }
}

// Planner to find the best single reduction factor
inline int GetBestReductionFactor(int num_partitions, int k_padded) {
    int best_factor = 2;
    float min_waste = std::numeric_limits<float>::max();

    for (int factor = 8; factor >= 2; --factor) {
        if (k_padded * factor > 4096) continue;

        float total_waste = 0.0f;
        int current_partitions = num_partitions;
        while(current_partitions > 1) {
            int num_blocks = CeilDiv(current_partitions, factor);
            int last_block_workload = current_partitions - (num_blocks - 1) * factor;
            if (last_block_workload > 0 && num_blocks > 1) {
              total_waste += 1.0f - (float)last_block_workload / factor;
            }
            current_partitions = num_blocks;
        }

        if (total_waste < min_waste) {
            min_waste = total_waste;
            best_factor = factor;
        }
    }
    return best_factor;
}

// Smarter planner that co-designs partition size and reduction factor
inline int EstimateBestPartitionSize(int vocab_size, int k_padded) {
    constexpr std::array<int, 6> kCandidatePartitionSizes = {1024, 1280, 1792, 2048, 3328, 4096};
    int best_partition_size = 4096;
    float min_total_cost = std::numeric_limits<float>::max();

    for (int p_size : kCandidatePartitionSizes) {
        int num_partitions = CeilDiv(vocab_size, p_size);
        if (num_partitions > 64) continue;

        int factor = GetBestReductionFactor(num_partitions, k_padded);
        
        float total_waste = 0.0f;
        int current_partitions = num_partitions;
        while(current_partitions > 1) {
            int num_blocks = CeilDiv(current_partitions, factor);
            int last_block_workload = current_partitions - (num_blocks - 1) * factor;
            if (last_block_workload > 0 && num_blocks > 1) {
              total_waste += (1.0f - (float)last_block_workload / factor) * num_blocks;
            }
            current_partitions = num_blocks;
        }

        float cost = total_waste;
        if (p_size > 1024) {
            cost *= 1.2f; // Penalty for slower Radix Sort in Stage 1
        }

        if (cost < min_total_cost) {
            min_total_cost = cost;
            best_partition_size = p_size;
        }
    }
    return best_partition_size;
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));
  
  int k_padded_val;
  if (k <= 4) k_padded_val = 4;
  else if (k <= 8) k_padded_val = 8;
  else if (k <= 16) k_padded_val = 16;
  else if (k <= 32) k_padded_val = 32;
  else k_padded_val = 64;

  if (data->iterative_sort_partition_size == 0) {
    data->iterative_sort_partition_size = EstimateBestPartitionSize(vocab_size, k_padded_val);
  }

  const int partition_size = data->iterative_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  const int reduction_factor = GetBestReductionFactor(num_partitions, k_padded_val);

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  auto launch_iterative_sort = [&](auto k_padded, auto r_factor) {
      constexpr int K_PADDED = decltype(k_padded)::value;
      constexpr int R_FACTOR = decltype(r_factor)::value;
      constexpr int kBlockSize = 256;
      dim3 grid(num_partitions, batch_size);
      dim3 block(kBlockSize);

#define LAUNCH_KERNEL_P(P_SIZE) CUDA_CHECK((cudaLaunchCooperativeKernel((void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, P_SIZE, R_FACTOR>, grid, block, kernel_args, 0, stream)))
      if (partition_size == 1024) { LAUNCH_KERNEL_P(1024); }
      else if (partition_size == 1280) { LAUNCH_KERNEL_P(1280); }
      else if (partition_size == 1792) { LAUNCH_KERNEL_P(1792); }
      else if (partition_size == 2048) { LAUNCH_KERNEL_P(2048); }
      else if (partition_size == 3328) { LAUNCH_KERNEL_P(3328); }
      else { LAUNCH_KERNEL_P(4096); }
#undef LAUNCH_KERNEL_P
  };

  auto dispatch_by_factor = [&](auto k_padded) {
      switch(reduction_factor) {
          case 2: launch_iterative_sort(k_padded, std::integral_constant<int, 2>()); break;
          case 3: launch_iterative_sort(k_padded, std::integral_constant<int, 3>()); break;
          case 4: launch_iterative_sort(k_padded, std::integral_constant<int, 4>()); break;
          case 5: launch_iterative_sort(k_padded, std::integral_constant<int, 5>()); break;
          case 6: launch_iterative_sort(k_padded, std::integral_constant<int, 6>()); break;
          case 7: launch_iterative_sort(k_padded, std::integral_constant<int, 7>()); break;
          case 8: launch_iterative_sort(k_padded, std::integral_constant<int, 8>()); break;
          default: launch_iterative_sort(k_padded, std::integral_constant<int, 4>());
      }
  };

  if (k_padded_val == 4) dispatch_by_factor(std::integral_constant<int, 4>());
  else if (k_padded_val == 8) dispatch_by_factor(std::integral_constant<int, 8>());
  else if (k_padded_val == 16) dispatch_by_factor(std::integral_constant<int, 16>());
  else if (k_padded_val == 32) dispatch_by_factor(std::integral_constant<int, 32>());
  else dispatch_by_factor(std::integral_constant<int, 64>());
  
  CUDA_CHECK_LAUNCH();

  int num_reduction_loops = 0;
  if (num_partitions > 1) {
    int partitions_remaining = num_partitions;
    while (partitions_remaining > 1) {
      partitions_remaining = CeilDiv(partitions_remaining, reduction_factor);
      num_reduction_loops++;
    }
  }

  if (num_reduction_loops % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }
  data->topk_stride = k_padded_val;
}

template <int K_PADDED, int kReductionFactor>
bool CheckSupportForFactor(int batch_size, int num_partitions, int partition_size) {
    constexpr int kBlockSize = 256;
    const int total_blocks = num_partitions * batch_size;
    void* kernel = nullptr;

#define GET_KERNEL_P(P_SIZE) (void*)AdaptiveIterativeSortKernel<K_PADDED, kBlockSize, P_SIZE, kReductionFactor>
    if (partition_size == 1024) { kernel = GET_KERNEL_P(1024); }
    else if (partition_size == 1280) { kernel = GET_KERNEL_P(1280); }
    else if (partition_size == 1792) { kernel = GET_KERNEL_P(1792); }
    else if (partition_size == 2048) { kernel = GET_KERNEL_P(2048); }
    else if (partition_size == 3328) { kernel = GET_KERNEL_P(3328); }
    else { kernel = GET_KERNEL_P(4096); }
#undef GET_KERNEL_P
    
    return topk_common::IsSupportedCooperative(kernel, total_blocks, kBlockSize);
}

template <int K_PADDED>
bool CheckSupport(int batch_size, int num_partitions, int partition_size) {
    const int reduction_factor = GetBestReductionFactor(num_partitions, K_PADDED);
    switch(reduction_factor) {
        case 2: return CheckSupportForFactor<K_PADDED, 2>(batch_size, num_partitions, partition_size);
        case 3: return CheckSupportForFactor<K_PADDED, 3>(batch_size, num_partitions, partition_size);
        case 4: return CheckSupportForFactor<K_PADDED, 4>(batch_size, num_partitions, partition_size);
        case 5: return CheckSupportForFactor<K_PADDED, 5>(batch_size, num_partitions, partition_size);
        case 6: return CheckSupportForFactor<K_PADDED, 6>(batch_size, num_partitions, partition_size);
        case 7: return CheckSupportForFactor<K_PADDED, 7>(batch_size, num_partitions, partition_size);
        case 8: return CheckSupportForFactor<K_PADDED, 8>(batch_size, num_partitions, partition_size);
    }
    return false;
}

bool IsSupported(int batch_size, int vocab_size, int k) {
    if (k > kIterativeSortMaxK) return false;
    
    int k_padded_val;
    if (k <= 4) k_padded_val = 4;
    else if (k <= 8) k_padded_val = 8;
    else if (k <= 16) k_padded_val = 16;
    else if (k <= 32) k_padded_val = 32;
    else k_padded_val = 64;

    const int partition_size = EstimateBestPartitionSize(vocab_size, k_padded_val);
    const int num_partitions = CeilDiv(vocab_size, partition_size);
    if (num_partitions > 64) return false;

    if (k_padded_val == 4) return CheckSupport<4>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 8) return CheckSupport<8>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 16) return CheckSupport<16>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 32) return CheckSupport<32>(batch_size, num_partitions, partition_size);
    if (k_padded_val == 64) return CheckSupport<64>(batch_size, num_partitions, partition_size);
    
    return false;
}

}  // namespace iterative_sort
}  // namespace cuda
}  // namespace Generators

