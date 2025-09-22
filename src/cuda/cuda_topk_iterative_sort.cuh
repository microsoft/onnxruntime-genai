// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <type_traits>  // For std::integral_constant
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_common.cuh"

namespace Generators {
namespace cuda {
namespace iterative_sort {

/**
 * @brief A high-performance, single-kernel cooperative Top-K algorithm.
 *
 * Algorithm Overview:
 * 1.  **Stage 1 (Partition Top-K)**: The input is partitioned, and `topk_common::FindPartitionTopK`
 * is used to find the top `K_PADDED` candidates within each partition in parallel.
 *
 * 2.  **Stage 2 (Iterative Reduction)**: The kernel then enters a loop to perform a tree-based reduction
 * on the candidates from Stage 1. It uses `cg::grid_group::sync()` to ensure all blocks have
 * completed a reduction level before starting the next.
 * -   In each iteration, `kReductionFactor` (fixed at 4) sets of candidates are merged by a single thread block.
 * -   This process repeats until only one set of candidates (the final Top-K) remains.
 * -   Two intermediate buffers are used in a "ping-pong" fashion to pass data between reduction levels.
 *
 * Performance Characteristics:
 * -   **Strengths**: By encapsulating all logic in a single kernel, it minimizes kernel launch overhead,
 * which is a significant advantage over multi-kernel approaches. It also has a specialized,
 * fast path for small `k` (<= 8) that uses a highly efficient warp-level bitonic sort for the reduction step.
 * -   **Weaknesses**: Requires a GPU that supports `cudaLaunchCooperativeKernel`. The fixed reduction
 * factor is less adaptive than the strategies used in `cascaded_sort` or `hybrid_sort`.
 */

namespace cg = cooperative_groups;

// A fixed reduction factor is used for simplicity and performance. Each reduction
// step merges 4 partitions of candidates.
constexpr int kReductionFactor = 4;

// Utility to swap pointers, used for ping-ponging between intermediate buffers during reduction.
__host__ __device__ inline void swap_ptr(float*& a, float*& b) {
  float* tmp = a;
  a = b;
  b = tmp;
}

__host__ __device__ inline void swap_ptr(int*& a, int*& b) {
  int* tmp = a;
  a = b;
  b = tmp;
}

template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void IterativeSortKernel(const float* __restrict__ input_scores,
                                    int* __restrict__ intermediate_indices_1,
                                    float* __restrict__ intermediate_scores_1,
                                    int* __restrict__ intermediate_indices_2,
                                    float* __restrict__ intermediate_scores_2,
                                    int vocab_size) {
  cg::grid_group grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  // --- Shared Memory Union ---
  // A union is used to efficiently reuse shared memory across different stages of the kernel.
  constexpr int kSortSize = K_PADDED * kReductionFactor;

  using Stage1TempStorageType = typename topk_common::Stage1TempStorage<kBlockSize, kPartitionSize>;
  union SharedStorage {
    Stage1TempStorageType stage1_storage;
    struct {
      __align__(128) float scores[kSortSize];
      __align__(128) int indices[kSortSize];
    } stage2_storage;
  };
  __shared__ SharedStorage smem;

  // --- Stage 1: Find Top-K within each partition ---
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED, Stage1TempStorageType>(
      input_scores, intermediate_indices_1, intermediate_scores_1, vocab_size, num_partitions, smem.stage1_storage);

  // Grid-wide synchronization to ensure all partitions are processed before starting reduction.
  grid.sync();

  // --- Stage 2: Iterative Tree Reduction ---
  int* p_indices_in = intermediate_indices_1;
  float* p_scores_in = intermediate_scores_1;
  int* p_indices_out = intermediate_indices_2;
  float* p_scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = (partitions_remaining + kReductionFactor - 1) / kReductionFactor;
    // Only the first `num_active_blocks` are responsible for merging in this iteration.
    if (partition_idx < num_active_blocks) {
      const size_t in_batch_offset = static_cast<size_t>(batch_idx) * partitions_remaining * K_PADDED;
      const size_t out_batch_offset = static_cast<size_t>(batch_idx) * num_active_blocks * K_PADDED;
      const int* indices_in_batch = p_indices_in + in_batch_offset;
      const float* scores_in_batch = p_scores_in + in_batch_offset;
      int* indices_out_batch = p_indices_out + out_batch_offset;
      float* scores_out_batch = p_scores_out + out_batch_offset;

      // Each active block loads candidates from `kReductionFactor` partitions into its shared memory.
      int first_child_partition = partition_idx * kReductionFactor;
      int num_partitions_to_process = min(kReductionFactor, partitions_remaining - first_child_partition);

      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        if (i < K_PADDED * num_partitions_to_process) {
          int part_idx = i / K_PADDED;
          int element_idx = i % K_PADDED;
          size_t local_offset = static_cast<size_t>(first_child_partition + part_idx) * K_PADDED + element_idx;
          smem.stage2_storage.scores[i] = scores_in_batch[local_offset];
          smem.stage2_storage.indices[i] = indices_in_batch[local_offset];
        } else {
          // Pad with sentinel values for a clean sort.
          smem.stage2_storage.scores[i] = -FLT_MAX;
          smem.stage2_storage.indices[i] = INT_MAX;
        }
      }
      __syncthreads();

      // --- Choose sorting strategy based on K ---
      if constexpr (K_PADDED <= 8) {
        // For small K, the total number of items to sort (kSortSize) is at most 32.
        // A warp-level bitonic sort is extremely fast as it operates entirely in registers
        // without needing shared memory for the sort itself.
        if (threadIdx.x < warpSize) {
          float my_score;
          int my_index;
          if (threadIdx.x < kSortSize) {
            my_score = smem.stage2_storage.scores[threadIdx.x];
            my_index = smem.stage2_storage.indices[threadIdx.x];
          } else {
            my_score = -FLT_MAX;
            my_index = INT_MAX;
          }
          bitonic_sort::WarpBitonicSort(my_score, my_index);
          if (threadIdx.x < K_PADDED) {
            smem.stage2_storage.scores[threadIdx.x] = my_score;
            smem.stage2_storage.indices[threadIdx.x] = my_index;
          }
        }
        __syncthreads();
      } else {
        // For larger K, a shared-memory bitonic sort is a robust choice.
        bitonic_sort::SharedMemBitonicSort<kBlockSize, kSortSize>(smem.stage2_storage.scores, smem.stage2_storage.indices);
      }

      // Write the top K results for this merged group to the output buffer.
      if (threadIdx.x < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
        scores_out_batch[out_offset] = smem.stage2_storage.scores[threadIdx.x];
        indices_out_batch[out_offset] = smem.stage2_storage.indices[threadIdx.x];
      }
    }
    partitions_remaining = num_active_blocks;
    // Swap input/output pointers for the next iteration (ping-pong).
    swap_ptr(p_scores_in, p_scores_out);
    swap_ptr(p_indices_in, p_indices_out);
    grid.sync();
  }
}

// Host-side launcher that selects the correct kernel template instantiation based on runtime parameters.
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(IsSupported(batch_size, vocab_size, k));

  constexpr int kBlockSize = 256;
  const int partition_size = data->iterative_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // Pad `k` to the next power of two or a supported value. This is necessary because
  // the templated kernels must have a compile-time constant for K.
  int k_padded_val = kIterativeSortMaxK;
  if (k == 1)
    k_padded_val = 1;
  else if (k <= 4)
    k_padded_val = 4;
  else if (k <= 8)
    k_padded_val = 8;
  else if (k <= 16)
    k_padded_val = 16;
  else if (k <= 32)
    k_padded_val = 32;
  else if (k <= 64)
    k_padded_val = 64;

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  // This lambda dispatches to the correct kernel based on the compile-time K_PADDED value.
  auto launch_iterative_sort = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 block(kBlockSize);
    dim3 grid(num_partitions, batch_size);
    switch (partition_size) {
      case 1024:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)IterativeSortKernel<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream));
        break;
      case 2048:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)IterativeSortKernel<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream));
        break;
      default:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)IterativeSortKernel<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream));
        break;
    }
  };

  // Select the padded K value at runtime and call the launch logic.
  if (k == 1) {
    launch_iterative_sort(std::integral_constant<int, 1>());
  } else if (k <= 4) {
    launch_iterative_sort(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    launch_iterative_sort(std::integral_constant<int, 8>());
  } else if (k <= 16) {
    launch_iterative_sort(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    launch_iterative_sort(std::integral_constant<int, 32>());
  } else if (k <= 64) {
    launch_iterative_sort(std::integral_constant<int, 64>());
  } else {
    if constexpr (kIterativeSortMaxK > 64) {
      launch_iterative_sort(std::integral_constant<int, kIterativeSortMaxK>());
    }
  }

  CUDA_CHECK_LAUNCH();

  // Determine the final output buffer based on the number of reduction loops.
  int num_reduction_loops = 0;
  if (num_partitions > 1) {
    int partitions_remaining = num_partitions;
    while (partitions_remaining > 1) {
      partitions_remaining = (partitions_remaining + kReductionFactor - 1) / kReductionFactor;
      num_reduction_loops++;
    }
  }

  // Stage 1 writes to buffer 1. After an odd number of loops, the result is in buffer 2.
  // After an even number of loops, the result is back in buffer 1.
  if (num_reduction_loops % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
  }
  data->topk_stride = k_padded_val;
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kIterativeSortMaxK;
}

inline int EstimateBestPartitionSize(int vocab_size) {
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  return 4096;
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kIterativeSortMaxK) {
    return false;
  }

  // Check for cooperative launch support
  int cooperative_launch_support = 0;
  cudaDeviceGetAttribute(&cooperative_launch_support, cudaDevAttrCooperativeLaunch, 0);
  if (!cooperative_launch_support) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  const int total_blocks = num_partitions * batch_size;

  // Choose kernel using the same logic as in RunTopK.
  auto get_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    switch (partition_size) {
      case 1024:
        return (void*)IterativeSortKernel<K_PADDED, kBlockSize, 1024>;
      case 2048:
        return (void*)IterativeSortKernel<K_PADDED, kBlockSize, 2048>;
      default:
        return (void*)IterativeSortKernel<K_PADDED, kBlockSize, 4096>;
    }
  };

  void* kernel = nullptr;
  if (k == 1) {
    kernel = get_kernel(std::integral_constant<int, 1>());
  } else if (k <= 4) {
    kernel = get_kernel(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    kernel = get_kernel(std::integral_constant<int, 8>());
  } else if (k <= 16) {
    kernel = get_kernel(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    kernel = get_kernel(std::integral_constant<int, 32>());
  } else if (k <= 64) {
    kernel = get_kernel(std::integral_constant<int, 64>());
  } else {
    if constexpr (kIterativeSortMaxK > 64) {
      kernel = get_kernel(std::integral_constant<int, kIterativeSortMaxK>());
    }
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));

  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm,
      kernel,
      kBlockSize,
      0));  // Pass 0 for dynamic shared memory. This kernel uses only STATIC shared memory.

  int max_active_blocks = num_sm * max_blocks_per_sm;

  if (total_blocks > max_active_blocks) {
    return false;
  }

  return true;
}

}  // namespace iterative_sort
}  // namespace cuda
}  // namespace Generators
