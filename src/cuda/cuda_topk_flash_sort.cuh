// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <type_traits>  // For std::integral_constant
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace flash_sort {

namespace cg = cooperative_groups;

// A fixed reduction factor is used for simplicity and performance.
constexpr int kReductionFactor = 4;

// Utility to swap pointers, used during the reduction phase.
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
__global__ void FlashSortKernel(const float* __restrict__ input_scores,
                                int* __restrict__ intermediate_indices_1,
                                float* __restrict__ intermediate_scores_1,
                                int* __restrict__ intermediate_indices_2,
                                float* __restrict__ intermediate_scores_2,
                                int vocab_size) {
  cg::grid_group grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int num_partitions = gridDim.x;

  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;

  // --- Shared Memory Union ---
  constexpr int kSortSize = K_PADDED * kReductionFactor;

  union SharedStorage {
    typename BlockRadixSort::TempStorage stage1_storage;
    struct {
      __align__(128) float scores[kSortSize];
      __align__(128) int indices[kSortSize];
    } stage2_storage;
  };
  __shared__ SharedStorage smem;

  const float* batch_input_scores = input_scores + static_cast<size_t>(batch_idx) * vocab_size;
  const size_t batch_intermediate_offset_stage1 = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
  int* batch_intermediate_indices_1 = intermediate_indices_1 + batch_intermediate_offset_stage1;
  float* batch_intermediate_scores_1 = intermediate_scores_1 + batch_intermediate_offset_stage1;

  // --- Stage 1: Find Top-K within each partition ---
  {
    const int partition_start = partition_idx * kPartitionSize;
    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i) {
      int global_idx = partition_start + threadIdx.x + i * kBlockSize;
      if (global_idx < vocab_size) {
        thread_keys[i] = batch_input_scores[global_idx];
        thread_values[i] = global_idx;
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = -1;
      }
    }
    BlockRadixSort(smem.stage1_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      batch_intermediate_scores_1[offset] = thread_keys[0];
      batch_intermediate_indices_1[offset] = thread_values[0];
    }
  }
  grid.sync();

  // --- Stage 2: Iterative Tree Reduction ---
  int* p_indices_in = intermediate_indices_1;
  float* p_scores_in = intermediate_scores_1;
  int* p_indices_out = intermediate_indices_2;
  float* p_scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = (partitions_remaining + kReductionFactor - 1) / kReductionFactor;
    if (partition_idx < num_active_blocks) {
      const size_t in_batch_offset = static_cast<size_t>(batch_idx) * partitions_remaining * K_PADDED;
      const size_t out_batch_offset = static_cast<size_t>(batch_idx) * num_active_blocks * K_PADDED;
      int* indices_in_batch = p_indices_in + in_batch_offset;
      float* scores_in_batch = p_scores_in + in_batch_offset;
      int* indices_out_batch = p_indices_out + out_batch_offset;
      float* scores_out_batch = p_scores_out + out_batch_offset;

      int first_child_partition = partition_idx * kReductionFactor;
      int num_partitions_to_process = min(kReductionFactor, partitions_remaining - first_child_partition);

      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        if (i < K_PADDED * num_partitions_to_process) {
          int part_idx = i / K_PADDED;
          int element_idx = i % K_PADDED;
          size_t local_offset = (first_child_partition + part_idx) * K_PADDED + element_idx;
          smem.stage2_storage.scores[i] = scores_in_batch[local_offset];
          smem.stage2_storage.indices[i] = indices_in_batch[local_offset];
        } else {
          smem.stage2_storage.scores[i] = -FLT_MAX;
          smem.stage2_storage.indices[i] = -1;
        }
      }
      __syncthreads();
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem.stage2_storage.scores, smem.stage2_storage.indices);

      if (threadIdx.x < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
        scores_out_batch[out_offset] = smem.stage2_storage.scores[threadIdx.x];
        indices_out_batch[out_offset] = smem.stage2_storage.indices[threadIdx.x];
      }
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(p_scores_in, p_scores_out);
    swap_ptr(p_indices_in, p_indices_out);
    grid.sync();
  }
}

// --- Unified Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int kBlockSize = 256;
  const int partition_size = data->flash_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, partition_size);

  // Determine the number of reduction loops to find the final output buffer
  int num_reduction_loops = 0;
  if (num_partitions > 1) {
    int partitions = num_partitions;
    while (partitions > 1) {
      partitions = (partitions + kReductionFactor - 1) / kReductionFactor;
      num_reduction_loops++;
    }
  }

  // Stage 1 writes to buffer 1 (data->intermediate_..._1)
  // After odd loops, result is in buffer 2. After even loops, result is in buffer 1.
  if (num_reduction_loops % 2 == 1) {
    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
  } else {
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
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
    k_padded_val = kFlashSortMaxK;
  data->topk_stride = k_padded_val;

  void* kernel_args[6];
  kernel_args[0] = (void*)&scores_in;
  kernel_args[1] = (void*)&data->intermediate_indices_1;
  kernel_args[2] = (void*)&data->intermediate_scores_1;
  kernel_args[3] = (void*)&data->intermediate_indices_2;
  kernel_args[4] = (void*)&data->intermediate_scores_2;
  kernel_args[5] = (void*)&vocab_size;

  // This lambda handles selecting the kernel and launching it with the correct K_PADDED value.
  auto launch_flash_sort = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 block(kBlockSize);
    dim3 grid(num_partitions, batch_size);
    switch (partition_size) {
      case 1024:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream));
        break;
      case 2048:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream));
        break;
      default:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortKernel<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream));
        break;
    }
  };

  // Select the padded K value at runtime and call the launch logic.
  if (k <= 4) {
    launch_flash_sort(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    launch_flash_sort(std::integral_constant<int, 8>());
  } else if (k <= 16) {
    launch_flash_sort(std::integral_constant<int, 16>());
  } else if (k <= 32) {
    launch_flash_sort(std::integral_constant<int, 32>());
  } else if (k <= 64) {
    launch_flash_sort(std::integral_constant<int, 64>());
  } else {
    launch_flash_sort(std::integral_constant<int, kFlashSortMaxK>());
  }

  CUDA_CHECK_LAUNCH();
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  ;
  return static_cast<size_t>(batch_size) * num_partitions * kFlashSortMaxK;
}

inline int EstimateBestPartitionSize(int vocab_size) {
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  // The kernle can support partition size up to 8192.
  // That could help better coverage of cooperative support since num_partitions is smaller.
  // However, it is slower for our target use case in benchmark so we exclude it for this kernel.
  return 4096;
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kFlashSortMaxK) {
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
        return (void*)FlashSortKernel<K_PADDED, kBlockSize, 1024>;
      case 2048:
        return (void*)FlashSortKernel<K_PADDED, kBlockSize, 2048>;
      default:
        return (void*)FlashSortKernel<K_PADDED, kBlockSize, 4096>;
    }
  };

  void* kernel = nullptr;
  if (k <= 4) {
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
    kernel = get_kernel(std::integral_constant<int, kFlashSortMaxK>());
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

}  // namespace flash_sort
}  // namespace cuda
}  // namespace Generators
