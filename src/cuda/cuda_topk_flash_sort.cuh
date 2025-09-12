// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"
#include "cuda_topk_hybrid_sort.cuh"

namespace Generators {
namespace cuda {
namespace flash_sort {
namespace cg = cooperative_groups;

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

// A single, cooperative kernel that performs a multi-stage Top-K sort for batch_size=1.
// It uses cooperative groups to synchronize the entire grid between stages,
// eliminating the high overhead of multiple kernel launches.
// Template parameters:
//   K_PADDED: The compile-time constant for K, padded for efficiency.
//   kBlockSize: The number of threads per block.
//   kPartitionSize: The size of the vocabulary partition each block handles in Stage 1.
template <int K_PADDED, int kBlockSize, int kPartitionSize>
__global__ void FlashSortBs1Kernel(const float* __restrict__ input_scores,
                                   int* __restrict__ final_indices_out,
                                   float* __restrict__ final_scores_out,
                                   int* __restrict__ intermediate_indices_1,
                                   float* __restrict__ intermediate_scores_1,
                                   int* __restrict__ intermediate_indices_2,
                                   float* __restrict__ intermediate_scores_2,
                                   int vocab_size, int k_final) {
  auto grid = cg::this_grid();
  const int partition_idx = blockIdx.x;
  const int num_partitions = gridDim.x;

  // --- Stage 1: Find Top-K within each partition ---
  // Each thread block finds the top K_PADDED candidates from its assigned partition.
  {
    constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
    using BlockRadixSort = cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    const int partition_start = partition_idx * kPartitionSize;
    float thread_keys[ItemsPerThread];
    int thread_values[ItemsPerThread];

    // Load data from global memory
    for (int i = 0; i < ItemsPerThread; ++i) {
      int global_idx = partition_start + threadIdx.x + i * kBlockSize;
      if (global_idx < vocab_size) {
        thread_keys[i] = input_scores[global_idx];
        thread_values[i] = global_idx;
      } else {
        thread_keys[i] = -FLT_MAX;
        thread_values[i] = -1;
      }
    }

    // Sort within the block
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Write the top K_PADDED candidates to the intermediate buffer
    if (threadIdx.x < K_PADDED) {
      size_t offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
      intermediate_scores_1[offset] = thread_keys[0];
      intermediate_indices_1[offset] = thread_values[0];
    }
  }

  grid.sync();

  // --- Stage 2: Iterative Tree Reduction ---
  // Blocks iteratively merge the top-K lists from the previous stage.
  int* indices_in = intermediate_indices_1;
  float* scores_in = intermediate_scores_1;
  int* indices_out = intermediate_indices_2;
  float* scores_out = intermediate_scores_2;

  int partitions_remaining = num_partitions;
  while (partitions_remaining > 1) {
    int num_active_blocks = (partitions_remaining + 1) / 2;
    if (partition_idx < num_active_blocks) {
      constexpr int kSortSize = K_PADDED * 2;
      __shared__ float smem_scores[kSortSize];
      __shared__ int smem_indices[kSortSize];

      int first_child_partition = partition_idx * 2;
      int second_child_partition = first_child_partition + 1;
      int num_partitions_to_process = (second_child_partition < partitions_remaining) ? 2 : 1;

      // Load candidates from two child partitions into shared memory
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        if (i < K_PADDED * num_partitions_to_process) {
          int part_idx = i / K_PADDED;
          int element_idx = i % K_PADDED;
          size_t global_offset = (first_child_partition + part_idx) * K_PADDED + element_idx;
          smem_scores[i] = scores_in[global_offset];
          smem_indices[i] = indices_in[global_offset];
        } else {
          smem_scores[i] = -FLT_MAX;
          smem_indices[i] = -1;
        }
      }
      __syncthreads();

      // Sort the merged list of 2*K_PADDED candidates
      bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, kSortSize>(smem_scores, smem_indices);

      // Write the top K_PADDED results to the output buffer for this reduction level
      if (threadIdx.x < K_PADDED) {
        size_t out_offset = static_cast<size_t>(partition_idx) * K_PADDED + threadIdx.x;
        scores_out[out_offset] = smem_scores[threadIdx.x];
        indices_out[out_offset] = smem_indices[threadIdx.x];
      }
    }
    partitions_remaining = num_active_blocks;
    swap_ptr(scores_in, scores_out);
    swap_ptr(indices_in, indices_out);
    grid.sync();
  }

  // --- Final Output ---
  // Block 0 writes the final top-k results to the output buffers.
  if (partition_idx == 0 && threadIdx.x < k_final) {
    final_scores_out[threadIdx.x] = scores_in[threadIdx.x];
    final_indices_out[threadIdx.x] = indices_in[threadIdx.x];
  }
}

// Host-side launcher for the FlashSort kernel.
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(batch_size == 1); // FlashSort currently only supports batch_size=1.
  assert(k <= kFlashSortMaxK);

  constexpr int kBlockSize = 256;
  const int kPartitionSize = data->hybrid_sort_partition_size;
  const int num_partitions = CeilDiv(vocab_size, kPartitionSize);

  // Lamda function to launch Flash Sort kernel for batch_size=1
  auto launch_bs1_kernel = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    dim3 grid(num_partitions);
    dim3 block(kBlockSize);

    void* kernel_args[9];
    kernel_args[0] = (void*)&scores_in;
    kernel_args[1] = (void*)&data->intermediate_indices_1;  // Final out is determined by reduction
    kernel_args[2] = (void*)&data->intermediate_scores_1;   // Final out is determined by reduction
    kernel_args[3] = (void*)&data->intermediate_indices_1;
    kernel_args[4] = (void*)&data->intermediate_scores_1;
    kernel_args[5] = (void*)&data->intermediate_indices_2;
    kernel_args[6] = (void*)&data->intermediate_scores_2;
    kernel_args[7] = (void*)&vocab_size;
    kernel_args[8] = (void*)&k;

    switch (kPartitionSize) {
      case 1024:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 1024>, grid, block, kernel_args, 0, stream));
        break;
      case 2048:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 2048>, grid, block, kernel_args, 0, stream));
        break;
      case 4096:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 4096>, grid, block, kernel_args, 0, stream));
        break;
      case 8192:
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)FlashSortBs1Kernel<K_PADDED, kBlockSize, 8192>, grid, block, kernel_args, 0, stream));
        break;
    }

    // The final result is in intermediate_indices_1 and intermediate_scores_1.
    // The reduction logic swaps pointers, and the final write goes to these buffers.
    data->topk_scores = data->intermediate_scores_1.get();
    data->topk_indices = data->intermediate_indices_1.get();
  };

  if (k <= 4) {
    launch_bs1_kernel(std::integral_constant<int, 4>());
  } else if (k <= 8) {
    launch_bs1_kernel(std::integral_constant<int, 8>());
  } else if (k <= 16)
    launch_bs1_kernel(std::integral_constant<int, 16>());
  else if (k <= 32)
    launch_bs1_kernel(std::integral_constant<int, 32>());
  else if (k <= 64)
    launch_bs1_kernel(std::integral_constant<int, 64>());
  else
    launch_bs1_kernel(std::integral_constant<int, kFlashSortMaxK>());
  data->topk_stride = k;

  CUDA_CHECK_LAUNCH();
}

} // namespace flash_sort
}  // namespace cuda
}  // namespace Generators
