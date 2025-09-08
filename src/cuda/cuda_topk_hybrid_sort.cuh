// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
// Stage 1 of Hybrid Sort: Find the top-k elements within large, contiguous partitions of the vocabulary.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void FindBlockTopK_CubRegisterSort(const float* __restrict__ scores_in,
                                              int* __restrict__ intermediate_indices,
                                              float* __restrict__ intermediate_scores, int vocab_size,
                                              int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + batch_idx * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

  // Coalesced load from global memory into per-thread registers.
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size && global_idx < partition_start + kPartitionSize) {
      thread_keys[i] = batch_scores_in[global_idx];
      thread_values[i] = global_idx;
    } else {
      thread_keys[i] = -FLT_MAX;
      thread_values[i] = -1;
    }
  }

  // Sort the keys and values held in registers across the entire block.
  BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

  // The first K threads now hold the top K elements for this partition. Write them out.
  if (threadIdx.x < K) {
    int offset = (batch_idx * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

// Helper to calculate the size of intermediate buffers needed by hybrid sort.
inline size_t GetHybridSortIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  return static_cast<size_t>(batch_size) * num_partitions * kHybridSortMaxK;
}

// Selects the most efficient reduction size (partitions per block) for a given number of partitions.
// The goal is to minimize the number of "wasted" slots in the final, partially-filled thread block.
// In case of a tie in waste, the larger reduction size is preferred to minimize kernel launch overhead.
inline int select_partitions_per_block(int num_partitions) {
  if (num_partitions <= 2) return 2;
  if (num_partitions <= 4) return 4;

  // Start with the largest reduction size as the default.
  int best_p_size = 8;
  // Waste is the number of empty slots we must process.
  int min_waste = ((num_partitions + 7) / 8) * 8 - num_partitions;
  if (min_waste == 0) {
    return 8;  // Perfect alignment, no need to check further.
  }

  // Check if a reduction size of 4 is strictly better.
  const int p4_waste = ((num_partitions + 3) / 4) * 4 - num_partitions;
  if (p4_waste < min_waste) {
    min_waste = p4_waste;
    best_p_size = 4;
  }

  // Check if a reduction size of 2 is strictly better than the current best.
  const int p2_waste = ((num_partitions + 1) / 2) * 2 - num_partitions;
  if (p2_waste < min_waste) {
    best_p_size = 2;
  }

  return best_p_size;
}

void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int max_k = kHybridSortMaxK;
  constexpr int block_size = 256;
  static_assert(kHybridSortMaxK <= block_size);

  int partition_size = data->hybrid_sort_partition_size;

  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  // Stage 1: Find Top-K within partitions.
  // The results are written to intermediate buffers.
  switch (partition_size) {
    case 1024:
      FindBlockTopK_CubRegisterSort<block_size, 1024, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 2048:
      FindBlockTopK_CubRegisterSort<block_size, 2048, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 4096:
      FindBlockTopK_CubRegisterSort<block_size, 4096, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    case 8192:
      FindBlockTopK_CubRegisterSort<block_size, 8192, max_k><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
      break;
    default:
      assert(false && "Unsupported partition_size");
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Iteratively reduce the candidates from each partition until only one partition remains.
  // This uses a ping-pong buffer scheme for scores and indices.
  int current_num_partitions = num_partitions;
  float* input_scores = data->intermediate_scores_1.get();
  float* output_scores = data->intermediate_scores_2.get();
  int* input_indices = data->intermediate_indices_1.get();
  int* output_indices = data->intermediate_indices_2.get();

  while (current_num_partitions > 1) {
    const int partitions_per_block = select_partitions_per_block(current_num_partitions);
    const int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);

    // Dispatch to the kernel with the optimal reduction size.
    switch (partitions_per_block) {
      case 8:
        bitonic::reduction::BlockReduceTopK<block_size, max_k, 8><<<grid_reduce, block_reduce, 0, stream>>>(
            input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      case 4:
        bitonic::reduction::BlockReduceTopK<block_size, max_k, 4><<<grid_reduce, block_reduce, 0, stream>>>(
            input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      case 2:
        bitonic::reduction::BlockReduceTopK<block_size, max_k, 2><<<grid_reduce, block_reduce, 0, stream>>>(
            input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);
    current_num_partitions = num_blocks;
  }


  // After reduction, input_scores and input_indices point to the device buffers containing the final top-`max_k` raw scores and indices.
  data->topk_scores = input_scores;
  data->topk_indices = input_indices;
  data->topk_stride = max_k;
  CUDA_CHECK(cudaGetLastError());
}


/**
 * @brief Estimates the best partition size for the hybrid Top-K sorting algorithm.
 *
 * This function uses a heuristic based on extensive benchmarking and analysis of the
 * hybrid sort's two-stage nature. The goal is to select a partition size that
 * creates an optimal amount of parallel work to saturate the GPU, while minimizing
 * the overhead of the reduction stage.
 *
 * The heuristic is based on the following rules derived from empirical data:
 * 1.  If the vocabulary fits within a single partition, the smallest partition that
 * can contain it is chosen to eliminate the reduction stage overhead.
 * 2.  If the vocabulary is larger than the biggest available partition, 8192 is
 * consistently the best choice across all batch sizes, as it minimizes the
 * number of partitions that need to be processed in the reduction stage.
 *
 * @param vocab_size The size of the vocabulary to sort over.
 * @return The estimated optimal partition size (e.g., 1024, 2048, 4096, 8192).
 */
inline int EstimateHybridSortBestPartitionSize(int vocab_size) {
  // --- Rule 1: Single Partition Dominance ---
  // If the vocabulary fits entirely within a partition, use the smallest one that fits.
  // This is the most efficient case as it completely avoids the reduction stage.
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  if (vocab_size <= 4096) return 4096;
  if (vocab_size <= 8192) return 8192;

  // --- Rule 2: Default to Largest Partition Size ---
  // For any vocabulary size larger than 8192, the benchmark data consistently
  // shows that using the largest partition size (8192) is optimal. This minimizes
  // the number of partitions, making the reduction stage as efficient as possible.
  return 8192;
}

}  // namespace cuda
}  // namespace Generators
