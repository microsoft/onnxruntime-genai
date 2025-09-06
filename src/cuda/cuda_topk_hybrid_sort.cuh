// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {

// Kernel to compact strided data into a dense layout.
// Used to convert data from a [batch, stride] layout to a dense [batch, k] layout.
template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int in_idx = batch_idx * input_stride + i;
    int out_idx = batch_idx * k + i;
    output[out_idx] = input[in_idx];
  }
}

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

void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size) {
  constexpr int max_k = kHybridSortMaxK;
  constexpr int block_size = 256;
  static_assert(max_k <= block_size);

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
    constexpr int partitions_per_block = 8;
    int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);
    bitonic::reduction::BlockReduceTopK<block_size, max_k, partitions_per_block>
        <<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices,
                                                   current_num_partitions);
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

}  // namespace cuda
}  // namespace Generators
