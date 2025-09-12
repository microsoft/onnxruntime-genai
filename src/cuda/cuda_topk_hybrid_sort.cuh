// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include "cuda_topk.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace hybrid_sort {
// Stage 1 of Hybrid Sort: Finds the top-k elements within partitions of the vocabulary.
// This single, unified kernel handles both padded and non-padded cases with the "on-the-fly" padding logic.
template <int kBlockSize, int kPartitionSize, int K>
__global__ void HybridSort_Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                                     int* __restrict__ intermediate_indices,
                                                     float* __restrict__ intermediate_scores,
                                                     int vocab_size, int num_partitions) {
  constexpr int ItemsPerThread = kPartitionSize / kBlockSize;
  typedef cub::BlockRadixSort<float, kBlockSize, ItemsPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;
  const int partition_start = partition_idx * kPartitionSize;
  const float* batch_scores_in = scores_in + static_cast<size_t>(batch_idx) * vocab_size;

  float thread_keys[ItemsPerThread];
  int thread_values[ItemsPerThread];

  // Coalesced load from global memory. The boundary check handles both the standard
  // case and the "on-the-fly" padding for the Flash version, where some threads
  // in the final partition will deliberately read out of bounds of the original
  // vocab_size and generate a sentinel value instead.
  for (int i = 0; i < ItemsPerThread; ++i) {
    int global_idx = partition_start + threadIdx.x + i * kBlockSize;
    if (global_idx < vocab_size) {
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
    size_t offset = (static_cast<size_t>(batch_idx) * num_partitions + partition_idx) * K;
    intermediate_scores[offset + threadIdx.x] = thread_keys[0];
    intermediate_indices[offset + threadIdx.x] = thread_values[0];
  }
}

// Helper to calculate the size of intermediate buffers needed by hybrid sort.
inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  return static_cast<size_t>(batch_size) * num_partitions * kHybridSortMaxK;
}

// Selects the most efficient reduction size (partitions per block) for a given number of partitions.
// The goal is to minimize the number of "wasted" slots in the final, partially-filled thread block.
// In case of a tie in waste, the larger reduction size is preferred to minimize kernel launch overhead.
inline int select_partitions_per_block(int num_partitions) {
  if (num_partitions <= 2) return 2;
  if (num_partitions <= 4) return 4;

  int best_p_size = 8;
  int min_waste = ((num_partitions + 7) / 8) * 8 - num_partitions;
  if (min_waste == 0) return 8;

  const int p4_waste = ((num_partitions + 3) / 4) * 4 - num_partitions;
  if (p4_waste < min_waste) {
    min_waste = p4_waste;
    best_p_size = 4;
  }

  const int p2_waste = ((num_partitions + 1) / 2) * 2 - num_partitions;
  if (p2_waste < min_waste) {
    best_p_size = 2;
  }

  return best_p_size;
}

// Kernel for the special case where the vocab fits in one partition. It takes the
// unsorted top-K candidates from Stage 1 and performs a final sort in shared
// memory to produce the true top-k result.
template <int kBlockSize, int K_padded>
__global__ void FinalSinglePartitionSort(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                         float* __restrict__ scores_out, int* __restrict__ indices_out, int k_final) {
  __shared__ float smem_scores[K_padded];
  __shared__ int smem_indices[K_padded];

  const int batch_idx = blockIdx.y;
  const size_t in_offset = static_cast<size_t>(batch_idx) * K_padded;
  const size_t out_offset = static_cast<size_t>(batch_idx) * k_final;

  // Load the K_padded candidates into shared memory
  for (int i = threadIdx.x; i < K_padded; i += kBlockSize) {
    smem_scores[i] = scores_in[in_offset + i];
    smem_indices[i] = indices_in[in_offset + i];
  }
  __syncthreads();

  // Sort the K_padded candidates in shared memory
  bitonic_sort::SharedMemBitonicSort_SoA<kBlockSize, K_padded>(smem_scores, smem_indices);

  // Write out the final top-k_final results
  if (threadIdx.x < k_final) {
    scores_out[out_offset + threadIdx.x] = smem_scores[threadIdx.x];
    indices_out[out_offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

// Stage 2 of Hybrid Sort: Iteratively reduces partitions to find the final top-K.
// This logic is shared between the standard and Flash versions.
template <int K>
void HybridSort_ReducePartitions(TopkData* data, cudaStream_t stream, int num_partitions, int batch_size, int k) {
  if (num_partitions == 1) {
    // Special case: Vocab fits in one partition. Stage 1 found the top K candidates,
    // but they are not fully sorted. We must perform a final, single-block sort on
    // these K candidates to get the true top k. This is more efficient than running
    // the multi-partition reduction loop for this scenario.
    constexpr int block_size = 256;
    dim3 grid(1, batch_size);
    dim3 block(block_size);

    FinalSinglePartitionSort<block_size, K><<<grid, block, 0, stream>>>(
        data->intermediate_scores_1.get(), data->intermediate_indices_1.get(),
        data->intermediate_scores_2.get(), data->intermediate_indices_2.get(),
        k);
    CUDA_CHECK(cudaGetLastError());

    // The final sorted data is in buffer 2. The output is compact with size k.
    data->topk_scores = data->intermediate_scores_2.get();
    data->topk_indices = data->intermediate_indices_2.get();
    data->topk_stride = k;
    return;
  }

  int current_num_partitions = num_partitions;
  float* input_scores = data->intermediate_scores_1.get();
  float* output_scores = data->intermediate_scores_2.get();
  int* input_indices = data->intermediate_indices_1.get();
  int* output_indices = data->intermediate_indices_2.get();
  constexpr int block_size = 256;

  while (current_num_partitions > 1) {
    const int partitions_per_block = select_partitions_per_block(current_num_partitions);
    const int num_blocks = (current_num_partitions + partitions_per_block - 1) / partitions_per_block;
    dim3 grid_reduce(num_blocks, batch_size);
    dim3 block_reduce(block_size);

    switch (partitions_per_block) {
      case 8:
        bitonic_sort::reduction::BlockReduceTopK_SoA<block_size, K, 8><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      case 4:
        bitonic_sort::reduction::BlockReduceTopK_SoA<block_size, K, 4><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
      case 2:
        bitonic_sort::reduction::BlockReduceTopK_SoA<block_size, K, 2><<<grid_reduce, block_reduce, 0, stream>>>(input_scores, input_indices, output_scores, output_indices, current_num_partitions);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    std::swap(input_scores, output_scores);
    std::swap(input_indices, output_indices);
    current_num_partitions = num_blocks;
  }

  data->topk_scores = input_scores;
  data->topk_indices = input_indices;
  data->topk_stride = K;
  CUDA_CHECK(cudaGetLastError());
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int block_size = 256;

  const int partition_size = data->hybrid_sort_partition_size;
  const int num_partitions = (vocab_size + partition_size - 1) / partition_size;
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  auto launch_stage1_flash = [&](auto k_const) {
    constexpr int K = decltype(k_const)::value;
    switch (partition_size) {
      case 1024:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 1024, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 2048:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 2048, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 4096:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 4096, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      case 8192:
        HybridSort_Stage1_FindPartitionsTopK<block_size, 8192, K><<<grid_stage1, block_stage1, 0, stream>>>(
            scores_in, data->intermediate_indices_1.get(), data->intermediate_scores_1.get(), vocab_size, num_partitions);
        break;
      default:
        assert(false && "Unsupported partition_size");
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    HybridSort_ReducePartitions<K>(data, stream, num_partitions, batch_size, k);
  };

  if (k <= 64) {
    launch_stage1_flash(std::integral_constant<int, 64>());
  } else {
    launch_stage1_flash(std::integral_constant<int, kHybridSortMaxK>());
  }
}

/**
 * @brief Estimates the best partition size for the hybrid Top-K sorting algorithm.
 */
inline int EstimateBestPartitionSize(int vocab_size) {
  if (vocab_size <= 1024) return 1024;
  if (vocab_size <= 2048) return 2048;
  if (vocab_size <= 4096) return 4096;
  if (vocab_size <= 8192) return 8192;

  return 8192;
}

} // namespace hybrid_sort
}  // namespace cuda
}  // namespace Generators
