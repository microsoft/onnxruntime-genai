// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/util_type.cuh>  // Required for cub::Traits
#include <cuda_runtime.h>
#include <float.h>  // For FLT_MAX
#include "cuda_topk.h"
#include "cuda_topk_common.cuh"
#include "cuda_topk_stable_sort_helper.cuh"

namespace Generators {
namespace cuda {
namespace radix_sort {

constexpr int kMaxPartitions = 64;

// These are the partition sizes for which the Stage 1 kernel has template specializations.
constexpr std::array<int, 4> kSupportedPartitionSizes = {2048, 3328, 4352, 4864};

// Flexible heuristic for radix_sort partition size.
inline int EstimateBestPartitionSize(int vocab_size) {
  // 1. Calculate the absolute minimum partition size required to stay under the 64-partition limit.
  const int min_required_size = CeilDiv(vocab_size, kMaxPartitions);

  // 2. Find the smallest supported size that is >= the minimum required size.
  for (int supported_size : kSupportedPartitionSizes) {
    if (supported_size >= min_required_size) {
      return supported_size;
    }
  }

  // 3. If no supported size is large enough, return the largest one.
  //    The assert in RunTopK will catch if this results in too many partitions.
  return kSupportedPartitionSizes.back();
}

inline size_t GetIntermediateSize(int batch_size, int vocab_size, int partition_size) {
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  return static_cast<size_t>(batch_size) * num_partitions * kRadixSortMaxK;
}

// --- Stage 1: Find Top-K within each vocabulary partition ---
// This stage is identical in function to the first stage of llm_sort and flash_sort.
template <int kBlockSize, int kPartitionSize, int K_PADDED>
__global__ void Stage1_FindPartitionsTopK(const float* __restrict__ scores_in,
                                          int* __restrict__ intermediate_indices,
                                          float* __restrict__ intermediate_scores,
                                          int vocab_size, int num_partitions) {
  __shared__ typename Stage1TempStorage smem;
  topk_common::FindPartitionTopK<kBlockSize, kPartitionSize, K_PADDED>(
      scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions, smem);
}

// --- Stage 2: One-Step Reduction Kernel ---
// A single thread block is launched per batch item. It loads ALL candidates from
// Stage 1 into registers, sorts them using a striped layout, and writes the final top-k.
template <int kBlockSize, int K_PADDED, int kMaxPartitions>
__global__ void Stage2_ReduceKernel(const float* __restrict__ scores_in,
                                    const int* __restrict__ indices_in,
                                    float* __restrict__ scores_out,
                                    int* __restrict__ indices_out,
                                    int num_partitions) {
  const int batch_idx = blockIdx.x;
  constexpr int kSortSize = K_PADDED * kMaxPartitions;
  constexpr int kItemsPerThread = CeilDiv(kSortSize, kBlockSize);

#ifdef STABLE_TOPK
  // --- STABLE SORT IMPLEMENTATION ---
  // For stable sort, we combine the score and index into a single 64-bit key.
  // This allows CUB to perform a single, stable sort.
  union SharedStorage {
    struct {
      __align__(128) float scores[K_PADDED];
      __align__(128) int indices[K_PADDED];
    } final_topk;
    typename cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThread>::TempStorage cub_temp_storage;
  };
  __shared__ SharedStorage smem;

  // --- 1. Load data and create combined 64-bit keys ---
  uint64_t thread_keys[kItemsPerThread];
  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
  const int num_elements_to_load = num_partitions * K_PADDED;

  for (int i = 0; i < kItemsPerThread; ++i) {
    int load_idx = threadIdx.x * kItemsPerThread + i;
    if (load_idx < num_elements_to_load) {
      float score = scores_in[in_base_offset + load_idx];
      int index = indices_in[in_base_offset + load_idx];
      thread_keys[i] = topk_common::PackStableSortKey(score, index);
    } else {
      // Corresponds to -FLT_MAX, INT_MAX
      thread_keys[i] = topk_common::PackStableSortKey(-FLT_MAX, INT_MAX);
    }
  }

  // --- 2. Sort the combined keys (descending) ---
  // Sorting these combined keys descending achieves a descending sort by score
  // with a stable, ascending tie-breaker on the index.
  cub::BlockRadixSort<uint64_t, kBlockSize, kItemsPerThread>(smem.cub_temp_storage)
      .SortDescendingBlockedToStriped(thread_keys);

  __syncthreads();

  // --- 3. Write top-K results to shared memory ---
  if (threadIdx.x < K_PADDED) {
    // Unpack the 64-bit key to get the original score and index.
    uint64_t key = thread_keys[0];
    smem.final_topk.scores[threadIdx.x] = topk_common::UnpackStableSortScore(key);
    smem.final_topk.indices[threadIdx.x] = topk_common::UnpackStableSortIndex(key);
  }

#else
  // --- UNSTABLE SORT IMPLEMENTATION (Original) ---
  union SharedStorage {
    struct {
      __align__(128) float scores[K_PADDED];
      __align__(128) int indices[K_PADDED];
    } final_topk;
    typename cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>::TempStorage cub_temp_storage;
  };
  __shared__ SharedStorage smem;

  // --- 1. Load data from Global into Registers (Blocked arrangement) ---
  float thread_scores[kItemsPerThread];
  int thread_indices[kItemsPerThread];
  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions * K_PADDED;
  const int num_elements_to_load = num_partitions * K_PADDED;

  for (int i = 0; i < kItemsPerThread; ++i) {
    int load_idx = threadIdx.x * kItemsPerThread + i;
    if (load_idx < num_elements_to_load) {
      thread_scores[i] = scores_in[in_base_offset + load_idx];
      thread_indices[i] = indices_in[in_base_offset + load_idx];
    } else {
      thread_scores[i] = -FLT_MAX;
      thread_indices[i] = INT_MAX;
    }
  }

  // --- 2. Sort data held in registers ---
  cub::BlockRadixSort<float, kBlockSize, kItemsPerThread, int>(smem.cub_temp_storage)
      .SortDescendingBlockedToStriped(thread_scores, thread_indices);

  __syncthreads();

  // --- 3. Write top-K results from registers to shared memory ---
  // After the striped sort, the top K_PADDED elements are in the first item slot
  // of the first K_PADDED threads.
  if (threadIdx.x < K_PADDED) {
    smem.final_topk.scores[threadIdx.x] = thread_scores[0];
    smem.final_topk.indices[threadIdx.x] = thread_indices[0];
  }
#endif

  __syncthreads();

  // --- 4. Write final top-k results from shared memory to global memory ---
  const size_t out_base_offset = static_cast<size_t>(batch_idx) * K_PADDED;
  if (threadIdx.x < K_PADDED) {
    scores_out[out_base_offset + threadIdx.x] = smem.final_topk.scores[threadIdx.x];
    indices_out[out_base_offset + threadIdx.x] = smem.final_topk.indices[threadIdx.x];
  }
}

// --- Host-Side Launcher ---
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  const int partition_size = EstimateBestPartitionSize(vocab_size);
  const int num_partitions = CeilDiv(vocab_size, partition_size);
  assert(num_partitions <= 64);

  int k_padded_val = kRadixSortMaxK;
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

  auto launch_stage1 = [&](auto k_padded) {
    constexpr int K_PADDED = decltype(k_padded)::value;
    constexpr int kBlockSize = 256;
    dim3 grid(num_partitions, batch_size);
    dim3 block(kBlockSize);

    // Dispatch to the correctly templated kernel based on the calculated partition_size.
    if (partition_size == kSupportedPartitionSizes[0]) {
      Stage1_FindPartitionsTopK<kBlockSize, kSupportedPartitionSizes[0], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else if (partition_size == kSupportedPartitionSizes[1]) {
      Stage1_FindPartitionsTopK<kBlockSize, kSupportedPartitionSizes[1], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else if (partition_size == kSupportedPartitionSizes[2]) {
      Stage1_FindPartitionsTopK<kBlockSize, kSupportedPartitionSizes[2], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    } else {
      Stage1_FindPartitionsTopK<kBlockSize, kSupportedPartitionSizes[3], K_PADDED><<<grid, block, 0, stream>>>(
          scores_in, data->intermediate_indices_1, data->intermediate_scores_1, vocab_size, num_partitions);
    }
  };

  auto dispatch_k = [&](auto dispatcher) {
    if (k_padded_val == 4)
      dispatcher(std::integral_constant<int, 4>());
    else if (k_padded_val == 8)
      dispatcher(std::integral_constant<int, 8>());
    else if (k_padded_val == 16)
      dispatcher(std::integral_constant<int, 16>());
    else if (k_padded_val == 32)
      dispatcher(std::integral_constant<int, 32>());
    else
      dispatcher(std::integral_constant<int, 64>());
  };

  if (num_partitions == 1) {
    dispatch_k(launch_stage1);
    CUDA_CHECK_LAUNCH();
    data->topk_scores = data->intermediate_scores_1;
    data->topk_indices = data->intermediate_indices_1;
    data->topk_stride = k_padded_val;
  } else {
    auto launch_stage2 = [&](auto k_padded) {
      constexpr int K_PADDED = decltype(k_padded)::value;
      constexpr int kBlockSize = 1024;
      dim3 grid(batch_size);
      dim3 block(kBlockSize);
      Stage2_ReduceKernel<kBlockSize, K_PADDED, 64><<<grid, block, 0, stream>>>(
          data->intermediate_scores_1, data->intermediate_indices_1,
          data->intermediate_scores_2, data->intermediate_indices_2, num_partitions);
    };

    dispatch_k([&](auto k_padded) {
      launch_stage1(k_padded);
      CUDA_CHECK_LAUNCH();
      launch_stage2(k_padded);
    });
    CUDA_CHECK_LAUNCH();

    data->topk_scores = data->intermediate_scores_2;
    data->topk_indices = data->intermediate_indices_2;
    data->topk_stride = k_padded_val;
  }
}

bool IsSupported(int batch_size, int vocab_size, int k) {
  if (k > kRadixSortMaxK) {
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
  return true;
}

}  // namespace radix_sort
}  // namespace cuda
}  // namespace Generators
