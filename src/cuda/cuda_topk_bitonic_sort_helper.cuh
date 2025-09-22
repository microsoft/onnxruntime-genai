// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include <math_constants.h>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace bitonic_sort {

/**
 * @brief Performs an in-place bitonic sort on data in shared memory.
 * This specialized version is for when the number of threads (`kBlockSize`)
 * is greater than or equal to the number of items to sort (`SortSize`).
 * Each element is handled by a dedicated thread, leading to high parallelism.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Small(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize >= SortSize);

  // This implementation uses one thread per element for the sort.
  const int ix = threadIdx.x;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      if (ix < SortSize) {
        int paired_ix = ix ^ j;
        if (paired_ix > ix) {
          // A standard bitonic network sorts ascending with `(ix & k) == 0`.
          // The swap condition is inverted to produce a descending sort.
          bool ascending = ((ix & k) == 0);

#ifdef STABLE_TOPK
          // For stable sort, include tie-breaking logic (smaller index wins).
          bool is_ix_greater = (smem_scores[ix] > smem_scores[paired_ix]) ||
                               (smem_scores[ix] == smem_scores[paired_ix] && smem_indices[ix] < smem_indices[paired_ix]);
#else
          // For unstable sort, no tie-breaking is needed for performance.
          bool is_ix_greater = smem_scores[ix] > smem_scores[paired_ix];
#endif

          if (is_ix_greater != ascending) {
            float temp_score = smem_scores[ix];
            smem_scores[ix] = smem_scores[paired_ix];
            smem_scores[paired_ix] = temp_score;

            int temp_index = smem_indices[ix];
            smem_indices[ix] = smem_indices[paired_ix];
            smem_indices[paired_ix] = temp_index;
          }
        }
      }
    }
  }
  __syncthreads();
}

/**
 * @brief A generic, in-place bitonic sort on data in shared memory.
 * This version handles cases where there are fewer threads than elements to sort.
 * Threads loop to cover all necessary comparisons in the sort network.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Big(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0, "SortSize must be power of two");

  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  __syncthreads();

  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < N; i += kBlockSize) {
        const int ixj = i ^ j;
        if (ixj > i) {
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          // A standard bitonic network sorts ascending with `(i & k) == 0`.
          // The swap condition is inverted to produce a descending sort.
          bool ascending = ((i & k) == 0);

#if STABLE_TOPK
          // For stable sort, include tie-breaking logic (smaller index wins).
          bool is_i_greater = (a_i > a_j) || (a_i == a_j && idx_i < idx_j);
#else
          // For unstable sort, no tie-breaking is needed for performance.
          bool is_i_greater = a_i > a_j;
#endif

          if (is_i_greater != ascending) {
            smem_scores[i] = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i] = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
}

/**
 * @brief A dispatch wrapper for shared memory bitonic sort.
 * At compile time, it selects the optimal implementation (`_Small` or `_Big`)
 * based on the relationship between block size and sort size.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(float* smem_scores, int* smem_indices) {
  if constexpr (kBlockSize >= SortSize) {
    SharedMemBitonicSort_Small<kBlockSize, SortSize>(smem_scores, smem_indices);
  } else {
    SharedMemBitonicSort_Big<kBlockSize, SortSize>(smem_scores, smem_indices);
  }
}

template <int kBlockSize, int K, int PartitionsPerBlock>
__global__ void BlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                float* __restrict__ scores_out, int* __restrict__ indices_out, int num_partitions_in) {
  constexpr int SortSize = K * PartitionsPerBlock;
  __shared__ float smem_scores[SortSize];
  __shared__ int smem_indices[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions_in * K;
  const size_t out_base_offset = (static_cast<size_t>(batch_idx) * gridDim.x + blockIdx.x) * K;

  // Load data from global memory into shared memory using an SoA layout
  for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
    if (i < K * num_partitions_to_process) {
      int partition_idx = i / K;
      int element_idx = i % K;
      size_t global_offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K + element_idx;
      smem_scores[i] = scores_in[global_offset];
      smem_indices[i] = indices_in[global_offset];
    } else {
      smem_scores[i] = -FLT_MAX;
      smem_indices[i] = INT_MAX;
    }
  }
  __syncthreads();

  // Perform the sort on the SoA data in shared memory.
  SharedMemBitonicSort<kBlockSize, SortSize>(smem_scores, smem_indices);

  // Write the top K results back to global memory
  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_indices[threadIdx.x];
    scores_out[out_base_offset + threadIdx.x] = smem_scores[threadIdx.x];
  }
}

/**
 * @brief Performs an in-place, warp-wide bitonic sort on data held entirely in registers.
 *
 * This function sorts `warpSize` (typically 32) score/index pairs distributed across the
 * threads of a single warp. It uses `__shfl_sync` instructions for extremely fast
 * data exchange between threads in the same warp, avoiding shared memory latency entirely.
 * This is highly effective for the reduction phase of algorithms like `iterative_sort` when `k` is small.
 */
__device__ inline void WarpBitonicSort(float& score, int& index) {
  const int lane_id = threadIdx.x % warpSize;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= warpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      // Exchange data with a paired lane using a warp shuffle instruction.
      int paired_lane = lane_id ^ j;
      float paired_score = __shfl_sync(0xFFFFFFFF, score, paired_lane);
      int paired_index = __shfl_sync(0xFFFFFFFF, index, paired_lane);

      // A standard bitonic network sorts ascending with `(lane_id & k) == 0`.
      // The swap condition is inverted as needed to produce an overall descending sort.
      bool direction = ((lane_id & k) == 0);

#ifdef STABLE_TOPK
      // For stable sort, include tie-breaking logic (smaller index wins for equal scores).
      bool is_mine_greater = (score > paired_score) || (score == paired_score && index < paired_index);
#else
      bool is_mine_greater = score > paired_score;
#endif

      // In-register min/max calculation.
      float s_max = is_mine_greater ? score : paired_score;
      int i_max = is_mine_greater ? index : paired_index;
      float s_min = is_mine_greater ? paired_score : score;
      int i_min = is_mine_greater ? paired_index : index;

      // Redistribute the min/max values based on the sort direction for this stage.
      if (direction) {
        score = (lane_id < paired_lane) ? s_max : s_min;
        index = (lane_id < paired_lane) ? i_max : i_min;
      } else {
        score = (lane_id < paired_lane) ? s_min : s_max;
        index = (lane_id < paired_lane) ? i_min : i_max;
      }
    }
  }
}

}  // namespace bitonic_sort
}  // namespace cuda
}  // namespace Generators
