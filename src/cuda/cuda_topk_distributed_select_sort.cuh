// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include "cuda_topk.h"
#include <cub/cub.cuh>
#include "cuda_topk_select_sort.cuh"

namespace Generators {
namespace cuda {
namespace distributed_select_sort {

/**
 * @brief A distributed selection sort that partitions the vocabulary across Streaming Multiprocessors (SMs).
 *
 * Algorithm Overview:
 * This is a highly specialized two-stage algorithm designed for extremely large vocabularies
 * where `batch_size` is small (typically 1).
 *
 * 1.  **Stage 1 (SM-Level Selection)**: The kernel is launched with a grid of thread blocks
 * where each block corresponds to a "shard" and is intended to run on a separate SM.
 * Each block independently performs a full selection sort (`select_sort::GetTopKKernel`) on its
 * assigned partition of the vocabulary. The local Top-K results from each block are written
 * to a shared global buffer.
 *
 * 2.  **Synchronization**: After completing its work, each block atomically increments a global lock counter.
 * A designated "reducer" block (`blockIdx.z == 0`) spins on this counter, waiting for all
 * other blocks to finish Stage 1. This is a form of manual grid-wide synchronization.
 *
 * 3.  **Stage 2 (Final Reduction)**: Once all blocks are done, the reducer block loads all the
 * intermediate Top-K candidates (`num_shards * k`) from the global buffer into its own shared memory.
 * It then performs a final, small selection sort on these candidates to find the true global Top-K.
 *
 * Performance Characteristics:
 * -   **Strengths**: Can be very effective for its niche: extremely large vocabularies
 * where partitioning the work at the SM level reduces the search space for each block significantly.
 * -   **Weaknesses**: Highly specialized. Its performance is poor outside of its target scenario
 * (massive vocabulary, small k, batch_size=1). The manual spin-wait synchronization can be
 * inefficient on some architectures.
 * -   **Use Case**: A niche algorithm for memory-bound scenarios with huge vocabularies.
 */

using TopK_Pair = select_sort::TopK_Pair;

// A simple struct to hold a key-value pair for inter-TB reduction.
// Has an indirection element id (`elem_id_indirection`) to point to
// the actual element id in the original array whereas element id
// (`elem_id`) is the local element id with respect to the processing TB.
struct TopK_Pair_Reduce {
  float u = -FLT_MAX;
  int p = -1;
  int p_indirection = -1;

  __device__ __forceinline__ void insert(float elem, int elem_id, int elem_id_indirection) {
    if (elem > u || (elem == u && elem_id_indirection < p_indirection)) {
      u = elem;
      p = elem_id;
      p_indirection = elem_id_indirection;
    }
  }

  __device__ __forceinline__ void init() {
    u = -FLT_MAX;
    p = -1;
    p_indirection = -1;
  }
};

// CUB reduction operator for finding the maximum value and its index.
__device__ __forceinline__ TopK_Pair_Reduce reduce_topk_pair_reduce_op(TopK_Pair_Reduce const& a, TopK_Pair_Reduce const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p_indirection < b.p_indirection) ? a
                                                                           : b;
}

// TODO(hasesh): Experiment with using co-operative groups for synchronization in the following kernel
// to see if it can give tangible perf gains
template <int kBlockSize>
__global__ void GetTopKKernelDistributedSelectSort(float* scores_in, float* scores_out, int* indices_out,
                                                   int vocab_size, int k,
                                                   int* top_k_distributed_lock, int* distributed_indices_out,
                                                   float* distributed_scores_out) {
  // Each block processes a different shard of the vocabulary.
  int top_k_shard = blockIdx.z;
  int tid = threadIdx.x;

  int num_top_k_shards = gridDim.z;

  TopK_Pair partial;

  // Use a very small number to blank out selected scores, avoiding picking them again.
  constexpr float MIN_FLOAT = -std::numeric_limits<float>::max();

  int* distributed_indices_out_curr = distributed_indices_out + top_k_shard * k;
  float* distributed_scores_out_curr = distributed_scores_out + top_k_shard * k;

  int start_i = kBlockSize * top_k_shard + tid;

  for (int ite = 0; ite < k; ite++) {
    partial.Init();

    for (auto elemId = start_i; elemId < vocab_size; elemId += kBlockSize * num_top_k_shards) {
      float elem = scores_in[elemId];
      partial.Insert(elem, elemId);
    }

    // reduce in thread block
    typedef cub::BlockReduce<TopK_Pair, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_Pair top_k_sequence = BlockReduce(temp_storage).Reduce(partial, select_sort::reduce_topk_op);

    if (tid == 0) {
      distributed_scores_out_curr[ite] = top_k_sequence.u;
      distributed_indices_out_curr[ite] = top_k_sequence.p;
      scores_in[top_k_sequence.p] = MIN_FLOAT;
      __threadfence_block();
    }
    __syncthreads();
  }

  // --- Synchronization ---
  // Ensure all global writes from Stage 1 are visible.
  __threadfence();
  __syncthreads();

  // Signal completion by incrementing the atomic lock.
  if (threadIdx.x == 0) {
    atomicAdd(top_k_distributed_lock, 1);
  }

  // --- Stage 2: Final Reduction by a single block ---
  if (blockIdx.z == 0) {
    // The reducer block spins until the lock indicates all other blocks are done.
    if (threadIdx.x == 0) {
      int count_of_completed_TBs = 0;

      asm volatile("ld.volatile.global.s32 %0, [%1];"
                   : "=r"(count_of_completed_TBs)
                   : "l"(top_k_distributed_lock));
      while (count_of_completed_TBs < num_top_k_shards) {
        asm volatile("ld.volatile.global.s32 %0, [%1];"
                     : "=r"(count_of_completed_TBs)
                     : "l"(top_k_distributed_lock));
      }

      // Reset the lock for next kernel call
      atomicExch(top_k_distributed_lock, 0);
    }

    // Number of shards is atmost `kTopKDistributedSelectSortMaxShards` and top_k for this kernel is atmost `kTopKDistributedSelectSortMaxTopK`
    __shared__ float shared_distributed_scores_out[topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK];
    __shared__ int shared_distributed_indices_out[topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK];

    // Load distributed results into shared memory for fast access
    for (int i = threadIdx.x; i < num_top_k_shards * k; i += kBlockSize) {
      shared_distributed_scores_out[i] = distributed_scores_out[i];
      shared_distributed_indices_out[i] = distributed_indices_out[i];
    }

    __syncthreads();

    // Perform the reduction
    TopK_Pair_Reduce reduce_partial;
    for (int ite = 0; ite < k; ite++) {
      reduce_partial.init();

      for (int i = threadIdx.x; i < num_top_k_shards * k; i += kBlockSize) {
        float score = shared_distributed_scores_out[i];
        int vocab_index = shared_distributed_indices_out[i];
        reduce_partial.insert(score, i, vocab_index);
      }

      // reduce in thread block
      typedef cub::BlockReduce<TopK_Pair_Reduce, kBlockSize> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      TopK_Pair_Reduce top_k_sequence_reduced = BlockReduce(temp_storage).Reduce(reduce_partial, reduce_topk_pair_reduce_op);

      if (tid == 0) {
        int index = top_k_sequence_reduced.p;
        int vocab_index = top_k_sequence_reduced.p_indirection;

        scores_out[ite] = top_k_sequence_reduced.u;
        indices_out[ite] = vocab_index;
        shared_distributed_scores_out[index] = MIN_FLOAT;

        __threadfence_block();
      }

      __syncthreads();
    }
  }
}

void LaunchGetDistributedSelectSortTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size,
                                        int k, int top_k_shards,
                                        int* top_k_distributed_lock, int* distributed_indices_out, float* distributed_scores_out) {
  const int block_size = 1024;
  dim3 grid(1, 1, top_k_shards);
  dim3 block(block_size, 1, 1);
  GetTopKKernelDistributedSelectSort<block_size><<<grid, block, 0, stream>>>(scores_in, scores_out, indices_out, vocab_size, k,
                                                                             top_k_distributed_lock,
                                                                             distributed_indices_out,
                                                                             distributed_scores_out);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int /*batch_size*/, int k) {
  float* topk_scores = data->intermediate_scores_1;
  int* topk_indices = data->intermediate_indices_1;

  // This algorithm modifies the input scores, so a copy is required.
  float* mutable_scores = data->intermediate_scores_2;
  size_t buffer_size = static_cast<size_t>(vocab_size) * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));

  LaunchGetDistributedSelectSortTopK(stream, mutable_scores, topk_scores, topk_indices, vocab_size, k,
                                     data->top_k_distributed_select_sort_shards,
                                     data->top_k_distributed_select_sort_lock,
                                     data->top_k_distributed_select_sort_keys,
                                     data->top_k_distributed_select_sort_values);
  CUDA_CHECK_LAUNCH();

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}

// This algorithm is only supported for a specific set of problem dimensions.
bool IsSupported(int batch_size, int vocab_size, int k) {
  return (batch_size <= topk_impl_details::kTopKDistributedSelectSortMaxBatchSize) &&
         (k <= topk_impl_details::kTopKDistributedSelectSortMaxTopK) &&
         (vocab_size >= topk_impl_details::kTopKDistributedSelectSortMinVocabSize);
}

}  // namespace distributed_select_sort
}  // namespace cuda
}  // namespace Generators
