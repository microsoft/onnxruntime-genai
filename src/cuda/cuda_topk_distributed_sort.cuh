// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include <cub/cub.cuh>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// A simple struct to hold a key-value pair for reduction.
struct KVPair {
  float value;
  int index;

  __device__ __forceinline__ KVPair() : value(-FLT_MAX), index(-1) {}
  __device__ __forceinline__ KVPair(float v, int i) : value(v), index(i) {}
};

// Custom reduction operator for KVPair
__device__ __forceinline__ KVPair reduce_kv_op(const KVPair& a, const KVPair& b) {
    if (a.value > b.value) return a;
    if (b.value > a.value) return b;
    return (a.index < b.index) ? a : b;
}

// Distributed Top-K kernel ported from the v0 implementation.
// It consists of two main parts:
// 1. Sharded candidate search: Each thread block (shard) finds its local top-k candidates
//    from a strided view of the vocabulary.
// 2. Reduction: A master shard for each batch reduces the candidates from all shards to
//    find the final top-k elements.
template <int kBlockSize>
__global__ void DistributedTopKKernel_v0(
    float* scores_in,
    int* final_indices_out,
    float* final_scores_out,
    int vocab_size,
    int k,
    int top_k_shards,
    int* top_k_distributed_lock,
    int* distributed_indices_out,
    float* distributed_scores_out) {

    const int batch_idx = blockIdx.y;
    const int shard_idx = blockIdx.x;
    const int tid = threadIdx.x;

    float* current_scores = scores_in + (size_t)batch_idx * vocab_size;
    int* current_final_indices_out = final_indices_out + (size_t)batch_idx * k;
    float* current_final_scores_out = final_scores_out + (size_t)batch_idx * k;

    // Each batch has its own area in the distributed candidate buffers
    const size_t distributed_buffer_stride = (size_t)top_k_shards * kDistributedSortMaxK;
    int* current_distributed_indices = distributed_indices_out + batch_idx * distributed_buffer_stride;
    float* current_distributed_scores = distributed_scores_out + batch_idx * distributed_buffer_stride;
    
    // Each shard gets a pointer to its designated area in the intermediate buffer
    int* shard_distributed_indices = current_distributed_indices + shard_idx * k;
    float* shard_distributed_scores = current_distributed_scores + shard_idx * k;

    const int start_idx = shard_idx * kBlockSize + tid;
    const int stride = gridDim.x * kBlockSize; // gridDim.x is top_k_shards

    // Part 1: Each shard finds its local top-k candidates.
    // NOTE: This part has a potential data race. Multiple shards read from and write to
    // the same `current_scores` buffer without grid-level synchronization inside the loop.
    // Replicating the original's behavior as requested.
    KVPair partial;
    for (int ite = 0; ite < k; ite++) {
        partial.value = -FLT_MAX;
        partial.index = -1;

        for (auto elemId = start_idx; elemId < vocab_size; elemId += stride) {
            float elem = current_scores[elemId];
            if (elem > partial.value || (elem == partial.value && elemId < partial.index)) {
                partial.value = elem;
                partial.index = elemId;
            }
        }
        
        using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        KVPair top_k_shard = BlockReduce(temp_storage).Reduce(partial, reduce_kv_op);

        if (tid == 0) {
            shard_distributed_scores[ite] = top_k_shard.value;
            shard_distributed_indices[ite] = top_k_shard.index;
            if (top_k_shard.index != -1) {
                current_scores[top_k_shard.index] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    // Grid-level barrier to ensure all shards have written their candidates.
    __threadfence();
    __syncthreads();

    if (tid == 0) {
        atomicAdd(top_k_distributed_lock, 1);
    }
    
    // Part 2: The master shard (shard_idx == 0) reduces the candidates.
    if (shard_idx == 0) {
        // Spin-wait for all blocks in the grid to finish Part 1.
        if (tid == 0) {
            int count_of_completed_TBs = 0;
            while (count_of_completed_TBs < gridDim.x * gridDim.y) {
                 asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(count_of_completed_TBs) : "l"(top_k_distributed_lock));
            }
            atomicExch(top_k_distributed_lock, 0);
        }
        __syncthreads();

        // Perform selection sort on the collected candidates.
        const int num_candidates = top_k_shards * k;
        for (int ite = 0; ite < k; ite++) {
            partial.value = -FLT_MAX;
            partial.index = -1;

            for (int i = tid; i < num_candidates; i += kBlockSize) {
                float score = current_distributed_scores[i];
                if (score > partial.value || (score == partial.value && current_distributed_indices[i] < partial.index)) {
                     partial.value = score;
                     partial.index = i;
                }
            }
            
            using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            KVPair top_candidate = BlockReduce(temp_storage).Reduce(partial, reduce_kv_op);

            if (tid == 0) {
                current_final_scores_out[ite] = top_candidate.value;
                if (top_candidate.index != -1) {
                    current_final_indices_out[ite] = current_distributed_indices[top_candidate.index];
                    current_distributed_scores[top_candidate.index] = -FLT_MAX;
                } else {
                    current_final_indices_out[ite] = -1;
                }
            }
            __syncthreads();
        }
    }
}


void RunTopKViaDistributedSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  // This kernel modifies scores_in, so we need to copy it first.
  float* mutable_scores = data->intermediate_scores_2.get();
  size_t buffer_size = static_cast<size_t>(batch_size) * vocab_size * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));

  float* topk_scores = data->intermediate_scores_1.get();
  int* topk_indices = data->intermediate_indices_1.get();
  
  dim3 grid(data->top_k_shards, batch_size);
  dim3 block(1024);

  DistributedTopKKernel_v0<1024><<<grid, block, 0, stream>>>(
      mutable_scores, 
      topk_indices, 
      topk_scores,
      vocab_size, 
      k,
      data->top_k_shards,
      data->top_k_distributed_lock.get(),
      data->top_k_distributed_keys.get(),
      data->top_k_distributed_values.get());
      
  CUDA_CHECK_LAUNCH();
  
  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}

}  // namespace cuda
}  // namespace Generators

