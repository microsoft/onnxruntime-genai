// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include <limits.h>
#include <cub/cub.cuh>
#include "cuda_topk.h"
#include "cuda_common.h"

namespace Generators {
namespace cuda {
// A simple struct to hold a key-value pair.
// In the initial search, 'index' is the vocabulary index.
// In the final reduction, 'index' is the intermediate buffer index.
struct KVPair {
  float value;
  int index;

  __device__ __forceinline__ KVPair() : value(-FLT_MAX), index(INT_MAX) {}
};

// Custom reduction operator for KVPair.
// When scores are equal, the one with the smaller index wins.
__device__ __forceinline__ KVPair reduce_kv_op(const KVPair& a, const KVPair& b) {
  if (a.value > b.value) return a;
  if (b.value > a.value) return b;
  return (a.index < b.index) ? a : b;
}

// Read-only kernel specialized for k=1.
// This avoids the overhead of iterating and modifying the input scores buffer.
template <int kBlockSize>
__global__ void DistributedTop1Kernel(
    const float* scores_in,
    int* final_indices_out,
    float* final_scores_out,
    int vocab_size,
    const int num_shards,
    int* top_k_distributed_lock,
    int* distributed_indices_out,
    float* distributed_scores_out) {
  const int batch_idx = blockIdx.y;
  const int shard_idx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ typename cub::BlockReduce<KVPair, kBlockSize>::TempStorage kv_temp_storage;

  const float* current_scores = scores_in + (size_t)batch_idx * vocab_size;
  int* current_final_indices_out = final_indices_out + (size_t)batch_idx * 1;
  float* current_final_scores_out = final_scores_out + (size_t)batch_idx * 1;

  const size_t distributed_buffer_stride_per_batch = (size_t)gridDim.x * kDistributedSortMaxK;
  int* batch_distributed_indices = distributed_indices_out + batch_idx * distributed_buffer_stride_per_batch;
  float* batch_distributed_scores = distributed_scores_out + batch_idx * distributed_buffer_stride_per_batch;
  int* shard_distributed_indices = batch_distributed_indices + shard_idx * kDistributedSortMaxK;
  float* shard_distributed_scores = batch_distributed_scores + shard_idx * kDistributedSortMaxK;

  const int partition_size = (vocab_size + num_shards - 1) / num_shards;
  const int partition_start = shard_idx * partition_size;
  const int partition_end = min((shard_idx + 1) * partition_size, vocab_size);

  // Part 1: Each shard finds its local top-1 candidate.
  KVPair partial;
  for (auto elemId = partition_start + tid; elemId < partition_end; elemId += kBlockSize) {
    const float elem = current_scores[elemId];
    if (elem > partial.value || (elem == partial.value && elemId < partial.index)) {
      partial.value = elem;
      partial.index = elemId;
    }
  }

  using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
  KVPair top_k_shard = BlockReduce(kv_temp_storage).Reduce(partial, reduce_kv_op);

  if (tid == 0) {
    shard_distributed_scores[0] = top_k_shard.value;
    shard_distributed_indices[0] = (top_k_shard.index == INT_MAX) ? -1 : top_k_shard.index;
  }
  __syncthreads();

  if (tid == 0) {
    atomicAdd(&top_k_distributed_lock[batch_idx], 1);
  }

  // Part 2: Master shard reduces candidates.
  if (shard_idx == 0) {
    if (tid == 0) {
      int count_of_completed_TBs = 0;
      while (count_of_completed_TBs < num_shards) {
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(count_of_completed_TBs) : "l"(&top_k_distributed_lock[batch_idx]));
      }
    }
    __syncthreads();

    KVPair partial_candidate; // Using simpler KVPair for reduction
    partial_candidate.index = -1; // Initialize index to invalid

    const int num_total_candidates = num_shards;
    for (int i = tid; i < num_total_candidates; i += kBlockSize) {
      int candidate_buffer_idx = i * kDistributedSortMaxK;
      float score = batch_distributed_scores[candidate_buffer_idx];
      
      // Manually perform comparison to ensure correct tie-breaking
      bool is_better = false;
      if (score > partial_candidate.value) {
        is_better = true;
      } else if (score == partial_candidate.value && score > -FLT_MAX) {
        int current_vocab_idx = batch_distributed_indices[candidate_buffer_idx];
        int best_vocab_idx = (partial_candidate.index == -1) ? INT_MAX : batch_distributed_indices[partial_candidate.index];
        if (current_vocab_idx < best_vocab_idx) {
          is_better = true;
        }
      }

      if (is_better) {
        partial_candidate.value = score;
        partial_candidate.index = candidate_buffer_idx;
      }
    }

    using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
    KVPair top_candidate = BlockReduce(kv_temp_storage).Reduce(partial_candidate, reduce_kv_op);

    if (tid == 0) {
      current_final_scores_out[0] = top_candidate.value;
      current_final_indices_out[0] = (top_candidate.index == -1) ? -1 : batch_distributed_indices[top_candidate.index];
    }
  }
}

template <int kBlockSize>
__global__ void DistributedTopKKernel(
    volatile float* scores_in,
    int* final_indices_out,
    float* final_scores_out,
    int vocab_size,
    int k,
    const int num_shards,
    int* top_k_distributed_lock,
    int* distributed_indices_out,
    float* distributed_scores_out) {
  const int batch_idx = blockIdx.y;
  const int shard_idx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ typename cub::BlockReduce<KVPair, kBlockSize>::TempStorage kv_temp_storage;

  volatile float* current_scores = scores_in + (size_t)batch_idx * vocab_size;
  int* current_final_indices_out = final_indices_out + (size_t)batch_idx * k;
  float* current_final_scores_out = final_scores_out + (size_t)batch_idx * k;

  const size_t distributed_buffer_stride_per_batch = (size_t)gridDim.x * kDistributedSortMaxK;
  int* batch_distributed_indices = distributed_indices_out + batch_idx * distributed_buffer_stride_per_batch;
  float* batch_distributed_scores = distributed_scores_out + batch_idx * distributed_buffer_stride_per_batch;
  int* shard_distributed_indices = batch_distributed_indices + shard_idx * kDistributedSortMaxK;
  float* shard_distributed_scores = batch_distributed_scores + shard_idx * kDistributedSortMaxK;

  const int partition_size = (vocab_size + num_shards - 1) / num_shards;
  const int partition_start = shard_idx * partition_size;
  const int partition_end = min((shard_idx + 1) * partition_size, vocab_size);

  // Part 1: Each shard finds its local top-k candidates.
  for (int ite = 0; ite < k; ite++) {
    KVPair partial;
    for (auto elemId = partition_start + tid; elemId < partition_end; elemId += kBlockSize) {
      float elem = current_scores[elemId];
      if (elem > partial.value || (elem == partial.value && elemId < partial.index)) {
        partial.value = elem;
        partial.index = elemId;
      }
    }

    using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
    KVPair top_k_shard = BlockReduce(kv_temp_storage).Reduce(partial, reduce_kv_op);

    if (tid == 0) {
      shard_distributed_scores[ite] = top_k_shard.value;
      shard_distributed_indices[ite] = (top_k_shard.index == INT_MAX) ? -1 : top_k_shard.index;
      if (top_k_shard.index != INT_MAX) {
        current_scores[top_k_shard.index] = -FLT_MAX;
      }
    }
    __threadfence();
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&top_k_distributed_lock[batch_idx], 1);
  }

  // Part 2: The master shard (shard_idx == 0) for each batch reduces the candidates.
  if (shard_idx == 0) {
    if (tid == 0) {
      int count_of_completed_TBs = 0;
      while (count_of_completed_TBs < num_shards) {
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(count_of_completed_TBs) : "l"(&top_k_distributed_lock[batch_idx]));
      }
    }
    __syncthreads();

    // Perform selection sort on the collected candidates for the current batch.
    for (int ite = 0; ite < k; ite++) {
      // Using simpler KVPair. `partial.index` will store the buffer_idx.
      KVPair partial;
      partial.index = -1; // Initialize with invalid index

      const int num_total_candidates = num_shards * k;
      for (int i = tid; i < num_total_candidates; i += kBlockSize) {
        int shard_scan = i / k;
        int k_scan = i % k;
        int candidate_buffer_idx = shard_scan * kDistributedSortMaxK + k_scan;

        float score = batch_distributed_scores[candidate_buffer_idx];
        
        // Manually perform comparison to ensure correct tie-breaking,
        // as the simple KVPair doesn't carry the vocab_idx needed for reduction.
        bool is_better = false;
        if (score > partial.value) {
            is_better = true;
        } else if (score == partial.value && score > -FLT_MAX) {
            // Tie-break on the original vocabulary index
            int current_vocab_idx = batch_distributed_indices[candidate_buffer_idx];
            // Get vocab_idx of the current best. partial.index is the buffer_idx.
            int best_vocab_idx = (partial.index == -1) ? INT_MAX : batch_distributed_indices[partial.index];
            if (current_vocab_idx < best_vocab_idx) {
                is_better = true;
            }
        }

        if (is_better) {
            partial.value = score;
            partial.index = candidate_buffer_idx;
        }
      }

      using BlockReduce = cub::BlockReduce<KVPair, kBlockSize>;
      // The reduce_kv_op now reduces partials that have already been correctly tie-broken.
      KVPair top_candidate = BlockReduce(kv_temp_storage).Reduce(partial, reduce_kv_op);

      if (tid == 0) {
        if (top_candidate.index != -1) {
            current_final_scores_out[ite] = top_candidate.value;
            // Get the final vocab_idx from the buffer_idx (top_candidate.index)
            current_final_indices_out[ite] = batch_distributed_indices[top_candidate.index];
            // Blank out the winner in the candidate list
            batch_distributed_scores[top_candidate.index] = -FLT_MAX;
        } else {
            current_final_scores_out[ite] = -FLT_MAX;
            current_final_indices_out[ite] = -1;
        }
        __threadfence_block();
      }
      __syncthreads();
    }
  }
}

void RunTopKViaDistributedSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(data != nullptr);
  assert(scores_in != nullptr);
  assert(vocab_size > 0);
  assert(batch_size > 0);
  assert(k > 0 && k <= kDistributedSortMaxK && k <= vocab_size);

  constexpr int kBlockSize = 1024;
  const int num_shards = std::min(data->top_k_shards, CeilDiv(vocab_size, kBlockSize));
  
  CUDA_CHECK(cudaMemsetAsync(data->top_k_distributed_lock.get(), 0, static_cast<size_t>(batch_size) * sizeof(int), stream));

  float* topk_scores = data->intermediate_scores_1.get();
  int* topk_indices = data->intermediate_indices_1.get();

  dim3 grid(num_shards, batch_size);
  dim3 block(kBlockSize);

  if (k == 1) {
    DistributedTop1Kernel<kBlockSize><<<grid, block, 0, stream>>>(
        scores_in,
        topk_indices,
        topk_scores,
        vocab_size,
        num_shards,
        data->top_k_distributed_lock.get(),
        data->top_k_distributed_keys.get(),
        data->top_k_distributed_values.get());
  } else {
    float* mutable_scores = data->intermediate_scores_2.get();
    size_t buffer_size = static_cast<size_t>(batch_size) * vocab_size * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));

    DistributedTopKKernel<kBlockSize><<<grid, block, 0, stream>>>(
        mutable_scores,
        topk_indices,
        topk_scores,
        vocab_size,
        k,
        num_shards,
        data->top_k_distributed_lock.get(),
        data->top_k_distributed_keys.get(),
        data->top_k_distributed_values.get());
  }

  CUDA_CHECK_LAUNCH();

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}

}  // namespace cuda
}  // namespace Generators