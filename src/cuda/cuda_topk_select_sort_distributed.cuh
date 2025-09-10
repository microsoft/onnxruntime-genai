// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include <limits.h>
#include <cub/cub.cuh>
#include "cuda_topk.h"
#include "cuda_common.h"
#include "cuda_topk_bitonic_sort_helper.cuh"

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

  // Part 2: The master shard (shard_idx == 0) for each batch reduces the candidates using a parallel bitonic sort.
  if (shard_idx == 0) {
    if (tid == 0) {
      int count_of_completed_TBs = 0;
      while (count_of_completed_TBs < num_shards) {
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(count_of_completed_TBs) : "l"(&top_k_distributed_lock[batch_idx]));
      }
    }
    __syncthreads();

    // The maximum number of candidates is 32 shards * 64 k = 2048.
    constexpr int kMaxSortSize = 2048;
    __shared__ float smem_scores[kMaxSortSize];
    __shared__ int smem_indices[kMaxSortSize];

    const int num_total_candidates = num_shards * k;

    // Calculate the smallest power of two that is >= num_total_candidates for the bitonic sort.
    int sort_size = 1;
    while(sort_size < num_total_candidates) {
        sort_size <<= 1;
    }
    if (sort_size > kMaxSortSize) sort_size = kMaxSortSize;

    // Cooperatively load candidates into shared memory.
    for (int i = tid; i < sort_size; i += kBlockSize) {
        if (i < num_total_candidates) {
            int shard_scan = i / k;
            int k_scan = i % k;
            int candidate_buffer_idx = shard_scan * kDistributedSortMaxK + k_scan;
            smem_scores[i] = batch_distributed_scores[candidate_buffer_idx];
            smem_indices[i] = batch_distributed_indices[candidate_buffer_idx];
        } else {
            // Pad the rest with sentinel values for the sort
            smem_scores[i] = -FLT_MAX;
            smem_indices[i] = -1;
        }
    }
    __syncthreads();

    // Perform the sort on the data in shared memory using a switch to call the correctly templated function.
    switch (sort_size) {
        case 2:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2>(smem_scores, smem_indices); break;
        case 4:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 4>(smem_scores, smem_indices); break;
        case 8:    bitonic::SharedMemBitonicSort_SoA<kBlockSize, 8>(smem_scores, smem_indices); break;
        case 16:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 16>(smem_scores, smem_indices); break;
        case 32:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 32>(smem_scores, smem_indices); break;
        case 64:   bitonic::SharedMemBitonicSort_SoA<kBlockSize, 64>(smem_scores, smem_indices); break;
        case 128:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 128>(smem_scores, smem_indices); break;
        case 256:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 256>(smem_scores, smem_indices); break;
        case 512:  bitonic::SharedMemBitonicSort_SoA<kBlockSize, 512>(smem_scores, smem_indices); break;
        case 1024: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 1024>(smem_scores, smem_indices); break;
        case 2048: bitonic::SharedMemBitonicSort_SoA<kBlockSize, 2048>(smem_scores, smem_indices); break;
        default: if(sort_size > 1) { /* This case should ideally not be hit with the logic above */ } break;
    }
     __syncthreads();

    // The first k threads write out the final top-k results.
    if (tid < k) {
        current_final_scores_out[tid] = smem_scores[tid];
        current_final_indices_out[tid] = smem_indices[tid];
    }
  }
}

void RunTopKViaDistributedSelectionSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
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