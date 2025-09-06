// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>  // For FLT_MAX

namespace Generators {
namespace cuda {
namespace bitonic {

struct KeyValue {
  float score;
  int index;
};

// Performs a full bitonic sort in shared memory for `SortSize` elements.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(KeyValue* smem_data) {
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          KeyValue* a = &smem_data[i];
          KeyValue* b = &smem_data[ixj];
          bool is_greater = (a->score > b->score) || (a->score == b->score && a->index < b->index);
          if (is_greater != ascending) {
            KeyValue temp = *a;
            *a = *b;
            *b = temp;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int j = SortSize >> 1; j > 0; j >>= 1) {
    for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
      int ixj = i ^ j;
      if (ixj > i) {
        KeyValue* a = &smem_data[i];
        KeyValue* b = &smem_data[ixj];
        if ((a->score < b->score) || (a->score == b->score && a->index > b->index)) {
          KeyValue temp = *a;
          *a = *b;
          *b = temp;
        }
      }
    }
    __syncthreads();
  }
}

namespace reduction {
template <int N>
__device__ void RegisterBitonicSort(float scores[N], int indices[N]) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool ascending = ((i & k) == 0);
          bool is_greater = (scores[i] > scores[ixj]) || (scores[i] == scores[ixj] && indices[i] < indices[ixj]);
          if (is_greater != ascending) {
            float temp_s = scores[i];
            scores[i] = scores[ixj];
            scores[ixj] = temp_s;
            int temp_i = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = temp_i;
          }
        }
      }
    }
  }
  for (int j = N >> 1; j > 0; j >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int ixj = i ^ j;
      if (ixj > i) {
        if ((scores[i] < scores[ixj]) || (scores[i] == scores[ixj] && indices[i] > indices[ixj])) {
          float temp_s = scores[i];
          scores[i] = scores[ixj];
          scores[ixj] = temp_s;
          int temp_i = indices[i];
          indices[i] = indices[ixj];
          indices[ixj] = temp_i;
        }
      }
    }
  }
}

template <int kBlockSize, int K, int PartitionsPerBlock>
__global__ void BlockReduceTopK(const float* __restrict__ scores_in, const int* __restrict__ indices_in,
                                float* __restrict__ scores_out, int* __restrict__ indices_out, int num_partitions_in) {
  constexpr int SortSize = K * PartitionsPerBlock;
  __shared__ KeyValue smem_buffer[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const int in_base_offset = batch_idx * num_partitions_in * K;
  const int out_base_offset = (batch_idx * gridDim.x + blockIdx.x) * K;

  for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
    if (i < K * num_partitions_to_process) {
      int partition_idx = i / K;
      int element_idx = i % K;
      int global_offset = in_base_offset + (block_start_partition + partition_idx) * K + element_idx;
      smem_buffer[i].score = scores_in[global_offset];
      smem_buffer[i].index = indices_in[global_offset];
    } else {
      smem_buffer[i].score = -FLT_MAX;
      smem_buffer[i].index = -1;
    }
  }
  __syncthreads();

  SharedMemBitonicSort<kBlockSize, SortSize>(smem_buffer);

  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_buffer[threadIdx.x].index;
    scores_out[out_base_offset + threadIdx.x] = smem_buffer[threadIdx.x].score;
  }
}
}  // namespace reduction

}  // namespace bitonic
}  // namespace cuda
}  // namespace Generators
