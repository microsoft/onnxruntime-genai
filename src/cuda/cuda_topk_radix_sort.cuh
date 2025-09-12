// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cub/cub.cuh"
#include "cub/util_type.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cuda_common.h"

#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>

#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace radix_sort {
__global__ void FillInput(const float* input_x_batch, float* temp_v, int* temp_i, int vocab_size) {
  for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < vocab_size; id += blockDim.x * gridDim.x) {
    temp_v[id] = input_x_batch[id];
    temp_i[id] = id;
  }
}

inline size_t GetTempStorageBytes(int vocab_size, cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size, 0, sizeof(float) * 8, stream));
  return temp_storage_bytes;
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  constexpr int block_size = 256;
  int blocks_per_batch = CeilDiv(vocab_size, block_size);

  auto* temp_storage = data->cub_temp_storage.get();
  auto temp_storage_bytes = data->cub_temp_storage_bytes;

  auto* final_scores_buffer = data->intermediate_scores_1.get();
  auto* final_indices_buffer = data->intermediate_indices_1.get();

  auto* workspace_scores = data->intermediate_scores_2.get();
  auto* workspace_indices = data->intermediate_indices_2.get();

  for (int i = 0; i < batch_size; i++) {
    const float* current_scores_in = scores_in + static_cast<size_t>(i) * vocab_size;
  
    // Sort from workspace directly into the final strided destination buffer.
    float* final_scores_out = final_scores_buffer + static_cast<size_t>(i) * vocab_size;
    int* final_indices_out = final_indices_buffer + static_cast<size_t>(i) * vocab_size;

    FillInput<<<blocks_per_batch, block_size, 0, stream>>>(current_scores_in, workspace_scores, workspace_indices, vocab_size);
    cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_bytes, workspace_scores, final_scores_out, workspace_indices, final_indices_out, vocab_size, 0, sizeof(float) * 8, stream);
  }

  data->topk_scores = final_scores_buffer;
  data->topk_indices = final_indices_buffer;
  data->topk_stride = vocab_size;
}
} // namespace radix_sort
}  // namespace cuda
}  // namespace Generators

