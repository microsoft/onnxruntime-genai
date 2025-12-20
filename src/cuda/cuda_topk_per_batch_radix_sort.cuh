// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cub/cub.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cuda_common.h"
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace per_batch_radix_sort {

/**
 * @brief An algorithm that performs an independent, full radix sort for each batch item.
 *
 * Algorithm Overview:
 * This approach iterates through each item in the batch on the host and launches a
 * separate `cub::DeviceRadixSort` kernel for each one. It fully sorts the `vocab_size`
 * scores for that batch item and stores the result.
 *
 * Performance Characteristics:
 * -   **Strengths**: This method is straightforward and can be efficient for small batch sizes
 * (e.g., `batch_size` <= 8). When the batch size is small, the overhead of launching
 * separate kernels is minimal, and the GPU can still achieve good parallelism by executing
 * these independent sorts concurrently on different Streaming Multiprocessors.
 * -   **Weaknesses**: The performance degrades as the batch size increases because of the
 * host-side loop and the overhead of launching many small kernels. For larger batches,
 * `full_sort` (`SegmentedRadixSort`) is generally much faster as it sorts the entire
 * batch in a single, more efficient kernel launch. Like `full_sort`, it performs
 * much more work than necessary by sorting the entire vocabulary instead of just
 * selecting the top k elements.
 * -   **Use Case**: Primarily serves as a benchmark candidate for small-batch-size scenarios.
 */

// Fills temporary buffers with the scores and their original indices (0 to vocab_size-1).
__global__ void FillInput(const float* input_x_batch, float* temp_v, int* temp_i, int vocab_size) {
  for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < vocab_size; id += blockDim.x * gridDim.x) {
    temp_v[id] = input_x_batch[id];
    temp_i[id] = id;
  }
}

// Calculates the required temporary storage size for CUB's DeviceRadixSort.
inline size_t GetTempStorageBytes(int vocab_size, cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size, 0, sizeof(float) * 8, stream));
  return temp_storage_bytes;
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int /*k*/) {
  constexpr int block_size = 256;
  int blocks_per_batch = CeilDiv(vocab_size, block_size);

  auto* temp_storage = data->cub_temp_storage;
  auto temp_storage_bytes = data->cub_temp_storage_bytes;

  // The final output buffers.
  auto* final_scores_buffer = data->intermediate_scores_1;
  auto* final_indices_buffer = data->intermediate_indices_1;

  // Workspace buffers for the unsorted data.
  auto* workspace_scores = data->intermediate_scores_2;
  auto* workspace_indices = data->intermediate_indices_2;

  // Host-side loop to launch one sort per batch item.
  for (int i = 0; i < batch_size; i++) {
    const float* current_scores_in = scores_in + static_cast<size_t>(i) * vocab_size;

    // Set pointers for the final output location for this batch item.
    // The stride will be vocab_size.
    float* final_scores_out = final_scores_buffer + static_cast<size_t>(i) * vocab_size;
    int* final_indices_out = final_indices_buffer + static_cast<size_t>(i) * vocab_size;

    // Populate workspace buffers with the current batch item's scores and indices.
    FillInput<<<blocks_per_batch, block_size, 0, stream>>>(current_scores_in, workspace_scores, workspace_indices, vocab_size);
    // Launch the CUB radix sort. It sorts from the workspace directly into the final output buffers.
    cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_bytes, workspace_scores, final_scores_out, workspace_indices, final_indices_out, vocab_size, 0, sizeof(float) * 8, stream);
  }

  data->topk_scores = final_scores_buffer;
  data->topk_indices = final_indices_buffer;
  data->topk_stride = vocab_size;
}
}  // namespace per_batch_radix_sort
}  // namespace cuda
}  // namespace Generators
