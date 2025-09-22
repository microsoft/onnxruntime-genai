// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/device/device_segmented_radix_sort.cuh>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace full_sort {

/**
 * @brief A robust fallback algorithm that sorts the entire vocabulary for all batch items at once.
 *
 * Algorithm Overview:
 * This algorithm uses CUB's `DeviceSegmentedRadixSort` to perform a single,
 * large sort operation on the entire input tensor, which is treated as a concatenation
 * of all batch items. For example, if `batch_size=4` and `vocab_size=50000`, it sorts
 * a single array of 200,000 elements. The "segmented" nature of the sort ensures that
 * each batch item is sorted independently within its segment of the larger array.
 *
 * Performance Characteristics:
 * -   **Strengths**: It is very efficient for large batch sizes because it processes the entire
 * batch in a single kernel launch, maximizing GPU utilization. CUB's radix sort is
 * highly optimized. It is also simple to implement and serves as a robust fallback.
 * -   **Weaknesses**: This algorithm performs significantly more work than necessary. It sorts
 * all `vocab_size` elements for each batch item, even though only the top `k` are needed.
 * For small `k`, this is very inefficient compared to partition-based methods.
 * -   **Use Case**: Acts as the ultimate fallback algorithm when more specialized methods are
 * not supported or are slower. It's the go-to choice for large batch sizes where `k` is also large
 * (i.e., `k` is a significant fraction of `vocab_size`).
 */

// Kernel to populate an array with the original indices (0 to vocab_size-1) for each batch item.
__global__ void PopulateIndices(int* indices, int vocab_size, int batch_size) {
  int global_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_index < vocab_size * batch_size) {
    indices[global_index] = global_index % vocab_size;
  }
}

inline void LaunchPopulateIndices(int* indices, int vocab_size, int batch_size, cudaStream_t stream) {
  dim3 grid(CeilDiv(batch_size * vocab_size, 256), 1, 1);
  dim3 block(256, 1, 1);
  PopulateIndices<<<grid, block, 0, stream>>>(indices, vocab_size, batch_size);
}

// Kernel to create the segment offsets array required by CUB.
// For a batch of size N, it creates an array [0, V, 2V, ..., NV], where V is vocab_size.
__global__ void PopulateOffsets(int* offsets, int vocab_size, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size + 1) {
    offsets[index] = index * vocab_size;
  }
}

inline void LaunchPopulateOffsets(int* offsets, int vocab_size, int batch_size, cudaStream_t stream) {
  // We need batch_size + 1 offsets for CUB's segmented sort.
  dim3 grid(CeilDiv(batch_size + 1, 256), 1, 1);
  dim3 block(256, 1, 1);
  PopulateOffsets<<<grid, block, 0, stream>>>(offsets, vocab_size, batch_size);
}

inline void LaunchSortPairs(void* d_temp_storage, size_t temp_storage_bytes, const float* d_keys_in, float* d_keys_out,
                            const int* d_values_in, int* d_values_out, int num_items, int num_segments, int* d_offsets,
                            cudaStream_t stream) {
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, num_segments,
      d_offsets, d_offsets + 1, 0, sizeof(float) * 8, stream));
}

// Calculates the required temporary storage size for CUB's DeviceSegmentedRadixSort.
inline size_t GetTempStorageBytes(int num_items, int num_segments, cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_storage_bytes, (float*)nullptr, (float*)nullptr, (int*)nullptr, (int*)nullptr, num_items,
      num_segments, (int*)nullptr, (int*)nullptr, 0, sizeof(float) * 8, stream));
  return temp_storage_bytes;
}

// Orchestrates the setup and launch of the full sort.
inline void LaunchSort(TopkData* data, cudaStream_t stream, const float* scores_in, float* scores_out,
                       int* indices_out, int vocab_size, int batch_size) {
  // Create the segment offsets [0, V, 2V, ...].
  LaunchPopulateOffsets(data->batch_offsets, vocab_size, batch_size, stream);
  // Create the initial indices [0,1,2,...,V-1, 0,1,2,...].
  LaunchPopulateIndices(data->intermediate_indices_2, vocab_size, batch_size, stream);
  // Launch the main sort.
  LaunchSortPairs(data->cub_temp_storage, data->cub_temp_storage_bytes, scores_in, scores_out,
                  data->intermediate_indices_2, indices_out, vocab_size * batch_size, batch_size,
                  data->batch_offsets, stream);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int /*k*/) {
  float* topk_scores = data->intermediate_scores_1;
  int* topk_indices = data->intermediate_indices_1;
  LaunchSort(data, stream, scores_in, topk_scores, topk_indices, vocab_size, batch_size);

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  // The output is strided by vocab_size because the entire sorted vocabulary is returned for each batch item.
  data->topk_stride = vocab_size;
  CUDA_CHECK_LAUNCH();
}
}  // namespace full_sort
}  // namespace cuda
}  // namespace Generators
