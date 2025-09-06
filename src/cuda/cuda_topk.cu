// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_select_sort.cuh"

namespace Generators {
namespace cuda {

// Helper to determine the optimal partition size for the hybrid sort algorithm
// based on vocabulary and batch size.
inline int GetHybridSortPartitionSize(int vocab_size, int batch_size) {
  if (vocab_size >= 147456) {
    return (vocab_size > 256 * 1024) ? 8192 : 4096;
  } else {
    if (vocab_size >= 65536 || batch_size >= 4 && vocab_size > 49152) {
      return 2048;
    }
  }

  return 1024;
}

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream) {
  // The intermediate buffers are used by hybrid and full sort algorithms.
  int partition_size = GetHybridSortPartitionSize(vocab_size, batch_size);
  size_t intermediate_buffer_elements = GetHybridSortIntermediateSize(batch_size, vocab_size, partition_size);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  // Selection sort uses a buffer of batch_size * 64 elements, which is smaller than intermediate_buffer_elements.
  size_t max_buffer_elements = std::max(vocab_batch_size, intermediate_buffer_elements);

  // Allocate all necessary device memory
  intermediate_indices_1 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_indices_2 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_scores_1 = CudaMallocArray<float>(max_buffer_elements);
  intermediate_scores_2 = CudaMallocArray<float>(max_buffer_elements);
  batch_offsets = CudaMallocArray<int>(batch_size + 1);

  cub_temp_storage_bytes = GetFullSortCubTempStorageBytes(vocab_batch_size, batch_size, stream);
  cub_temp_storage = CudaMallocArray<unsigned char>(this->cub_temp_storage_bytes);
}

void TopkDataCompact::CompactOutput(int batch_size, int vocab_size, cudaStream_t stream, int k) {
  topk_scores_compact = CudaMallocArray<float>(static_cast<size_t>(batch_size) * k);
  topk_indices_compact = CudaMallocArray<int>(static_cast<size_t>(batch_size) * k);
  dim3 grid(batch_size);
  dim3 block(256);
  CompactStridedData<float><<<grid, block, 0, stream>>>(topk_scores, topk_scores_compact.get(), k, batch_size, topk_stride);
  CompactStridedData<int><<<grid, block, 0, stream>>>(topk_indices, topk_indices_compact.get(), k, batch_size, topk_stride);
  cudaStreamSynchronize(stream);
}

void GetTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(topk_data != nullptr);

  if (k > kHybridSortMaxK) {
    LaunchSort(topk_data, stream, scores_in, topk_data->intermediate_scores_1.get(),
               topk_data->intermediate_indices_1.get(), vocab_size, batch_size);
  } else if (k <= 8 || vocab_size < 1024) {
    // NOTE: This modifies scores_in in-place
    RunTopKViaSelectionSort(topk_data, stream, const_cast<float*>(scores_in), vocab_size, batch_size, k);
  } else {
    int partition_size = GetHybridSortPartitionSize(vocab_size, batch_size);
    RunTopKViaHybridSort(topk_data, stream, scores_in, vocab_size, batch_size, k, partition_size);
  }
}

}  // namespace cuda
}  // namespace Generators
