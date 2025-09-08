// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_threshold.cuh"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_radix_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_select_sort.cuh"

namespace Generators {
namespace cuda {

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream) {
  hybrid_sort_partition_size = EstimateHybridSortBestPartitionSize(vocab_size);
  selection_sort_k_threshold = EstimateThresholdK(batch_size, vocab_size);

  size_t hybrid_sort_buffer_elements = GetHybridSortIntermediateSize(batch_size, vocab_size, hybrid_sort_partition_size);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;
  size_t max_buffer_elements = std::max(vocab_batch_size, hybrid_sort_buffer_elements);

  // Allocate all necessary device memory
  intermediate_indices_1 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_indices_2 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_scores_1 = CudaMallocArray<float>(max_buffer_elements);
  intermediate_scores_2 = CudaMallocArray<float>(max_buffer_elements);
  batch_offsets = CudaMallocArray<int>(batch_size + 1);

  auto radix_sort_temp_storage_bytes = GetRadixSortCubTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = GetFullSortCubTempStorageBytes(vocab_batch_size, batch_size, stream);
  cub_temp_storage_bytes = std::max(radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);
  cub_temp_storage = CudaMallocArray<unsigned char>(this->cub_temp_storage_bytes);
}

// Kernel to compact strided data into a dense layout.
// Used to convert data from a [batch, stride] layout to a dense [batch, k] layout.
template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int in_idx = batch_idx * input_stride + i;
    int out_idx = batch_idx * k + i;
    output[out_idx] = input[in_idx];
  }
}

void TopkDataCompact::CompactOutput(int batch_size, int vocab_size, cudaStream_t stream, int k) {
  topk_scores_compact = CudaMallocArray<float>(static_cast<size_t>(batch_size) * k);
  topk_indices_compact = CudaMallocArray<int>(static_cast<size_t>(batch_size) * k);
  dim3 grid(batch_size);
  dim3 block(256);
  CompactStridedData<float><<<grid, block, 0, stream>>>(topk_scores, topk_scores_compact.get(), k, batch_size, topk_stride);
  CompactStridedData<int><<<grid, block, 0, stream>>>(topk_indices, topk_indices_compact.get(), k, batch_size, topk_stride);
}

void GetTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(topk_data != nullptr);

  if (k <= topk_data->selection_sort_k_threshold) {
    RunTopKViaSelectionSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
    return;
  }

  if (k <= kHybridSortMaxK) {
    RunTopKViaHybridSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
    return;
  }

  if (batch_size <= 2) {
    RunTopKViaRadixSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  } else {
    RunTopKViaFullSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  }
}

}  // namespace cuda
}  // namespace Generators
