// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include "cuda_topk_distributed_select_sort.cuh"

namespace Generators {
namespace cuda {

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream) {
  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  // Allocate all necessary device memory
  // The buffers are sized for the full sort algorithm, which is the largest.
  intermediate_indices_1 = CudaMallocArray<int>(vocab_batch_size);
  intermediate_indices_2 = CudaMallocArray<int>(vocab_batch_size);
  intermediate_scores_1 = CudaMallocArray<float>(vocab_batch_size);
  intermediate_scores_2 = CudaMallocArray<float>(vocab_batch_size);
  batch_offsets = CudaMallocArray<int>(batch_size + 1);

  cub_temp_storage_bytes = GetFullSortCubTempStorageBytes(vocab_batch_size, batch_size, stream);
  cub_temp_storage = CudaMallocArray<unsigned char>(this->cub_temp_storage_bytes);

  // Distributed Selection sort metadata/buffers
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);  // Get properties for device 0
  top_k_distributed_select_sort_shards = std::min(topk_impl_details::kTopKDistributedSelectSortMaxShards, deviceProp.multiProcessorCount);

  top_k_distributed_select_sort_lock = CudaMallocArray<int>(1);
  cudaMemset(top_k_distributed_select_sort_lock.get(), 0, sizeof(int));

  top_k_distributed_select_sort_keys = CudaMallocArray<int>(top_k_distributed_select_sort_shards * topk_impl_details::kTopKDistributedSelectSortMaxTopK);
  top_k_distributed_select_sort_values = CudaMallocArray<float>(top_k_distributed_select_sort_shards * topk_impl_details::kTopKDistributedSelectSortMaxTopK);
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

  bool enable_distributed_selection_sort = ((batch_size <= topk_impl_details::kTopKDistributedSelectSortMaxBatchSize) &&
                                            (k <= topk_impl_details::kTopKDistributedSelectSortMaxTopK) &&
                                            (vocab_size >= topk_impl_details::kTopKDistributedSelectSortMinVocabSize) &&
                                            (topk_data->top_k_distributed_select_sort_shards > 0));

  if (enable_distributed_selection_sort) {
    RunTopKViaDistributedSelectionSort(topk_data, stream, scores_in, vocab_size, k);
  } else if (k > 64) {
    RunTopKViaFullSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  } else {
    RunTopKViaSelectionSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  }
}

}  // namespace cuda
}  // namespace Generators
