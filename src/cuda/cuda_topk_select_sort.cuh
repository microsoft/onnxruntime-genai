// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>  // For FLT_MAX
#include <cub/cub.cuh>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace select_sort {

/**
 * @brief A direct implementation of the selection sort algorithm on the GPU.
 *
 * Algorithm Overview:
 * This algorithm mimics the classic selection sort. To find the Top-K elements, it iterates `k` times.
 * In each iteration, all threads in a block cooperatively scan the entire `vocab_size` to find the
 * single element with the highest score. This element is written to the output. To prevent it
 * from being selected again in the next iteration, its score in the input tensor is overwritten
 * with a very small number (`-FLT_MAX`).
 *
 * Performance Characteristics:
 * -   **Strengths**: Simple to understand and implement. It has a highly optimized path for the
 * common `k=1` case (`GetTop1Kernel`), which is read-only and avoids the overhead of modifying the input tensor.
 * -   **Weaknesses**: The algorithm has a complexity of roughly O(k * vocab_size), making it
 * extremely inefficient for anything other than very small `k` values (e.g., k <= 4).
 * For `k > 1`, it must modify the input tensor, requiring a "copy-on-write" strategy
 * to avoid altering the original user-provided scores.
 * -   **Use Case**: Serves as a performance baseline and is only competitive for `k=1` or other tiny `k` values.
 */

// A simple struct to hold a key-value pair for CUB reduction.
struct TopK_Pair {
  int p = INT_MAX;     // Index
  float u = -FLT_MAX;  // Score

  // Inserts an element if it's greater than the current max, with tie-breaking for stability.
  __device__ __forceinline__ void Insert(float elem, int elem_id) {
    if (elem > u || (elem == u && elem_id < p)) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void Init() {
    u = -FLT_MAX;
    p = INT_MAX;
  }
};

// CUB reduction operator for finding the maximum score and its corresponding index.
__device__ __forceinline__ TopK_Pair reduce_topk_op(TopK_Pair const& a, TopK_Pair const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p < b.p) ? a
                                                   : b;
}

/**
 * @brief Specialized kernel optimized for finding only the Top-1 element.
 * It is read-only on `scores_in` and avoids the overhead of the iterative version.
 */
template <int kBlockSize>
__global__ void GetTop1Kernel(const float* scores_in, float* scores_out, int* indices_out, int batch_size, int vocab_size) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_Pair partial;

  // Each thread block processes one batch item.
  // Threads cooperatively scan the vocabulary to find the max element.
  for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
    float elem = scores_in[elemId + batch * vocab_size];
    partial.Insert(elem, elemId);
  }
  // Reduce within the thread block to find the block's top element.
  typedef cub::BlockReduce<TopK_Pair, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  TopK_Pair top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op);

  // Thread 0 writes the final result. No fence or write-back to scores_in is needed.
  if (tid == 0) {
    scores_out[batch] = top_k_sequence.u;
    indices_out[batch] = top_k_sequence.p;
  }
}

// General kernel to find the top K elements using iterative selection sort.
// This version modifies its input `scores_in` in-place for maximum performance.
template <int kBlockSize>
__global__ void GetTopKKernel(volatile float* scores_in, float* scores_out, int* indices_out, int batch_size, int vocab_size,
                              int k) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_Pair partial;

  for (int ite = 0; ite < k; ite++) {
    partial.Init();
    for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
      float elem = scores_in[elemId + batch * vocab_size];
      partial.Insert(elem, elemId);
    }
    typedef cub::BlockReduce<TopK_Pair, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_Pair top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op);

    if (tid == 0) {
      scores_out[ite + batch * k] = top_k_sequence.u;
      indices_out[ite + batch * k] = top_k_sequence.p;
      scores_in[batch * vocab_size + top_k_sequence.p] = -FLT_MAX;

      __threadfence_block();
    }
    __syncthreads();
  }
}

void LaunchGetTop1(cudaStream_t stream, const float* scores_in, float* scores_out, int* indices_out, int vocab_size,
                   int batch_size) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(1024, 1, 1);
  GetTop1Kernel<1024><<<grid, block, 0, stream>>>(scores_in, scores_out, indices_out, batch_size, vocab_size);
}

void LaunchGetTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size,
                   int batch_size, int k) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(1024, 1, 1);
  GetTopKKernel<1024><<<grid, block, 0, stream>>>(scores_in, scores_out, indices_out, batch_size, vocab_size, k);
}

void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  float* topk_scores = data->intermediate_scores_1;
  int* topk_indices = data->intermediate_indices_1;

  if (k == 1) {
    // For Top-1, use the specialized read-only kernel. No copy is needed.
    LaunchGetTop1(stream, scores_in, topk_scores, topk_indices, vocab_size, batch_size);
  } else {
    // For k > 1, use the "copy-on-write" strategy. We copy the input scores to a
    // mutable buffer because the general kernel modifies the scores in-place.
    float* mutable_scores = data->intermediate_scores_2;
    size_t buffer_size = static_cast<size_t>(batch_size) * vocab_size * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mutable_scores, scores_in, buffer_size, cudaMemcpyDeviceToDevice, stream));
    LaunchGetTopK(stream, mutable_scores, topk_scores, topk_indices, vocab_size, batch_size, k);
  }
  CUDA_CHECK_LAUNCH();

  data->topk_scores = topk_scores;
  data->topk_indices = topk_indices;
  data->topk_stride = k;
}

}  // namespace select_sort
}  // namespace cuda
}  // namespace Generators
