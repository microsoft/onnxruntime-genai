// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>           // For FLT_MAX
#include <math_constants.h>  // For CUDART_INF_F

namespace Generators {
namespace cuda {
namespace bitonic_sort {

// Full bitonic sort implementation for cases where kBlockSize < SortSize.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Large(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize > 0 && (kBlockSize & (kBlockSize - 1)) == 0,
                "kBlockSize must be a power of 2");
  static_assert(SortSize >= kBlockSize);

  const int tid = threadIdx.x;
  constexpr int elements_per_thread = SortSize / kBlockSize;

  auto compareAndSwap = [&](int i, int j, bool ascending) {
    bool is_i_greater = (smem_scores[i] > smem_scores[j]) ||
                        (smem_scores[i] == smem_scores[j] && smem_indices[i] < smem_indices[j]);

    // For a bitonic network, we swap if the element order is contrary to the
    // required direction of the current sub-sequence.
    // Ascending region: swap if first element is greater.
    // Descending region: swap if first element is smaller.
    // This simplifies to: swap if (is_i_greater != ascending).
    if (is_i_greater != ascending) {
      float temp_score = smem_scores[i];
      smem_scores[i] = smem_scores[j];
      smem_scores[j] = temp_score;

      int temp_index = smem_indices[i];
      smem_indices[i] = smem_indices[j];
      smem_indices[j] = temp_index;
    }
  };

  // Phase 1: Sort local elements within each thread (descending order)
  if constexpr (elements_per_thread > 1) {
    int base_idx = tid * elements_per_thread;
    for (int i = base_idx; i < base_idx + elements_per_thread; ++i) {
      for (int j = i + 1; j < base_idx + elements_per_thread; ++j) {
        compareAndSwap(i, j, false);  // simple bubble sort for local elements
      }
    }
    __syncthreads();
  }

  // Phase 2: Bitonic merge phases
  for (int size = 2; size <= SortSize; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      if (elements_per_thread > 1) {
        for (int t = 0; t < elements_per_thread; t++) {
          int idx = tid * elements_per_thread + t;
          int partner = idx ^ stride;

          if (partner > idx) {
            // A standard bitonic network sorts ascending with `(idx & size) == 0`.
            bool ascending = ((idx & size) == 0);
            compareAndSwap(idx, partner, ascending);
          }
        }
      } else {
        int partner = tid ^ stride;
        if (partner > tid) {
          // A standard bitonic network sorts ascending with `(idx & size) == 0`.
          bool ascending = ((tid & size) == 0);
          compareAndSwap(tid, partner, ascending);
        }
      }
      __syncthreads();
    }
  }

  // Phase 3: Final merge to create fully sorted sequence in descending order
  // This phase is now redundant because the main loop produces a descending sort.
  // However, running a descending merge on a descending-sorted list is harmless (a no-op).
  // It is left here for clarity and to minimize structural code changes.
  for (int stride = SortSize >> 1; stride > 0; stride >>= 1) {
    if constexpr (elements_per_thread > 1) {
      for (int t = 0; t < elements_per_thread; ++t) {
        int idx = tid * elements_per_thread + t;
        int partner = idx ^ stride;
        if (partner > idx) {
          compareAndSwap(idx, partner, false);
        }
      }
    } else {  // one element per thread
      int partner = tid ^ stride;
      if (partner > tid) {
        compareAndSwap(tid, partner, false);  // Always descending
      }
    }
    __syncthreads();
  }
}

/**
 * @brief Performs an in-place bitonic sort on data in shared memory.
 * This version is for kBlockSize >= SortSize.
 */
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Small(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0,
                "SortSize must be a power of 2");
  static_assert(kBlockSize >= SortSize);

  // This implementation uses one thread per element for the sort.
  const int ix = threadIdx.x;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= SortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      if (ix < SortSize) {
        int paired_ix = ix ^ j;
        if (paired_ix > ix) {
          // A standard bitonic network sorts ascending with `(ix & k) == 0`.
          // The swap condition is inverted to produce a descending sort.
          bool ascending = ((ix & k) == 0);

          bool is_ix_greater = (smem_scores[ix] > smem_scores[paired_ix]) ||
                               (smem_scores[ix] == smem_scores[paired_ix] && smem_indices[ix] < smem_indices[paired_ix]);

          if (is_ix_greater != ascending) {
            float temp_score = smem_scores[ix];
            smem_scores[ix] = smem_scores[paired_ix];
            smem_scores[paired_ix] = temp_score;

            int temp_index = smem_indices[ix];
            smem_indices[ix] = smem_indices[paired_ix];
            smem_indices[paired_ix] = temp_index;
          }
        }
      }
    }
  }
  __syncthreads();
}

// Generic implementation for bitonic sort in shared memory.
// Operating on separate score and index arrays (SoA).
// - SortSize must be a power of two.
// - All threads in the block must call this function.
// - Result: sorted in-place by score descending, tie-breaker index ascending.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Big(float* smem_scores, int* smem_indices) {
  static_assert(SortSize > 0 && (SortSize & (SortSize - 1)) == 0, "SortSize must be power of two");

  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  __syncthreads();

  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < N; i += kBlockSize) {
        const int ixj = i ^ j;
        if (ixj > i) {
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          // A standard bitonic network sorts ascending with `(i & k) == 0`.
          // The swap condition is inverted to produce a descending sort.
          bool ascending = ((i & k) == 0);
          bool is_i_greater = (a_i > a_j) || (a_i == a_j && idx_i < idx_j);

          if (is_i_greater != ascending) {
            smem_scores[i] = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i] = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
}

constexpr int NextPowerOfTwo(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Generic implementation for bitonic sort in shared memory (SoA).
// Supports arbitrary SortSize by padding to next power of two.
template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort_Pad(float* smem_scores, int* smem_indices) {
  const int tid = threadIdx.x;
  constexpr int N = SortSize;

  constexpr int Npad = NextPowerOfTwo(N);

  for (int i = tid; i < Npad; i += kBlockSize) {
    if (i >= N) {
      smem_scores[i] = -CUDART_INF_F;
      smem_indices[i] = INT_MAX;
    }
  }
  __syncthreads();

  for (int k = 2; k <= Npad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < Npad; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          float a_i = smem_scores[i];
          float a_j = smem_scores[ixj];
          int idx_i = smem_indices[i];
          int idx_j = smem_indices[ixj];

          // A standard bitonic network sorts ascending with `(i & k) == 0`.
          // The swap condition is inverted to produce a descending sort.
          bool ascending = ((i & k) == 0);
          bool is_i_greater = (a_i > a_j) || (a_i == a_j && idx_i < idx_j);

          if (is_i_greater != ascending) {
            smem_scores[i] = a_j;
            smem_scores[ixj] = a_i;
            smem_indices[i] = idx_j;
            smem_indices[ixj] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
}

template <int kBlockSize, int SortSize>
__device__ void SharedMemBitonicSort(float* smem_scores, int* smem_indices) {
  if constexpr ((SortSize & (SortSize - 1)) != 0) {
    // Non-power-of-two sizes must be padded and fully sorted.
    SharedMemBitonicSort_Pad<kBlockSize, SortSize>(smem_scores, smem_indices);
  } else {
    if constexpr (kBlockSize >= SortSize) {
      // With enough threads, a full sort is very fast.
      SharedMemBitonicSort_Small<kBlockSize, SortSize>(smem_scores, smem_indices);
    } else {
      // Fewer threads than elements. Use the generic full sort for power-of-two sizes.
      SharedMemBitonicSort_Big<kBlockSize, SortSize>(smem_scores, smem_indices);
      // SharedMemBitonicSort_Large<kBlockSize, SortSize>(smem_scores, smem_indices);
    }
  }
}

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
  __shared__ float smem_scores[SortSize];
  __shared__ int smem_indices[SortSize];

  const int batch_idx = blockIdx.y;
  const int block_start_partition = blockIdx.x * PartitionsPerBlock;
  const int num_partitions_to_process = min(PartitionsPerBlock, num_partitions_in - block_start_partition);

  const size_t in_base_offset = static_cast<size_t>(batch_idx) * num_partitions_in * K;
  const size_t out_base_offset = (static_cast<size_t>(batch_idx) * gridDim.x + blockIdx.x) * K;

  // Load data from global memory into shared memory using an SoA layout
  for (int i = threadIdx.x; i < SortSize; i += kBlockSize) {
    if (i < K * num_partitions_to_process) {
      int partition_idx = i / K;
      int element_idx = i % K;
      size_t global_offset = in_base_offset + static_cast<size_t>(block_start_partition + partition_idx) * K + element_idx;
      smem_scores[i] = scores_in[global_offset];
      smem_indices[i] = indices_in[global_offset];
    } else {
      smem_scores[i] = -FLT_MAX;
      smem_indices[i] = INT_MAX;
    }
  }
  __syncthreads();

  // Perform the sort on the SoA data in shared memory.
  SharedMemBitonicSort<kBlockSize, SortSize>(smem_scores, smem_indices);

  // Write the top K results back to global memory
  if (threadIdx.x < K) {
    indices_out[out_base_offset + threadIdx.x] = smem_indices[threadIdx.x];
    scores_out[out_base_offset + threadIdx.x] = smem_scores[threadIdx.x];
  }
}

}  // namespace bitonic_sort
}  // namespace cuda
}  // namespace Generators
