// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <limits>

#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// Measures the average execution time of a CUDA kernel over several runs.
// This is a lightweight version for online benchmarking, using fewer iterations
// than an offline profiler to minimize runtime overhead.
static float TimeKernel(cudaStream_t stream, std::function<void()> kernel_func) {
  const int warm_up_runs = 2;
  const int total_runs = 15;

  cuda_event_holder start_event, stop_event;

  // Warm-up runs to handle any one-time kernel loading costs.
  for (int i = 0; i < warm_up_runs; ++i) {
    kernel_func();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timed runs
  CUDA_CHECK(cudaEventRecord(start_event, stream));
  for (int i = 0; i < total_runs; ++i) {
    kernel_func();
  }
  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

  return ms / total_runs;
}

// Performs online benchmarking for small k to select the best Top-K algorithm.
// It times several candidate algorithms and picks the fastest one. The result
// is cached in TopkData for subsequent calls with the same k.
static TopkAlgo BenchmarkAndSelectBestAlgo(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  float min_latency = std::numeric_limits<float>::max();
  TopkAlgo best_algo = TopkAlgo::UNKNOWN;

  // Candidate 1: Selection Sort
  float selection_latency = TimeKernel(stream, [&]() {
    RunTopKViaSelectionSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  });
  min_latency = selection_latency;
  best_algo = TopkAlgo::SELECTION;

  // Candidate 2: Distributed Sort (if k is within its supported range)
  if (batch_size == 1 && k <= kDistributedSortMaxK) {
    float distributed_latency = TimeKernel(stream, [&]() {
      RunTopKViaDistributedSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
    if (distributed_latency < min_latency) {
      min_latency = distributed_latency;
      best_algo = TopkAlgo::DISTRIBUTED;
    }
  }

  // Candidate 3: Hybrid Sort
  if (k <= kHybridSortMaxK) {
    float hybrid_latency = TimeKernel(stream, [&]() {
      RunTopKViaHybridSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
    if (hybrid_latency < min_latency) {
      min_latency = hybrid_latency;
      best_algo = TopkAlgo::HYBRID;
    }
  }

  // Cache the result for future calls to avoid re-benchmarking.
  topk_data->best_algo_cache[k] = best_algo;
  return best_algo;
}

}  // namespace cuda
}  // namespace Generators
