// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <array>

#include "cuda_topk.h"
#include "cuda_topk_benchmark_cache.h"

namespace Generators {
namespace cuda {

// Measures the average execution time of a CUDA kernel over several runs.
// This is a lightweight version for online benchmarking, using fewer iterations
// than an offline profiler to minimize runtime overhead.
static float TimeKernel(cudaStream_t stream, std::function<void()> kernel_func) {
  const int warm_up_runs = 2;
  const int total_runs = 5;

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
// is cached for subsequent calls with the same k.
static TopkAlgo BenchmarkAndSelectBestAlgo(TopkData* topk_data,
                                           cudaStream_t stream,
                                           const float* scores_in,
                                           int vocab_size,
                                           int batch_size,
                                           int k) {
  float min_latency = std::numeric_limits<float>::max();
  TopkAlgo best_algo = TopkAlgo::UNKNOWN;

  // Selection sort helps only for small k. This threshold is based on benchmark results.
  constexpr int kSelectionSortBenchmarkMaxK = 8;

  // Radix sort helps only for small batch size. This threshold is based on benchmark results.
  constexpr int kRadixSortBenchmarkMaxBatchSize = 8;

  // Candidate: Selection Sort
  if (k <= kSelectionSortBenchmarkMaxK) {
    float selection_latency = TimeKernel(stream, [&]() {
      selection_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
    min_latency = selection_latency;
    best_algo = TopkAlgo::SELECTION;
  }

  // Candidate: LLM Sort
  bool use_llm_sort = llm_sort::IsSupported(batch_size, vocab_size, k);
  if (use_llm_sort) {
    float llm_latency = TimeKernel(stream, [&]() {
      llm_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
    if (llm_latency < min_latency) {
      min_latency = llm_latency;
      best_algo = TopkAlgo::LLM;
    }
  }

  // Candidate: Flash Sort (Cooperative Kernel)
  bool use_flash_sort = flash_sort::IsSupported(batch_size, vocab_size, k);
  if (use_flash_sort) {
    float flash_latency = TimeKernel(stream, [&]() {
      flash_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
    if (flash_latency < min_latency) {
      min_latency = flash_latency;
      best_algo = TopkAlgo::FLASH;
    }
  }

  if (!use_flash_sort && !use_llm_sort) {
      // Candidate: Hybrid Sort
      if (k <= kHybridSortMaxK) {
      float hybrid_latency = TimeKernel(stream, [&]() {
        hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
      if (hybrid_latency < min_latency) {
        min_latency = hybrid_latency;
        best_algo = TopkAlgo::HYBRID;
      }
    } else { // No fast algorithms for this k, benchmark the fallbacks.
      float full_sort_latency = TimeKernel(stream, [&]() {
        full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
      if (full_sort_latency < min_latency) {
        min_latency = full_sort_latency;
        best_algo = TopkAlgo::FULL;
      }

      if (batch_size <= kRadixSortBenchmarkMaxBatchSize) {
        float radix_sort_latency = TimeKernel(stream, [&]() {
          radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
        });
        if (radix_sort_latency < min_latency) {
          min_latency = radix_sort_latency;
          best_algo = TopkAlgo::RADIX;
        }
      }
    }
  }

  // Cache the result in the shared cache for future calls to avoid re-benchmarking.
  int device_id = topk_data->device_id;
  SetTopkBenchmarkCache(device_id, batch_size, vocab_size, k, best_algo);

  return best_algo;
}

}  // namespace cuda
}  // namespace Generators
