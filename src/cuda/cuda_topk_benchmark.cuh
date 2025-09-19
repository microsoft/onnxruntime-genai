// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <array>
#include <iostream>

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

// Added helper to convert TopkAlgo to its string representation for printing.
static const char* TopkAlgoToString(TopkAlgo algo) {
  switch (algo) {
    case TopkAlgo::SELECTION:
      return "Selection Sort";
    case TopkAlgo::HYBRID:
      return "Hybrid Sort";
    case TopkAlgo::FLASH:
      return "Flash Sort";
    case TopkAlgo::LLM:
      return "LLM Sort";
    case TopkAlgo::PARTITION:
      return "Partition Sort";
    case TopkAlgo::RADIX:
      return "Radix Sort";
    case TopkAlgo::FULL:
      return "Full Sort";
    default:
      return "Unknown";
  }
}

// Helper macro to benchmark a kernel, update the best algorithm, and handle exceptions.
#define BENCHMARK_KERNEL(algo_enum, kernel_lambda)                                      \
  try {                                                                                 \
    float latency = TimeKernel(stream, kernel_lambda);                                  \
    if (latency < min_latency) {                                                        \
      min_latency = latency;                                                            \
      best_algo = algo_enum;                                                            \
    }                                                                                   \
  } catch (const Generators::CudaError& e) {                                            \
    std::cerr << "Benchmarking failed for " << TopkAlgoToString(algo_enum)              \
              << " kernel with k=" << k << ", batch_size=" << batch_size                \
              << ", vocab_size=" << vocab_size << ". Error: " << e.what() << std::endl; \
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
  constexpr int kSelectionSortBenchmarkMaxK = 4;

  // Radix sort helps only for small batch size. This threshold is based on benchmark results.
  constexpr int kRadixSortBenchmarkMaxBatchSize = 8;

  // Candidate: Selection Sort is enabled only for very small k.
  if (k <= kSelectionSortBenchmarkMaxK) {
    BENCHMARK_KERNEL(TopkAlgo::SELECTION, [&]() {
      selection_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });

    if (distributed_select_sort::IsSupported(batch_size, vocab_size, k)) {
      BENCHMARK_KERNEL(TopkAlgo::DISTRIBUTED, [&]() {
        distributed_select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // Candidate: LLM Sort
  bool use_llm_sort = llm_sort::IsSupported(batch_size, vocab_size, k);
  if (use_llm_sort) {
    BENCHMARK_KERNEL(TopkAlgo::LLM, [&]() {
      llm_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }

  // Candidate: Flash Sort
  bool use_flash_sort = flash_sort::IsSupported(batch_size, vocab_size, k);
  if (use_flash_sort) {
    BENCHMARK_KERNEL(TopkAlgo::FLASH, [&]() {
      flash_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }
  
  // Candidate: Hybrid Sort. Only enabled when neither Flash Sort nor LLM Sort is used, or when vocab_size is small.
  if (!use_flash_sort && !use_llm_sort || vocab_size <= 4096) {
    if (k <= kHybridSortMaxK) {
      BENCHMARK_KERNEL(TopkAlgo::HYBRID, [&]() {
        hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // Candidate: Partition Sort. Only enabled when neither Flash Sort nor LLM Sort is used.
  if (!use_flash_sort && !use_llm_sort && radix_partition_sort::IsSupported(batch_size, vocab_size, k)) {
    BENCHMARK_KERNEL(TopkAlgo::PARTITION, [&]() {
      radix_partition_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }

  // No fast algorithm found, fallback to Full Sort and Radix Sort.
  if (best_algo == TopkAlgo::UNKNOWN) {
    BENCHMARK_KERNEL(TopkAlgo::FULL, [&]() {
      full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });

    if (batch_size <= kRadixSortBenchmarkMaxBatchSize) {
      BENCHMARK_KERNEL(TopkAlgo::RADIX, [&]() {
        radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // Cache the result in the shared cache for future calls to avoid re-benchmarking.
  int device_id = topk_data->device_id;
  SetTopkBenchmarkCache(device_id, batch_size, vocab_size, k, best_algo);

  return best_algo;
}

}  // namespace cuda
}  // namespace Generators
