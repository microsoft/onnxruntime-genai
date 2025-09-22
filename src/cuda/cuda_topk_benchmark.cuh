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

/**
 * @brief Measures the average execution time of a CUDA kernel over several runs.
 * This is a lightweight version for online benchmarking, using fewer iterations
 * than an offline profiler to minimize runtime overhead.
 * @param stream The CUDA stream to run the kernel on.
 * @param kernel_func A lambda function that launches the kernel.
 * @return The average execution time in milliseconds.
 */
static float TimeKernel(cudaStream_t stream, std::function<void()> kernel_func) {
  const int warm_up_runs = 2;
  const int total_runs = 5;

  cuda_event_holder start_event, stop_event;

  // Warm-up runs to handle any one-time kernel loading costs or JIT compilation.
  for (int i = 0; i < warm_up_runs; ++i) {
    kernel_func();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timed runs.
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

// Helper to convert TopkAlgo enum to its string representation for printing.
static const char* TopkAlgoToString(TopkAlgo algo) {
  switch (algo) {
    case TopkAlgo::SELECTION:
      return select_sort::kAlgorithmName;
    case TopkAlgo::HYBRID:
      return hybrid_sort::kAlgorithmName;
    case TopkAlgo::ITERATIVE:
      return iterative_sort::kAlgorithmName;
    case TopkAlgo::CASCADED:
      return cascaded_sort::kAlgorithmName;
    case TopkAlgo::CONVERGENT:
      return flash_convergent::kAlgorithmName;
    case TopkAlgo::DISTRIBUTED_SELECT:
      return distributed_select_sort::kAlgorithmName;
    case TopkAlgo::PER_BATCH_RADIX:
      return per_batch_radix_sort::kAlgorithmName;
    case TopkAlgo::FULL:
      return full_sort::kAlgorithmName;
    default:
      return "Unknown";
  }
}

// Helper macro to benchmark a kernel, update the best algorithm, and handle potential CUDA errors.
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

/**
 * @brief Performs online benchmarking to select the best Top-K algorithm for the current problem size and hardware.
 *
 * It times several candidate algorithms and picks the fastest one. The selection logic is heuristic-based,
 * prioritizing specialized, high-performance kernels (`cascaded_sort`, `iterative_sort`, `flash_convergent`) when supported.
 * If these are not applicable, or for certain problem sizes, it considers more general-purpose algorithms
 * like `hybrid_sort`. If no specialized algorithm is faster or supported, it falls back to robust baseline
 * implementations like `full_sort`. The result is cached for subsequent calls.
 *
 * @return The `TopkAlgo` enum corresponding to the fastest measured algorithm.
 */
static TopkAlgo BenchmarkAndSelectBestAlgo(TopkData* topk_data,
                                           cudaStream_t stream,
                                           const float* scores_in,
                                           int vocab_size,
                                           int batch_size,
                                           int k) {
  float min_latency = std::numeric_limits<float>::max();
  TopkAlgo best_algo = TopkAlgo::UNKNOWN;

  // Heuristic: Selection sort is only competitive for extremely small k.
  constexpr int kSelectionSortBenchmarkMaxK = 4;

  // Heuristic: Radix sort is typically best for small batch sizes.
  constexpr int kRadixSortBenchmarkMaxBatchSize = 8;

  // Candidate: Selection Sort (only for very small k).
  if (k <= kSelectionSortBenchmarkMaxK) {
    BENCHMARK_KERNEL(TopkAlgo::SELECTION, [&]() {
      select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });

    if (distributed_select_sort::IsSupported(batch_size, vocab_size, k)) {
      BENCHMARK_KERNEL(TopkAlgo::DISTRIBUTED_SELECT, [&]() {
        distributed_select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // Candidate: CASCADED Sort (high-performance cooperative kernel).
  bool use_cascaded_sort = cascaded_sort::IsSupported(batch_size, vocab_size, k);
  if (use_cascaded_sort) {
    BENCHMARK_KERNEL(TopkAlgo::CASCADED, [&]() {
      cascaded_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }

  // Candidate: Iterative Sort (high-performance cooperative kernel).
  bool use_iterative_sort = iterative_sort::IsSupported(batch_size, vocab_size, k);
  if (use_iterative_sort) {
    BENCHMARK_KERNEL(TopkAlgo::ITERATIVE, [&]() {
      iterative_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }

  // Candidate: Flash Convergent Sort (high-performance cooperative kernel).
  bool use_flash_convergent = flash_convergent::IsSupported(batch_size, vocab_size, k);
  if (use_flash_convergent) {
    BENCHMARK_KERNEL(TopkAlgo::CONVERGENT, [&]() {
      flash_convergent::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });
  }

  // Candidate: Hybrid Sort. This is a robust fallback. We benchmark it if either the cooperative
  // kernels are not supported, or if the vocab size is small, where hybrid can sometimes be faster.
  if (!use_iterative_sort && !use_cascaded_sort && !use_flash_convergent || vocab_size <= 4096) {
    if (k <= kHybridSortMaxK) {
      BENCHMARK_KERNEL(TopkAlgo::HYBRID, [&]() {
        hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // If no fast algorithm has been selected yet, it means either they were not supported or not faster.
  // We now benchmark the baseline fallbacks.
  if (best_algo == TopkAlgo::UNKNOWN) {
    BENCHMARK_KERNEL(TopkAlgo::FULL, [&]() {
      full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    });

    if (batch_size <= kRadixSortBenchmarkMaxBatchSize) {
      BENCHMARK_KERNEL(TopkAlgo::PER_BATCH_RADIX, [&]() {
        per_batch_radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      });
    }
  }

  // Cache the result in the shared cache for future runs to avoid re-benchmarking.
  int device_id = topk_data->device_id;
  SetTopkBenchmarkCache(device_id, batch_size, vocab_size, k, best_algo);

  return best_algo;
}

}  // namespace cuda
}  // namespace Generators
