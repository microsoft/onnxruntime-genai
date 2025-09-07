// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if USE_CUDA
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../src/cuda/cuda_topk.h"
#include "statistics_helper.h"

namespace {

// A struct to hold the parameters for a benchmark configuration
struct BenchmarkParams {
  int batch_size;
  int vocab_size;
  int k;
};

// A struct to hold the results of a single benchmark run
struct BenchmarkResult {
  BenchmarkParams params;
  std::string algo_name;
  int partition_size;

  float latency_ms;
  float latency_ms_stdev;
  float latency_ms_95_percentile;
};

void PrintSummary(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n--- TopK Cuda Kernel Benchmark Summary ---\n";
  std::cout << std::left << std::setw(12) << "Batch Size" << std::setw(12) << "Vocab Size" << std::setw(5) << "K"
            << std::setw(28) << "Algorithm" << std::setw(12) << "Latency(us)" << std::setw(12) << "Stdev(us)"
            << std::setw(12) << "P95(us)" << "\n";
  std::cout << std::string(97, '-') << "\n";

  for (const auto& result : results) {
    std::string full_algo_name = result.algo_name;
    if (result.partition_size > 0) {
      full_algo_name += " (p=" + std::to_string(result.partition_size) + ")";
    }

    std::cout << std::left << std::setw(12) << result.params.batch_size << std::setw(12) << result.params.vocab_size
              << std::setw(5) << result.params.k << std::setw(28) << full_algo_name << std::fixed
              << std::setprecision(2) << std::setw(12) << result.latency_ms * 1000.0f << std::setw(12)
              << result.latency_ms_stdev * 1000.0f << std::setw(12) << result.latency_ms_95_percentile * 1000.0f
              << "\n";
  }
}

void RunBenchmarks(const BenchmarkParams& params) {
  std::cout << "\n--- Running Benchmarks with batch_size=" << params.batch_size << ", vocab_size=" << params.vocab_size
            << ", k=" << params.k << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto scores_in_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.vocab_size);

  auto bench_algo = [&](auto func) {
    const int warm_up_runs = 5;
    const int total_runs = 20;
    std::vector<double> latencies;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warm_up_runs + total_runs; ++i) {
      CUDA_CHECK(cudaEventRecord(start, stream));
      func();
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      if (i >= warm_up_runs) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        latencies.push_back(ms);
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return std::make_tuple(static_cast<float>(mean(latencies)), static_cast<float>(stdev(latencies)),
                           static_cast<float>(percentile(latencies, 95.0)));
  };

  std::vector<BenchmarkResult> all_results;
  auto data = std::make_unique<Generators::cuda::TopkData>(params.batch_size, params.vocab_size, stream);
  // Benchmark Full Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::RunTopKViaFullSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                           params.batch_size, params.k);
    });
    all_results.push_back({params, "FULL_SORT", 0, mean_ms, stdev_ms, p95_ms});
  }

  // Benchmark Selection Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::RunTopKViaSelectionSort(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                                params.batch_size, params.k);
    });
    all_results.push_back({params, "SELECTION_SORT", 0, mean_ms, stdev_ms, p95_ms});
  }

  PrintSummary(all_results);
  CUDA_CHECK(cudaStreamDestroy(stream));
}
}  // namespace

TEST(TopKBenchmarks, PerformanceTests) {
  // Test K
  {
    std::vector<int> batch_sizes = {1};
    std::vector<int> vocab_sizes = {201088};
    std::vector<int> ks = {50, 1, 2, 4, 8, 10, 16, 32, 64};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }
    for (const auto& params : test_cases) {
      RunBenchmarks(params);
    }
  }

  constexpr bool is_build_pipeline = false;  // Change it false to trigger more runs in local machine.

  // Test batch sizes.
  if constexpr (!is_build_pipeline) {
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    std::vector<int> vocab_sizes = {201088};
    std::vector<int> ks = {50};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }

    for (const auto& params : test_cases) {
      RunBenchmarks(params);
    }
  }

  // Test vocab_sizes.
  if constexpr (!is_build_pipeline) {
    std::vector<int> batch_sizes = {1};
    std::vector<int> vocab_sizes = {512, 1 * 1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024};
    std::vector<int> ks = {8, 16, 50};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }

    for (const auto& params : test_cases) {
      RunBenchmarks(params);
    }
  }
}

#endif
