
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
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include "../src/cuda/cuda_sampling.h"
#include "statistics_helper.h"

namespace {

// A struct to hold the parameters for a benchmark configuration
struct BenchmarkParams {
  int batch_size;
  int vocab_size;
  int k;
  int topk_stride;
};

// A struct to hold the results of a single benchmark run
struct BenchmarkResult {
  BenchmarkParams params;
  std::string algo_name;

  float latency_ms;
  float latency_ms_stdev;
  float latency_ms_95_percentile;
};

void PrintSummary(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n--- Sampling Cuda Kernel Benchmark Summary ---\n";
  std::cout << std::left << std::setw(12) << "Batch Size" << std::setw(8) << "K" << std::setw(12) << "Stride"
            << std::setw(20) << "Algorithm" << std::setw(15) << "Latency(us)" << std::setw(15) << "Stdev(us)"
            << std::setw(15) << "P95(us)" << "\n";
  std::cout << std::string(97, '-') << "\n";

  for (const auto& result : results) {
    std::cout << std::left << std::setw(12) << result.params.batch_size << std::setw(8) << result.params.k
              << std::setw(12) << result.params.topk_stride << std::setw(20) << result.algo_name << std::fixed
              << std::setprecision(2) << std::setw(15) << result.latency_ms * 1000.0f << std::setw(15)
              << result.latency_ms_stdev * 1000.0f << std::setw(15) << result.latency_ms_95_percentile * 1000.0f
              << "\n";
  }
}

void RunBenchmarks(const BenchmarkParams& params) {
  std::cout << "\n--- Running Benchmarks with batch_size=" << params.batch_size << ", vocab_size=" << params.vocab_size
            << ", k=" << params.k << ", stride=" << params.topk_stride << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate dummy input buffers. The content doesn't matter for latency testing.
  const size_t input_buffer_size = static_cast<size_t>(params.batch_size) * params.topk_stride;
  auto d_topk_scores = Generators::CudaMallocArray<float>(input_buffer_size);
  auto d_topk_indices = Generators::CudaMallocArray<int>(input_buffer_size);
  auto d_next_token = Generators::CudaMallocArray<int32_t>(params.batch_size);

  const float p_value = 0.9f;
  const float temperature = 1.0f;
  const unsigned long long random_seed = 1234;

  auto data =
      std::make_unique<Generators::cuda::SamplingData>(random_seed, params.batch_size, params.vocab_size, stream);

  auto bench_algo = [&](auto func) {
    const int warm_up_runs = 5;
    const int total_runs = 1000;
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

  // Benchmark Fused Kernel
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::LaunchFusedSampleKernel(data.get(), stream, d_topk_scores.get(), d_topk_indices.get(),
                                                d_next_token.get(), params.k, params.batch_size, p_value,
                                                temperature, params.topk_stride);
    });
    all_results.push_back({params, "FUSED", mean_ms, stdev_ms, p95_ms});
  }

  // Benchmark Multi-Stage Kernel
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::LaunchMultiStageSampleKernel(data.get(), stream, d_topk_scores.get(), d_topk_indices.get(),
                                                     d_next_token.get(), params.k, params.batch_size, p_value,
                                                     temperature, params.topk_stride);
    });
    all_results.push_back({params, "MULTI_STAGE", mean_ms, stdev_ms, p95_ms});
  }

  PrintSummary(all_results);
  CUDA_CHECK(cudaStreamDestroy(stream));
}
}  // namespace

TEST(CudaSamplingBenchmarks, PerformanceTests) {
  std::vector<int> batch_sizes = {1};
  std::vector<int> vocab_sizes = {204800};
  std::vector<int> ks = {1, 8, 50};

  for (int batch_size : batch_sizes) {
    for (int vocab_size : vocab_sizes) {
      for (int k : ks) {
        // Test different stride scenarios from the Top-K stage
        int stride;
        if (k <= 8) {
          stride = k;  // selection sort
        } else if (k <= Generators::cuda::kHybridSortMaxK) {
          stride = Generators::cuda::kHybridSortMaxK;  // hybrid sort
        } else {
          stride = vocab_size;  // Full sort
        }

        RunBenchmarks({batch_size, vocab_size, k, stride});
      }
    }
  }
}
#endif