// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if USE_CUDA
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
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

#include "../../src/cuda/cuda_topk.h"
#include "../statistics_helper.h"

#define TEST_NEW_ALGO_ONLY 0

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

  float latency_ms;
  float latency_ms_stdev;
  float latency_ms_95_percentile;
};

// A struct to hold the aggregated results for the final CSV summary.
struct CsvSummaryResult {
  BenchmarkParams params;

  // Latency for each algorithm. A negative value indicates it was not run.
#if TEST_NEW_ALGO_ONLY == 0
  float default_latency = -1.0f;
  float full_sort_latency = -1.0f;
  float radix_sort_latency = -1.0f;
  float selection_sort_latency = -1.0f;
#endif
  float hybrid_sort_latency = -1.0f;
  float flash_sort_latency = -1.0f;
  float llm_sort_latency = -1.0f;

  std::string best_algorithm = "NA";
  float best_latency = std::numeric_limits<float>::max();
};

void PrintSummary(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n--- TopK Cuda Kernel Benchmark Summary ---\n";
  std::cout << std::left << std::setw(12) << "Batch Size" << std::setw(12) << "Vocab Size" << std::setw(5) << "K"
            << std::setw(28) << "Algorithm" << std::setw(12) << "Latency(us)" << std::setw(12) << "Stdev(us)"
            << std::setw(12) << "P95(us)" << "\n";
  std::cout << std::string(97, '-') << "\n";

  for (const auto& result : results) {
    std::string full_algo_name = result.algo_name;

    std::cout << std::left << std::setw(12) << result.params.batch_size << std::setw(12) << result.params.vocab_size
              << std::setw(5) << result.params.k << std::setw(28) << full_algo_name << std::fixed
              << std::setprecision(2) << std::setw(12) << result.latency_ms * 1000.0f << std::setw(12)
              << result.latency_ms_stdev * 1000.0f << std::setw(12) << result.latency_ms_95_percentile * 1000.0f
              << "\n";
  }
}

void PrintCsvSummary(const std::vector<CsvSummaryResult>& results) {
  if (results.empty()) {
    return;
  }

  const char* filename = "topk_benchmark_summary.csv";
  std::ofstream summary_file(filename);
  if (!summary_file.is_open()) {
    std::cerr << "Error: Could not open summary file '" << filename << "' for writing." << std::endl;
    return;
  }
  std::cout << "\n--- Writing TopK Benchmark CSV Summary to " << filename << " ---\n";

  // Write header
#if TEST_NEW_ALGO_ONLY == 0
  summary_file << "batch_size,vocab_size,k,full_sort,radix_sort,selection_sort,hybrid_sort,flash_sort,llm_sort,best_algorithm,best_latency,default\n";
#else
  summary_file << "batch_size,vocab_size,k,hybrid_sort,flash_sort,llm_sort,best_algorithm,best_latency\n";
#endif

  for (const auto& result : results) {
    summary_file << result.params.batch_size << ","
                 << result.params.vocab_size << ","
                 << result.params.k << ",";

    // Helper lambda to print latency values, or "NA" if not applicable.
    auto print_latency = [](std::ostream& out, float latency) {
      if (latency < 0.0f) {
        out << "NA";
      } else {
        out << std::fixed << std::setprecision(4) << latency * 1000.0f;
      }
    };

#if TEST_NEW_ALGO_ONLY == 0
    print_latency(summary_file, result.full_sort_latency);
    summary_file << ",";
    print_latency(summary_file, result.radix_sort_latency);
    summary_file << ",";
    print_latency(summary_file, result.selection_sort_latency);
    summary_file << ",";
#endif
    print_latency(summary_file, result.hybrid_sort_latency);
    summary_file << ",";
    print_latency(summary_file, result.flash_sort_latency);
    summary_file << ",";
    print_latency(summary_file, result.llm_sort_latency);
    summary_file << ",";
    summary_file << result.best_algorithm << ",";
    if (result.best_latency == std::numeric_limits<float>::max()) {
      summary_file << "NA";
    } else {
      print_latency(summary_file, result.best_latency);
    }
#if TEST_NEW_ALGO_ONLY == 0
    summary_file << ",";
    print_latency(summary_file, result.default_latency);
#endif
    summary_file << "\n";
  }

  summary_file.close();
  std::cout << "--- CSV summary successfully written. ---\n";
}

void RunBenchmarks(const BenchmarkParams& params, std::vector<CsvSummaryResult>& csv_results) {
  std::cout << "\n--- Running Benchmarks with batch_size=" << params.batch_size << ", vocab_size=" << params.vocab_size
            << ", k=" << params.k << " ---\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto scores_in_d = Generators::CudaMallocArray<float>(static_cast<size_t>(params.batch_size) * params.vocab_size);

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
  CsvSummaryResult current_csv_result;
  current_csv_result.params = params;
  std::map<std::string, float> algo_latencies;

  auto data = std::make_unique<Generators::cuda::TopkData>(params.batch_size, params.vocab_size, stream);
#if TEST_NEW_ALGO_ONLY == 0
  // Benchmark Full Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::full_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                           params.batch_size, params.k);
    });
    all_results.push_back({params, "FULL_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.full_sort_latency = mean_ms;
    algo_latencies["FULL_SORT"] = mean_ms;
  }

  // Benchmark Selection Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::selection_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                                params.batch_size, params.k);
    });
    all_results.push_back({params, "SELECTION_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.selection_sort_latency = mean_ms;
    algo_latencies["SELECTION_SORT"] = mean_ms;
  }

  // Benchmark Radix Sort
  {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::radix_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                            params.batch_size, params.k);
    });
    all_results.push_back({params, "RADIX_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.radix_sort_latency = mean_ms;
    algo_latencies["RADIX_SORT"] = mean_ms;
  }
#endif

  // Benchmark Hybrid Sort
  if (params.k <= Generators::cuda::kHybridSortMaxK) {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::hybrid_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                             params.batch_size, params.k);
    });
    all_results.push_back({params, "HYBRID_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.hybrid_sort_latency = mean_ms;
    algo_latencies["HYBRID_SORT"] = mean_ms;
  }

  // Benchmark Flash Sort
  if (Generators::cuda::flash_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::flash_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                            params.batch_size, params.k);
    });
    all_results.push_back({params, "FLASH_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.flash_sort_latency = mean_ms;
    algo_latencies["FLASH_SORT"] = mean_ms;
  }

  if (Generators::cuda::llm_sort::IsSupported(params.batch_size, params.vocab_size, params.k)) {
    auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
      Generators::cuda::llm_sort::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                                          params.batch_size, params.k);
    });
    all_results.push_back({params, "LLM_SORT", mean_ms, stdev_ms, p95_ms});
    current_csv_result.llm_sort_latency = mean_ms;
    algo_latencies["LLM_SORT"] = mean_ms;
  }

  // Find the best algorithm overall for this configuration
  for (const auto& pair : algo_latencies) {
    if (pair.second < current_csv_result.best_latency) {
      current_csv_result.best_latency = pair.second;
      current_csv_result.best_algorithm = pair.first;
    }
  }
#if TEST_NEW_ALGO_ONLY == 0
  // Benchmark RunTopK (Backend can be any of the above algorithms)
  auto [mean_ms, stdev_ms, p95_ms] = bench_algo([&]() {
    Generators::cuda::RunTopK(data.get(), stream, scores_in_d.get(), params.vocab_size,
                              params.batch_size, params.k);
  });
  all_results.push_back({params, "DEFAULT", mean_ms, stdev_ms, p95_ms});
  current_csv_result.default_latency = mean_ms;
  algo_latencies["DEFAULT"] = mean_ms;
#endif

  csv_results.push_back(current_csv_result);

  PrintSummary(all_results);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace

TEST(TopKBenchmarks, PerformanceTests) {
  std::vector<CsvSummaryResult> csv_summary_results;

  constexpr bool is_build_pipeline = true;
  if constexpr (is_build_pipeline) {
    std::vector<int> batch_sizes = {1, 4};
    std::vector<int> vocab_sizes = {201088};
    std::vector<int> ks = {1, 2, 4, 8, 16, 32, 50, 64, 128};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }
    for (const auto& params : test_cases) {
      RunBenchmarks(params, csv_summary_results);
    }
  } else {
    // Run comprehensive tests when it is not in CI pipeline.
    std::vector<int> batch_sizes = {1, 2, 4, 8};
    std::vector<int> vocab_sizes = {512, 1024, 2048, 4096};
    for (int v = 8 * 1024; v < 64 * 1024; v += 8 * 1024) {
      vocab_sizes.push_back(v);
    }
    for (int v = 64 * 1024; v <= 256 * 1024; v += 16 * 1024) {
      vocab_sizes.push_back(v);
    }
    std::vector<int> ks = {1, 2, 4, 8, 16, 32, 64};

    std::vector<BenchmarkParams> test_cases;
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        for (int k : ks) {
          test_cases.push_back({batch_size, vocab_size, k});
        }
      }
    }

    for (const auto& params : test_cases) {
      RunBenchmarks(params, csv_summary_results);
    }
  }
  PrintCsvSummary(csv_summary_results);
}

#endif
