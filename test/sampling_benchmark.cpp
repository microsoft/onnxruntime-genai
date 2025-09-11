// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define OGA_USE_SPAN 1
#include "../src/span.h"
#include <ort_genai.h>
#include "statistics_helper.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

// Defined in sampling_tests.cpp
void CreateRandomLogits(float* logits, int num_large, int vocab_size, int batch_size, std::mt19937& engine);

enum struct BenchmarkFunction {
  TopP,
  TopK,
  TopKTopP,
};

static const char* BenchmarkFunctionToString(BenchmarkFunction function) {
  switch (function) {
    case BenchmarkFunction::TopP:
      return "TopP";
    case BenchmarkFunction::TopK:
      return "TopK";
    case BenchmarkFunction::TopKTopP:
      return "TopKTopP";
    default:
      return "Unknown";
  }
}

struct BenchmarkParams {
  const char* device_type;
  int batch_size;
  int vocab_size;
  int k;
  BenchmarkFunction benchmark_function;
};

struct BenchmarkResult {
  BenchmarkParams params;
  float latency_us;
  float latency_us_stdev;
  float latency_us_95_percentile;
};

void PrintSummary(const std::vector<BenchmarkResult>& results) {
  std::vector<BenchmarkResult> cpu_results;
  std::vector<BenchmarkResult> cuda_results;

  for (const auto& result : results) {
    if (strcmp(result.params.device_type, "cpu") == 0) {
      cpu_results.push_back(result);
    } else {
      cuda_results.push_back(result);
    }
  }

  auto print_device_summary = [](const std::string& device_name, const std::vector<BenchmarkResult>& device_results) {
    if (device_results.empty()) {
      return;
    }
    // clang-format off
    std::cout << "\n--- " << device_name << " Sampling Benchmark Summary ---\n";
    std::cout << std::left
              << std::setw(8) << "Batch"
              << std::setw(12) << "Vocab"
              << std::setw(5) << "K"
              << std::setw(12) << "Function"
              << std::setw(15) << "Latency(us)"
              << std::setw(15) << "Stdev(us)"
              << std::setw(15) << "P95(us)" << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (const auto& result : device_results) {
      std::cout << std::left << std::fixed << std::setprecision(2)
                << std::setw(8) << result.params.batch_size
                << std::setw(12) << result.params.vocab_size
                << std::setw(5) << result.params.k
                << std::setw(12) << BenchmarkFunctionToString(result.params.benchmark_function)
                << std::setw(15) << result.latency_us
                << std::setw(15) << result.latency_us_stdev
                << std::setw(15) << result.latency_us_95_percentile
                << "\n";
    }
    // clang-format on 
  };

  print_device_summary("CPU", cpu_results);
  print_device_summary("CUDA", cuda_results);
}

BenchmarkResult RunBenchmark(const BenchmarkParams& params) {
  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::string overlay = R"({ "model": { "vocab_size" : )" + std::to_string(params.vocab_size) + R"( } })";
  config->Overlay(overlay.c_str());
  config->ClearProviders();
  if (strcmp(params.device_type, "cpu"))
    config->AppendProvider(params.device_type);

  auto model = OgaModel::Create(*config);
  auto generator_params = OgaGeneratorParams::Create(*model);
  generator_params->SetSearchOption("max_length", 10);
  generator_params->SetSearchOption("batch_size", params.batch_size);
  generator_params->SetSearchOptionBool("do_sample", true);

  switch (params.benchmark_function) {
    case BenchmarkFunction::TopP:
      generator_params->SetSearchOption("top_p", 0.95f);
      break;
    case BenchmarkFunction::TopK:
      generator_params->SetSearchOption("top_k", params.k);
      break;
    case BenchmarkFunction::TopKTopP:
      generator_params->SetSearchOption("top_k", params.k);
      generator_params->SetSearchOption("top_p", 0.95f);
      break;
  }

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  const int warm_up_runs = 5;
  const int total_runs = 500;

  std::vector<double> latencies;

  const int64_t tensor_size = static_cast<int64_t>(params.batch_size) * static_cast<int64_t>(params.vocab_size);
  std::vector<float> logits_data(tensor_size);
  auto logits_tensor = OgaTensor::Create(
      logits_data.data(),
      std::array<int64_t, 2>{static_cast<int64_t>(params.batch_size), static_cast<int64_t>(params.vocab_size)});

  for (int i = 0; i < warm_up_runs + total_runs; i++) {
    auto generator = OgaGenerator::Create(*model, *generator_params);

    int num_large = dist(engine);
    CreateRandomLogits(logits_data.data(), num_large, params.vocab_size, params.batch_size, engine);
    generator->SetLogits(*logits_tensor);

    auto start = std::chrono::high_resolution_clock::now();
    generator->GenerateNextToken();
    auto result = generator->GetNextTokens();
    auto stop = std::chrono::high_resolution_clock::now();

    if (i >= warm_up_runs) {
      latencies.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
    }
  }

  double mean_us = mean(latencies);
  double stdev_us = stdev(latencies);
  double p95_us = percentile(latencies, 95.0);

  return {params, static_cast<float>(mean_us), static_cast<float>(stdev_us), static_cast<float>(p95_us)};
}

TEST(SamplingBenchmarks, PerformanceTests) {
  std::vector<BenchmarkParams> test_cases;
  std::vector<const char*> device_types = {"cpu"};
#if USE_CUDA
  device_types.push_back("cuda");
#endif

  std::vector<int> batch_sizes = {1};
  std::vector<int> vocab_sizes = {201088};
  std::vector<int> ks = {1, 50};

  for (const auto& device_type : device_types) {
    for (int batch_size : batch_sizes) {
      for (int vocab_size : vocab_sizes) {
        test_cases.push_back({device_type, batch_size, vocab_size, 0, BenchmarkFunction::TopP});        
        for (int k : ks) {
          test_cases.push_back({device_type, batch_size, vocab_size, k, BenchmarkFunction::TopK});
          if (k >= 20) {
            test_cases.push_back({device_type, batch_size, vocab_size, k, BenchmarkFunction::TopKTopP});
          }
        }
      }
    }
  }

  std::vector<BenchmarkResult> all_results;
  for (const auto& params : test_cases) {
    all_results.push_back(RunBenchmark(params));
  }

  PrintSummary(all_results);
}
