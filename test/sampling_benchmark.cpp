// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
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

// External global variable from main.cpp for custom model path
extern std::string g_custom_model_path;

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
  std::string model_path = MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32";
  if (strcmp(params.device_type, "NvTensorRtRtx") == 0) {
    model_path = g_custom_model_path.empty() ? MODEL_PATH "hf-internal-testing/phi3-fp16-nvtrt" : g_custom_model_path;
  }
  auto config = OgaConfig::Create(model_path.c_str());
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
      generator_params->SetSearchOption("top_k", 0);  // top_k=0 routes to SampleTopP
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
  // Add NvTensorRtRtx if model is available
  std::string resolved_nvtrt_path = g_custom_model_path.empty() ? MODEL_PATH "hf-internal-testing/phi3-fp16-nvtrt" : g_custom_model_path;
  if (std::filesystem::exists(resolved_nvtrt_path)) {
    device_types.push_back("NvTensorRtRtx");
  }

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

// Targeted benchmark for ApplyRepetitionPenalty.
// The main PerformanceTests benchmark uses repetition_penalty=1.0 (no-op early return),
// so it cannot measure the impact of the flat-array optimization. This test sets
// repetition_penalty=1.2 and pre-fills the sequence with tokens to simulate a long
// context where repetition penalty must scan many tokens.
TEST(SamplingBenchmarks, DISABLED_RepetitionPenaltyBenchmark) {
  const int vocab_size = 1000;  // Must match the tiny-random-gpt2-fp32 model's actual vocab
  const int batch_size = 1;
  const int total_runs = 500;
  const int warm_up_runs = 5;
  // Number of tokens to pre-fill in the sequence before measuring.
  // This controls how many tokens ApplyRepetitionPenalty must scan.
  const std::vector<int> prefill_lengths = {10, 50, 100};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->ClearProviders();
  auto model = OgaModel::Create(*config);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<int32_t> token_dist(0, vocab_size - 1);

  const int64_t tensor_size = static_cast<int64_t>(batch_size) * static_cast<int64_t>(vocab_size);
  std::vector<float> logits_data(tensor_size);
  auto logits_tensor = OgaTensor::Create(
      logits_data.data(),
      std::array<int64_t, 2>{static_cast<int64_t>(batch_size), static_cast<int64_t>(vocab_size)});

  std::cout << "\n--- Repetition Penalty Benchmark (vocab=" << vocab_size
            << ", batch=" << batch_size << ", penalty=1.2) ---\n";
  std::cout << std::left << std::setw(15) << "SeqLen"
            << std::setw(15) << "Mean(us)"
            << std::setw(15) << "Stdev(us)"
            << std::setw(15) << "P95(us)" << "\n";
  std::cout << std::string(60, '-') << "\n";

  for (int seq_len : prefill_lengths) {
    std::vector<double> latencies;

    for (int i = 0; i < warm_up_runs + total_runs; i++) {
      auto params = OgaGeneratorParams::Create(*model);
      params->SetSearchOption("max_length", seq_len + 10);
      params->SetSearchOption("batch_size", batch_size);
      params->SetSearchOptionBool("do_sample", true);
      params->SetSearchOption("top_k", 50);
      params->SetSearchOption("repetition_penalty", 1.2);

      // Pre-fill the sequence with random tokens
      std::vector<int32_t> prefill_tokens(seq_len);
      for (auto& t : prefill_tokens) t = token_dist(engine);

      auto generator = OgaGenerator::Create(*model, *params);
      generator->AppendTokens(prefill_tokens.data(), seq_len);

      // Now measure a single GenerateNextToken call with the pre-filled sequence
      CreateRandomLogits(logits_data.data(), 10, vocab_size, batch_size, engine);
      generator->SetLogits(*logits_tensor);

      auto start = std::chrono::high_resolution_clock::now();
      generator->GenerateNextToken();
      auto stop = std::chrono::high_resolution_clock::now();

      if (i >= warm_up_runs) {
        latencies.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
      }
    }

    double mean_us_val = mean(latencies);
    double stdev_us_val = stdev(latencies);
    double p95_us_val = percentile(latencies, 95.0);

    std::cout << std::left << std::fixed << std::setprecision(2)
              << std::setw(15) << seq_len
              << std::setw(15) << mean_us_val
              << std::setw(15) << stdev_us_val
              << std::setw(15) << p95_us_val << "\n";
  }
}

// Benchmark for BeamSearch_Cpu::SelectTop.
// Measures the time for a single GenerateNextToken call with beam search (num_beams=4).
TEST(SamplingBenchmarks, DISABLED_BeamSearchBenchmark) {
  const int batch_size = 3;
  const int num_beams = 4;
  const int max_length = 20;
  const int total_runs = 50;
  const int warm_up_runs = 5;

  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  std::vector<double> latencies;

  for (int i = 0; i < warm_up_runs + total_runs; i++) {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", max_length);
    params->SetSearchOption("batch_size", batch_size);
    params->SetSearchOption("num_beams", num_beams);
    params->SetSearchOption("length_penalty", 1.0f);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokens(input_ids);

    auto start = std::chrono::high_resolution_clock::now();
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }
    auto stop = std::chrono::high_resolution_clock::now();

    if (i >= warm_up_runs) {
      latencies.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()));
    }
  }

  double mean_us_val = mean(latencies);
  double stdev_us_val = stdev(latencies);
  double p95_us_val = percentile(latencies, 95.0);

  std::cout << "\n--- Beam Search Benchmark (batch=" << batch_size
            << ", beams=" << num_beams << ", max_length=" << max_length << ") ---\n";
  std::cout << std::fixed << std::setprecision(2)
            << "Mean: " << mean_us_val << " us, Stdev: " << stdev_us_val
            << " us, P95: " << p95_us_val << " us\n";
}

// Direct micro-benchmark for ApplyRepetitionPenalty, bypassing the model.
// Creates a GreedySearch_Cpu with pre-filled sequences and measures
// the cost of ApplyRepetitionPenalty in isolation.
TEST(SamplingBenchmarks, DISABLED_RepetitionPenaltyMicro) {
  const int vocab_size = 1000;  // Must match actual model vocab
  const int total_runs = 2000;
  const int warm_up_runs = 100;
  const std::vector<int> seq_lengths = {10, 50, 100, 200, 400};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->ClearProviders();
  auto model = OgaModel::Create(*config);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<int32_t> token_dist(0, vocab_size - 1);

  std::cout << "\n--- RepetitionPenalty Micro-Benchmark (vocab=" << vocab_size << ", penalty=1.2) ---\n";
  std::cout << std::left << std::setw(12) << "SeqLen"
            << std::setw(15) << "Mean(us)"
            << std::setw(15) << "Stdev(us)"
            << std::setw(15) << "P95(us)" << "\n";
  std::cout << std::string(57, '-') << "\n";

  for (int seq_len : seq_lengths) {
    std::vector<double> latencies;

    for (int run = 0; run < warm_up_runs + total_runs; run++) {
      // Create generator with greedy search, pre-fill sequence
      auto params = OgaGeneratorParams::Create(*model);
      params->SetSearchOption("max_length", seq_len + 10);
      params->SetSearchOption("batch_size", 1);
      params->SetSearchOptionBool("do_sample", true);
      params->SetSearchOption("top_k", 50);
      params->SetSearchOption("repetition_penalty", 1.2);

      std::vector<int32_t> prefill(seq_len);
      for (auto& t : prefill) t = token_dist(engine);

      auto generator = OgaGenerator::Create(*model, *params);
      generator->AppendTokens(prefill.data(), seq_len);

      // Set random logits so SetLogits succeeds
      std::vector<float> logits(vocab_size, 0.0f);
      for (auto& l : logits) l = std::uniform_real_distribution<float>(-5.0f, 5.0f)(engine);
      auto logits_tensor = OgaTensor::Create(logits.data(), std::array<int64_t, 2>{1LL, static_cast<int64_t>(vocab_size)});
      generator->SetLogits(*logits_tensor);

      // Measure just GenerateNextToken (which includes ApplyRepetitionPenalty internally)
      auto start = std::chrono::high_resolution_clock::now();
      generator->GenerateNextToken();
      auto stop = std::chrono::high_resolution_clock::now();

      if (run >= warm_up_runs) {
        latencies.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0);
      }
    }

    std::cout << std::left << std::fixed << std::setprecision(2)
              << std::setw(12) << seq_len
              << std::setw(15) << mean(latencies)
              << std::setw(15) << stdev(latencies)
              << std::setw(15) << percentile(latencies, 95.0) << "\n";
  }
}

// Pure sampling throughput benchmark.
// Bypasses model inference by calling SetLogits + GenerateNextToken in a loop.
// This measures ONLY the sampling pipeline (ApplyMinLength, ApplyRepetitionPenalty,
// SampleTopKTopP) using the tiny-random-gpt2-fp32 test model (vocab=1000).
// The small vocab makes this a proxy for isolating relative seq_len-dependent
// overheads (e.g. repetition penalty scaling), not for measuring absolute
// throughput at production vocab sizes (128K-200K).
TEST(SamplingBenchmarks, DISABLED_PureSamplingThroughput) {
  const int vocab_size = 1000;  // Must match model's actual vocab
  const int total_runs = 20;
  const int warm_up_runs = 2;
  const std::vector<int> seq_lengths = {1, 32, 128};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->ClearProviders();
  auto model = OgaModel::Create(*config);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<int32_t> token_dist(0, vocab_size - 1);

  const int64_t tensor_size = static_cast<int64_t>(vocab_size);
  std::vector<float> logits_data(tensor_size);

  std::cout << "\n--- Pure Sampling Throughput (vocab=" << vocab_size
            << ", top_k=40, top_p=0.8, rep_penalty=1.1, temp=0.7) ---\n"
            << "Each measurement: single SetLogits + GenerateNextToken (no model inference).\n"
            << "Sequence is pre-filled to simulate decoding at different context lengths.\n\n";
  std::cout << std::left << std::setw(12) << "SeqLen"
            << std::setw(18) << "Mean(us/sample)"
            << std::setw(15) << "Stdev(us)"
            << std::setw(15) << "P95(us)"
            << std::setw(18) << "Throughput(k/s)" << "\n";
  std::cout << std::string(78, '-') << "\n";

  for (int seq_len : seq_lengths) {
    std::vector<double> latencies;
    const int iterations = 2000;

    for (int run = 0; run < warm_up_runs * iterations / total_runs + iterations; run++) {
      auto params = OgaGeneratorParams::Create(*model);
      params->SetSearchOption("max_length", seq_len + 10);
      params->SetSearchOption("batch_size", 1);
      params->SetSearchOptionBool("do_sample", true);
      params->SetSearchOption("top_k", 40);
      params->SetSearchOption("top_p", 0.8);
      params->SetSearchOption("temperature", 0.7);
      params->SetSearchOption("repetition_penalty", 1.1);

      std::vector<int32_t> prefill(seq_len);
      for (auto& t : prefill) t = token_dist(engine);

      auto generator = OgaGenerator::Create(*model, *params);
      generator->AppendTokens(prefill.data(), seq_len);

      CreateRandomLogits(logits_data.data(), 10, vocab_size, 1, engine);
      auto logits_tensor = OgaTensor::Create(
          logits_data.data(),
          std::array<int64_t, 2>{1LL, static_cast<int64_t>(vocab_size)});
      generator->SetLogits(*logits_tensor);

      auto start = std::chrono::high_resolution_clock::now();
      generator->GenerateNextToken();
      auto stop = std::chrono::high_resolution_clock::now();

      if (run >= static_cast<int>(warm_up_runs * iterations / total_runs)) {
        latencies.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()) / 1000.0);
      }
    }

    double mean_us = mean(latencies);
    double stdev_us = stdev(latencies);
    double p95_us = percentile(latencies, 95.0);

    std::cout << std::left << std::fixed
              << std::setw(12) << seq_len
              << std::setw(18) << std::setprecision(2) << mean_us
              << std::setw(15) << stdev_us
              << std::setw(15) << p95_us
              << std::setw(18) << std::setprecision(1) << (1000000.0 / mean_us / 1000.0)
              << "\n";
  }
}
