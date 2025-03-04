// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>

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
  SelectTop
};

struct SamplingBenchmark {
  void Run() {
    std::vector<int32_t> input_ids;
    for (int i = 0; i < batch_size_; i++)
      input_ids.push_back(i);

    const int vocab_size = 32000;
    auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    config->Overlay(R"({ "model": { "vocab_size" : 32000 } })");
    config->ClearProviders();
    if (strcmp(device_type_, "cpu"))
      config->AppendProvider(device_type_);

    auto model = OgaModel::Create(*config);
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 10);
    params->SetSearchOption("batch_size", batch_size_);
    params->SetSearchOptionBool("do_sample", true);
    switch (benchmark_function_) {
      case BenchmarkFunction::TopP:
        params->SetSearchOption("top_p", 0.95f);
        break;  
      case BenchmarkFunction::TopK:
        params->SetSearchOption("top_k", 5);
        break;
      case BenchmarkFunction::TopKTopP:
        params->SetSearchOption("top_k", 5);
        params->SetSearchOption("top_p", 0.95f);
        break;
      case BenchmarkFunction::SelectTop:
        params->SetSearchOption("top_k", 1);
        break;
      default:
        assert(false);
    }

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(5, 25);
    double total_time = 0.0;
    int num_iter = 1000;

    auto logits = OgaTensor::Create<float>(nullptr, std::array<int64_t, 1>{vocab_size*batch_size_});
    auto test_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
      auto generator = OgaGenerator::Create(*model, *params);
      int num_large = dist(engine);
      CreateRandomLogits(reinterpret_cast<float*>(logits->Data()), num_large, vocab_size, batch_size_, engine);
      generator->SetLogits(*logits);
      auto start = std::chrono::high_resolution_clock::now();
      generator->GenerateNextToken();
      auto result=generator->GetNextTokens();
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      total_time += duration.count();
      if (std::chrono::high_resolution_clock::now() - test_start > std::chrono::minutes(1)) {
        std::cout << "\x1b[41m ABORTING \x1b[0m loop due to slow performance(took more than 1 minute) on iteration " << i << std::endl;
        break;
      }
    }
    double average_time = total_time / double(num_iter);
    std::cout << "Average time taken: " << average_time << " microseconds" << std::endl;
  }

  BenchmarkFunction benchmark_function_;
  int batch_size_{1};
  const char* device_type_{"cpu"};
};

static const char* BenchmarkFunctionToString(BenchmarkFunction function) {
  switch (function) {
    case BenchmarkFunction::TopP:
      return "TopP";
    case BenchmarkFunction::TopK:
      return "TopK";
    case BenchmarkFunction::TopKTopP:
      return "TopKTopP";
    case BenchmarkFunction::SelectTop:
      return "SelectTop";
    default:
      return "Unknown";
  }
}

struct BenchmarkParams {
  const char* device_type;
  int batch_size;
  BenchmarkFunction benchmark_function;

  std::string Name() const {
    return std::string() + device_type + "_BatchSize_" + std::to_string(batch_size) + "_" + BenchmarkFunctionToString(benchmark_function);
  }
};

struct SamplingBenchmarkTest : ::testing::TestWithParam<BenchmarkParams> {};

TEST_P(SamplingBenchmarkTest, RunBenchmark) {
  SamplingBenchmark benchmark;
  auto params = GetParam();
  benchmark.device_type_ = params.device_type;
  benchmark.benchmark_function_ = params.benchmark_function;
  benchmark.batch_size_ = params.batch_size;
  benchmark.Run();
}

auto benchmark_values = ::testing::Values(
    BenchmarkParams{"cpu", 1, BenchmarkFunction::TopP},
    BenchmarkParams{"cpu", 1, BenchmarkFunction::TopK},
    BenchmarkParams{"cpu", 1, BenchmarkFunction::TopKTopP}
#if USE_CUDA
    ,
    BenchmarkParams{"cuda", 1, BenchmarkFunction::TopP},
    BenchmarkParams{"cuda", 1, BenchmarkFunction::TopK},
    BenchmarkParams{"cuda", 1, BenchmarkFunction::TopKTopP},
    BenchmarkParams{"cuda", 1, BenchmarkFunction::SelectTop},
    BenchmarkParams{"cuda", 6, BenchmarkFunction::SelectTop},
    BenchmarkParams{"cuda", 12, BenchmarkFunction::SelectTop}
#endif
);

INSTANTIATE_TEST_SUITE_P(Benchmarks, SamplingBenchmarkTest, benchmark_values,
                         [](const ::testing::TestParamInfo<BenchmarkParams>& info) { return info.param.Name(); });
