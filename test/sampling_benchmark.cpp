// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include "generators.h"
#include "models/model.h"
#include "search.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

// Defined in sampling_tests.cpp
void CreateRandomLogits(float* logits, int num_large, int vocab_size, int batch_size, std::mt19937& engine);

struct SamplingBenchmark {
  void Run() {
    std::vector<int32_t> input_ids;
    for (int i = 0; i < batch_size_; i++)
      input_ids.push_back(i);

    auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

    Generators::Config config;
    config.model.vocab_size = 32000;  // vocab size of llama

    auto params = Generators::CreateGeneratorParams(config);
    params->search.max_length = 10;
    params->search.batch_size = batch_size_;
    params->p_device = Generators::GetDeviceInterface(device_type_);
    params->device_type = device_type_;

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(5, 25);
    double total_time = 0.0;
    int num_iter = 1000;

    auto logits = params->p_device->Allocate<float>(static_cast<size_t>(config.model.vocab_size) * batch_size_);
    auto test_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iter; i++) {
      auto generator = Generators::CreateGenerator(*model, *params);
      int num_large = dist(engine);
      CreateRandomLogits(logits.CpuSpan().data(), num_large, config.model.vocab_size, batch_size_, engine);
      logits.CopyCpuToDevice();
      generator->search_->SetLogits(logits);
      params->p_device->Synchronize();
      auto start = std::chrono::high_resolution_clock::now();
      benchmark_function_(*generator);
      params->p_device->Synchronize();
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      total_time += duration.count();
      if (std::chrono::high_resolution_clock::now() - test_start > std::chrono::minutes(1)) {
        std::cout << Generators::SGR::Bg_Red << " ABORTING " << Generators::SGR::Reset << " loop due to slow performance(took more than 1 minute) on iteration " << i << std::endl;
        break;
      }
    }
    double average_time = total_time / double(num_iter);
    std::cout << "Average time taken: " << average_time << " microseconds" << std::endl;
  }

  std::function<void(Generators::Generator&)> benchmark_function_;
  int batch_size_{1};
  Generators::DeviceType device_type_{Generators::DeviceType::CPU};
};

static const char* DeviceTypeToString(Generators::DeviceType device_type) {
  switch (device_type) {
    case Generators::DeviceType::CPU:
      return "CPU";
    case Generators::DeviceType::CUDA:
      return "CUDA";
    case Generators::DeviceType::DML:
      return "DML";
    case Generators::DeviceType::WEBGPU:
      return "WEBGPU";
    default:
      return "Unknown";
  }
}

enum struct BenchmarkFunction {
  TopP,
  TopK,
  TopKTopP,
  SelectTop
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

std::function<void(Generators::Generator&)> GetBenchmarkFunction(BenchmarkFunction function) {
  switch (function) {
    case BenchmarkFunction::TopP:
      return [](Generators::Generator& generator) { generator.search_->SampleTopP(0.95f, 1.0f); };
    case BenchmarkFunction::TopK:
      return [](Generators::Generator& generator) { generator.search_->SampleTopK(5, 1.0f); };
    case BenchmarkFunction::TopKTopP:
      return [](Generators::Generator& generator) { generator.search_->SampleTopKTopP(5, 0.95f, 1.0f); };
    case BenchmarkFunction::SelectTop:
      return [](Generators::Generator& generator) { generator.search_->SelectTop(); };
    default:
      assert(false);
      return nullptr;
  }
}

struct BenchmarkParams {
  Generators::DeviceType device_type;
  int batch_size;
  BenchmarkFunction benchmark_function;

  std::string Name() const {
    return std::string() + DeviceTypeToString(device_type) + "_BatchSize_" + std::to_string(batch_size) + "_" + BenchmarkFunctionToString(benchmark_function);
  }
};

class SamplingBenchmarkTest : public ::testing::TestWithParam<BenchmarkParams> {};

TEST_P(SamplingBenchmarkTest, RunBenchmark) {
  SamplingBenchmark benchmark;
  auto params = GetParam();
  benchmark.device_type_ = params.device_type;
  benchmark.benchmark_function_ = GetBenchmarkFunction(params.benchmark_function);
  benchmark.batch_size_ = params.batch_size;
  benchmark.Run();
}

auto benchmark_values = ::testing::Values(
    BenchmarkParams{Generators::DeviceType::CPU, 1, BenchmarkFunction::TopP},
    BenchmarkParams{Generators::DeviceType::CPU, 1, BenchmarkFunction::TopK},
    BenchmarkParams{Generators::DeviceType::CPU, 1, BenchmarkFunction::TopKTopP}
#if USE_CUDA
    ,
    BenchmarkParams{Generators::DeviceType::CUDA, 1, BenchmarkFunction::TopP},
    BenchmarkParams{Generators::DeviceType::CUDA, 1, BenchmarkFunction::TopK},
    BenchmarkParams{Generators::DeviceType::CUDA, 1, BenchmarkFunction::TopKTopP},
    BenchmarkParams{Generators::DeviceType::CUDA, 1, BenchmarkFunction::SelectTop},
    BenchmarkParams{Generators::DeviceType::CUDA, 6, BenchmarkFunction::SelectTop},
    BenchmarkParams{Generators::DeviceType::CUDA, 12, BenchmarkFunction::SelectTop}
#endif
);

INSTANTIATE_TEST_SUITE_P(Benchmarks, SamplingBenchmarkTest, benchmark_values,
                         [](const ::testing::TestParamInfo<BenchmarkParams>& info) { return info.param.Name(); });
