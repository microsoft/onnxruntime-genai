// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <random>
#include <chrono>

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
    params->batch_size = batch_size_;
    params->sequence_length = 1;
    params->input_ids = input_ids;
    params->p_device = Generators::GetDeviceInterface(device_type_);
    params->device_type = device_type_;

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(5, 25);
    double total_time = 0.0;
    int num_iter = 1000;

    auto logits = params->p_device->Allocate<float>(config.model.vocab_size * batch_size_);

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
    }
    double average_time = total_time / double(num_iter);
    std::cout << "Average time taken: " << average_time << " microseconds" << std::endl;
  }

  std::function<void(Generators::Generator&)> benchmark_function_;
  int batch_size_ {1};
  Generators::DeviceType device_type_{Generators::DeviceType::CPU};
};

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPCpu) {
  SamplingBenchmark benchmark;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopP(0.95f, 1.0f);
  };
  benchmark.Run();
}
TEST(Benchmarks, BenchmarkRandomizedSamplingTopKCpu) {
  SamplingBenchmark benchmark;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopK(5, 1.0f);
  };
  benchmark.Run();
}
TEST(Benchmarks, BenchmarkRandomizedSamplingTopPAndKCpu) {
  SamplingBenchmark benchmark;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopKTopP(5, 0.95f, 1.0f);
  };
  benchmark.Run();
}

#if USE_CUDA
TEST(Benchmarks, BenchmarkRandomizedSamplingTopPCuda) {
  SamplingBenchmark benchmark;
  benchmark.device_type_ = Generators::DeviceType::CUDA;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopP(0.95f, 1.0f);
  };
  benchmark.Run();
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopKCuda) {
  SamplingBenchmark benchmark;
  benchmark.device_type_ = Generators::DeviceType::CUDA;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopK(5, 1.0f);
  };
  benchmark.Run();
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPAndKCuda) {
  SamplingBenchmark benchmark;
  benchmark.device_type_ = Generators::DeviceType::CUDA;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SampleTopKTopP(5, 0.95f, 1.0f);
  };
  benchmark.Run();
}

TEST(Benchmarks, BenchmarkRandomizedSelectTopCuda) {
  SamplingBenchmark benchmark;
  benchmark.batch_size_ = 12;
  benchmark.device_type_ = Generators::DeviceType::CUDA;
  benchmark.benchmark_function_ = [](Generators::Generator& generator) {
    generator.search_->SelectTop();
  };
  benchmark.Run();
}

#endif