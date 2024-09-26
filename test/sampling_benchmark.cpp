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

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CPU;
  std::vector<float> logits_cpu(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, *params);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.data(), num_large, config.model.vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.data(), config.model.vocab_size * batch_size));
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopP(0.95f, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP CPU: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CPU;
  std::vector<float> logits_cpu(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    CreateRandomLogits(logits_cpu.data(), num_large, config.model.vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.data(), config.model.vocab_size * batch_size));

    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopK(k, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopK CPU: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPAndKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CPU;
  std::vector<float> logits_cpu(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    CreateRandomLogits(logits_cpu.data(), num_large, config.model.vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.data(), config.model.vocab_size * batch_size));

    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopKTopP(k, p, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP+K CPU: "
            << average_time << " microseconds" << std::endl;
}

#if USE_CUDA
#include "tests_helper.cuh"

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CUDA;
  std::vector<float> cpu_logits(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  auto logits_gpu = Generators::CudaMallocArray<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(config.model.vocab_size * batch_size);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    LaunchGeometricDecayKernel(logits_gpu.get(), config.model.vocab_size, batch_size, num_large, 20.0f, params->cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), config.model.vocab_size, batch_size, params->cuda_stream);
    cudaMemcpy(cpu_logits.data(), logits_gpu.get(), config.model.vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), config.model.vocab_size * batch_size));

    cudaStreamSynchronize(params->cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopP(0.95f, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params->cuda_stream);
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP CUDA: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(config.model.vocab_size * batch_size);
  std::vector<float> cpu_logits(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    LaunchGeometricDecayKernel(logits_gpu.get(), config.model.vocab_size, batch_size, num_large, 20.0f, params->cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), config.model.vocab_size, batch_size, params->cuda_stream);
    cudaMemcpy(cpu_logits.data(), logits_gpu.get(), config.model.vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), config.model.vocab_size * batch_size));
    cudaStreamSynchronize(params->cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopK(k, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopK: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPAndKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 1;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(config.model.vocab_size * batch_size);
  std::vector<float> cpu_logits(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, *params);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), config.model.vocab_size, batch_size, num_large, 20.0f, params->cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), config.model.vocab_size, batch_size, params->cuda_stream);
    cudaMemcpy(cpu_logits.data(), logits_gpu.get(), config.model.vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), config.model.vocab_size * batch_size));

    cudaStreamSynchronize(params->cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopKTopP(k, p, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params->cuda_stream);
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP+K: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSelectTopCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 12;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Needs to match batch_size

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->batch_size = batch_size;
  params->sequence_length = 1;
  params->input_ids = input_ids;
  params->device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(config.model.vocab_size * batch_size);
  std::vector<float> cpu_logits(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    LaunchGeometricDecayKernel(logits_gpu.get(), config.model.vocab_size, batch_size, num_large, 20.0f, params->cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), config.model.vocab_size, batch_size, params->cuda_stream);
    cudaMemcpy(cpu_logits.data(), logits_gpu.get(), config.model.vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), config.model.vocab_size * batch_size));
    cudaStreamSynchronize(params->cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SelectTop();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by Top1: "
            << average_time << " microseconds" << std::endl;
}

#endif