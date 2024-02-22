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

extern std::unique_ptr<OrtEnv> g_ort_env;

// Defined in sampling_tests.cpp
void CreateRandomLogits(float* logits, int num_large, int vocab_size, int batch_size, std::mt19937& engine);

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPCpu) {
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, params);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));
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
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, params);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));

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
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, params);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));

    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopPAndK(p, k, 1.0f);
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
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  float* cpu_logits = new float[vocab_size * batch_size];
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, params);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));

    cudaStreamSynchronize(params.cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopP(0.95f, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP CUDA: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopKCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
  float* cpu_logits = new float[vocab_size * batch_size];
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, params);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    cudaStreamSynchronize(params.cuda_stream);
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
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 1;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
  float* cpu_logits = new float[vocab_size * batch_size];
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, params);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));

    cudaStreamSynchronize(params.cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopPAndK(p, k, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopP+K: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSelectTopCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 12;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
  auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
  float* cpu_logits = new float[vocab_size * batch_size];
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, params);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    cudaStreamSynchronize(params.cuda_stream);
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