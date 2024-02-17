// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_CUDA
#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <random>
#include <chrono>
#include "tests_helper.cuh"

TEST(Benchmarks, BenchmarkRandomizedSamplingTopP) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
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
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    auto generator = Generators::CreateGenerator(*model, params);
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
  std::cout << "Average time taken by TopP: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopK) {
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
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));

    cudaStreamSynchronize(params.cuda_stream);
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopK(k, 1.0f);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    total_time += duration.count();

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
  }
  double average_time = total_time / double(num_iter);
  std::cout << "Average time taken by TopK: "
            << average_time << " microseconds" << std::endl;
}

TEST(Benchmarks, BenchmarkRandomizedSamplingTopPAndK) {
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
  double total_time = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpy(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    auto generator = Generators::CreateGenerator(*model, params);
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

#endif