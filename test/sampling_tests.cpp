// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <random>

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../../test/test_models/"

TEST(SamplingTests, BatchedSamplingTopPCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  auto generator = Generators::CreateGenerator(*model, params);
  auto logits_span = Generators::cpu_span<float>(logits_cpu);
  generator->search_->SetLogits(logits_span);
  // Verify outputs match expected outputs
  generator->search_->SampleTopP(0.25f, 1.0f);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  EXPECT_TRUE(0 == std::memcmp(output_span.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  auto generator = Generators::CreateGenerator(*model, params);
  auto logits_span = Generators::cpu_span<float>(logits_cpu);
  generator->search_->SetLogits(logits_span);
  ;
  // Verify outputs match expected outputs
  int k = 2;
  generator->search_->SampleTopK(k, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  auto generator = Generators::CreateGenerator(*model, params);
  auto logits_span = Generators::cpu_span<float>(logits_cpu);
  generator->search_->SetLogits(logits_span);
  // Verify outputs match expected outputs
  float p = 0.25f;
  int k = 2;
  generator->search_->SampleTopPAndK(p, k, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

void CreateRandomLogits(float* logits, int num_large, int vocab_size, int batch_size, std::mt19937& engine) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int b = 0; b < batch_size; b++) {
    for (int v = 0; v < vocab_size; v++) {
      logits[v + b * vocab_size] = dist(engine);
    }
  }
  // Randomly set num_large elements to be large
  std::uniform_int_distribution<> dist_large(0, vocab_size - 1);
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < num_large; i++) {
      int index = dist_large(engine);
      logits[index + b * vocab_size] = 25.0f;
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopPCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopP(0.95f, 1.0f);
    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 1.0f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopKCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CPU;
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(5, 25);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopK(k, 1.0f);
    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCpu) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
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
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    std::unique_ptr<float[]> logits_cpu(new float[vocab_size * batch_size]);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(5, 25);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.get(), num_large, vocab_size, batch_size, engine);
    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::cpu_span<float>(logits_cpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopPAndK(p, k, 1.0f);
    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

#if USE_CUDA
#include "tests_helper.cuh"

TEST(SamplingTests, BatchedSamplingTopPCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);
  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  // Verify outputs match expected outputs
  generator->search_->SampleTopP(0.25f, 1.0f);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  EXPECT_TRUE(0 == std::memcmp(output_span.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);
  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  // Verify outputs match expected outputs
  int k = 2;
  generator->search_->SampleTopK(k, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);
  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  // Verify outputs match expected outputs
  float p = 0.25f;
  int k = 2;
  generator->search_->SampleTopPAndK(p, k, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopP(0.95f, 1.0f);

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_logits[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 0.0001f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopKCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopK(k, 1.0f);
    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_logits[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCuda) {
  std::unique_ptr<OrtEnv> g_ort_env;
  Ort::InitApi();
  g_ort_env = OrtEnv::Create();
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000;  // vocab size of llama
  int batch_size = 5;
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
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_int_distribution<> dist(1, 25);
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, num_large, 20.0f, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    generator->search_->SampleTopPAndK(p, k, 1.0f);
    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_logits[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

#endif