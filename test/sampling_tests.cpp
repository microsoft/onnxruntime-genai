// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include <gtest/gtest.h>

#include "generators.h"
#include "models/model.h"
#include "search.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

template<typename T>
auto AllocateFromCpuMem(Generators::DeviceInterface& device, std::span<const T> cpu_memory) {
  auto memory = device.Allocate<float>(cpu_memory.size());
  Generators::copy(cpu_memory, memory.CpuSpan());
  memory.CopyCpuToDevice();
  return memory;
}

TEST(SamplingTests, BatchedSamplingTopPCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};
  Generators::Config config;
  config.model.vocab_size = 5;

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_p = 0.25f;
  params->search.batch_size = 4;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto logits = params->p_device->WrapMemory<float>(logits_cpu);
  generator->SetLogits(logits);
  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  EXPECT_TRUE(0 == std::memcmp(output_span.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  Generators::Config config;
  config.model.vocab_size = 5;

  int batch_size = 4;
  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = 2;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto logits_copy = logits_cpu;
  auto logits = params->p_device->WrapMemory<float>(logits_copy);
  generator->SetLogits(logits);

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};

  Generators::Config config;
  config.model.vocab_size = 5;

  int batch_size = 4;
  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = 2;
  params->search.top_p = 0.25f;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto logits_copy = logits_cpu;
  auto logits = params->p_device->WrapMemory<float>(logits_copy);
  generator->SetLogits(logits);
  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

void CreateRandomLogits(float* logits, int num_large, int vocab_size, int batch_size, std::mt19937& engine) {
  assert(num_large < vocab_size / 2);  // num_large should be much smaller than vocab_size
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
      float& value = logits[dist_large(engine) + b * vocab_size];
      if (value == 25.0f)
        i--;  // We hit the same number twice, so do it again to ensure num_large values are set to 25.0f
      else
        value = 25.0f;
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopPCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_p = 0.95f;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  std::vector<float> logits_cpu(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, *params);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.data(), num_large, config.model.vocab_size, batch_size, engine);
    auto logits_copy = logits_cpu;
    auto logits = params->p_device->WrapMemory<float>(logits_copy);
    generator->SetLogits(logits);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
      EXPECT_GT(next_token_score, 1.0f);
    }
  }
}

void SoftMax(std::span<float> scores, float temperature) {
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) { return std::exp((score - max_score) / temperature); });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // Divide each score by the sum of exponentials
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score / exp_sum; });
}

TEST(SamplingTests, RandomizedSamplingTopKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  const int batch_size = 5;
  const int k = 5;

  Generators::Config config;
  const int vocab_size = 13;  // vocab size of llama
  config.model.vocab_size = vocab_size;  // vocab size of llama

  // Create a generator
  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = k;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);

  // Create data structures for testing
  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  std::vector<float> logits_cpu(vocab_size * batch_size);
  const int num_iter = 100;
  std::map<float, int> logit_to_count;
  
  // Run test
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, *params);
    logits_cpu = std::vector<float>(vocab_size * batch_size, 0.0f);
    // Shuffle integers 1 to k randomly into logits_cpu
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < k; j++)
        logits_cpu[indices[j] + vocab_size * b] = float(k - j);
    }
    // Set logits and get generated token
    auto logits_copy = logits_cpu;
    auto logits = params->p_device->WrapMemory<float>(logits_copy);
    generator->SetLogits(logits);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }
  // Calculate expected distribution of tokens by softmaxing given logits (integers 1 through k)
  std::vector<float> expected_distributions(k);
  for (int i = 0; i < k; i++)
    expected_distributions[i] = float(i + 1);
  SoftMax(expected_distributions, 1.0f);
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  const int total_count = batch_size * num_iter;
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.1);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCpu) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 5;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = k;
  params->search.top_p = p;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  std::vector<float> logits_cpu(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    CreateRandomLogits(logits_cpu.data(), num_large, config.model.vocab_size, batch_size, engine);
    auto logits_copy = logits_cpu;
    auto logits = params->p_device->WrapMemory<float>(logits_copy);
    generator->SetLogits(logits);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

#if USE_CUDA
#include "tests_helper.cuh"

TEST(SamplingTests, BatchedSamplingTopPCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};
  int batch_size = 4;

  Generators::Config config;
  config.model.vocab_size = 5;

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_p = 0.25f;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits = AllocateFromCpuMem<float>(*params->p_device, logits_cpu);
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->SetLogits(logits);
  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  EXPECT_TRUE(0 == std::memcmp(output_span.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int batch_size = 4;

  Generators::Config config;
  config.model.vocab_size = 5;

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = 2;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits = AllocateFromCpuMem<float>(*params->p_device, logits_cpu);
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->SetLogits(logits);
  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int batch_size = 4;

  Generators::Config config;
  config.model.vocab_size = 5;

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = 2;
  params->search.top_p = 0.25f;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits = AllocateFromCpuMem<float>(*params->p_device, logits_cpu);
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->SetLogits(logits);
  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_p = 0.95f;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits_gpu = params->p_device->Allocate<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = params->p_device->Allocate<int>(config.model.vocab_size * batch_size);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    LaunchGeometricDecayKernel(logits_gpu.Span().data(), config.model.vocab_size, batch_size, num_large, 20.0f, params->p_device->GetCudaStream());
    LaunchFisherYatesKernel(logits_gpu.Span().data(), indices_buffer.Span().data(), config.model.vocab_size, batch_size, params->p_device->GetCudaStream());
    generator->SetLogits(logits_gpu); 
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    auto logits_cpu = logits_gpu.CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
      EXPECT_GT(next_token_score, 0.0001f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingTopKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  const int batch_size = 5;
  const int k = 5;

  Generators::Config config;
  const int vocab_size = 17;  // vocab size of llama
  config.model.vocab_size = vocab_size;  // vocab size of llama

  // Create a generator
  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = k;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);

  // Create data structures for testing
  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  const int num_iter = 100;
  std::map<float, int> logit_to_count;

  // Run test
  for (int i = 0; i < num_iter; i++) {
    auto generator = Generators::CreateGenerator(*model, *params);
    Generators::DeviceSpan<float> logits_gpu = params->p_device->Allocate<float>(vocab_size * batch_size);
    auto cpu_span = logits_gpu.CpuSpan();
    // Shuffle integers 1 to k randomly into cpu_span
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < k; j++)
        cpu_span[indices[j] + vocab_size * b] = float(k - j);
    }
    // Copy logits onto device, set logits, and get generated token
    logits_gpu.CopyCpuToDevice();
    generator->SetLogits(logits_gpu);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_span[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }
  // Calculate expected distribution of tokens by softmaxing given logits (integers 1 through k)
  std::vector<float> expected_distributions(k);
  for (int i = 0; i < k; i++)
    expected_distributions[i] = float(i + 1);
  SoftMax(expected_distributions, 1.0f);
  const int total_count = batch_size * num_iter;
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.1);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 5;
  float p = 0.95f;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.do_sample = true;
  params->search.top_k = k;
  params->search.top_p = p;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits_gpu = params->p_device->Allocate<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = params->p_device->Allocate<int>(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    auto generator = Generators::CreateGenerator(*model, *params);
    LaunchGeometricDecayKernel(logits_gpu.Span().data(), config.model.vocab_size, batch_size, num_large, 20.0f, params->p_device->GetCudaStream());
    LaunchFisherYatesKernel(logits_gpu.Span().data(), indices_buffer.Span().data(), config.model.vocab_size, batch_size, params->p_device->GetCudaStream());
    generator->SetLogits(logits_gpu);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    auto logits_cpu = logits_gpu.CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

TEST(SamplingTests, RandomizedSamplingSelectTopCuda) {
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int batch_size = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  Generators::Config config;
  config.model.vocab_size = 32000;  // vocab size of llama

  auto params = Generators::CreateGeneratorParams(config);
  params->search.max_length = 10;
  params->search.batch_size = batch_size;
  params->p_device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto logits_gpu = params->p_device->Allocate<float>(config.model.vocab_size * batch_size);
  auto indices_buffer = params->p_device->Allocate<int>(config.model.vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    LaunchGeometricDecayKernel(logits_gpu.Span().data(), config.model.vocab_size, batch_size, num_large, 20.0f, params->p_device->GetCudaStream());
    LaunchFisherYatesKernel(logits_gpu.Span().data(), indices_buffer.Span().data(), config.model.vocab_size, batch_size, params->p_device->GetCudaStream());
    auto generator = Generators::CreateGenerator(*model, *params);
    generator->SetLogits(logits_gpu);
    generator->GenerateNextToken();
    auto next_tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
    auto logits_cpu = logits_gpu.CopyDeviceToCpu();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      float max_score = *std::max_element(logits_cpu.begin() + config.model.vocab_size * b, logits_cpu.begin() + config.model.vocab_size * (b + 1));
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + config.model.vocab_size * b];
      EXPECT_EQ(next_token_score, max_score);
    }
  }
}

#endif