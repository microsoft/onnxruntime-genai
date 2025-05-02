// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstring>  // for memcmp
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

TEST(SamplingTests, BatchedSamplingTopPCpu) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_p", 0.25f);
  params->SetSearchOption("batch_size", 4);

  auto generator = OgaGenerator::Create(*model, *params);
  auto logits_tensor = OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{4LL, 5LL});
  generator->SetLogits(*logits_tensor);

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  EXPECT_TRUE(0 == std::memcmp(expected_output.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCpu) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");

  int batch_size = 4;

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", 2);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  auto logits_tensor = OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{4LL, 5LL});
  generator->SetLogits(*logits_tensor);

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + 5 /*vocab_size*/ * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCpu) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");

  int batch_size = 4;

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", 2);
  params->SetSearchOption("top_p", 0.25f);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  auto logits_tensor = OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, 5LL});
  generator->SetLogits(*logits_tensor);

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + 5 /*vocab_size*/ * b];
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
  int batch_size = 5;
  int vocab_size = 32000;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 32000 } })");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_p", 0.95f);
  params->SetSearchOption("batch_size", batch_size);

  std::vector<float> logits_cpu(vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    auto generator = OgaGenerator::Create(*model, *params);
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.data(), num_large, vocab_size, batch_size, engine);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));

    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 1.0f);
    }
  }
}

void Softmax(std::span<float> scores, float temperature) {
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) { return std::exp((score - max_score) / temperature); });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // Divide each score by the sum of exponentials
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score / exp_sum; });
}

TEST(SamplingTests, RandomizedSamplingTopKCpu) {
  const int batch_size = 5;
  const int k = 5;
  const int vocab_size = 13;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 13 } })");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", k);
  params->SetSearchOption("batch_size", batch_size);

  // Create data structures for testing
  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  std::vector<float> logits_cpu(vocab_size * batch_size);
  const int num_iter = 100;
  std::map<float, int> logit_to_count;

  // Run test
  for (int i = 0; i < num_iter; i++) {
    logits_cpu = std::vector<float>(vocab_size * batch_size, 0.0f);
    // Shuffle integers 1 to k randomly into logits_cpu
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < k; j++)
        logits_cpu[indices[j] + vocab_size * b] = float(k - j);
    }
    // Set logits and get generated token
    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }
  // Calculate expected distribution of tokens by Softmaxing given logits (integers 1 through k)
  std::vector<float> expected_distributions(k);
  for (int i = 0; i < k; i++)
    expected_distributions[i] = float(i + 1);
  Softmax(expected_distributions, 1.0f);
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  const int total_count = batch_size * num_iter;
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.1);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCpu) {
  int batch_size = 5;
  float p = 0.95f;
  int k = 5;
  const int vocab_size = 32000;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 32000 } })");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", k);
  params->SetSearchOption("top_p", p);
  params->SetSearchOption("batch_size", batch_size);

  std::vector<float> logits_cpu(vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(5, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.data(), num_large, vocab_size, batch_size, engine);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_GT(next_token_score, 10.0f);
    }
  }
}

#if USE_CUDA
TEST(SamplingTests, BatchedSamplingTopPCuda) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  std::vector<float> logits_cpu = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.6f, 0.1f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.6f, 0.1f,
                                   0.1f, 0.1f, 0.1f, 0.1f, 0.6f};
  int batch_size = 4;
  int vocab_size = 5;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_p", 0.25f);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  EXPECT_TRUE(0 == std::memcmp(expected_output.data(), next_tokens.data(), expected_output.size() * sizeof(int32_t)));
}

TEST(SamplingTests, BatchedSamplingTopKCuda) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int batch_size = 4;
  int vocab_size = 5;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", 2);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, BatchedSamplingTopPAndKCuda) {
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0f, 1.5f, 1.25f, 0.25f, 0.25f,
                                0.25f, 2.0f, 1.25f, 1.5f, 0.25f,
                                0.25f, 2.0f, 0.25f, 1.5f, 1.25f,
                                1.25f, 0.25f, 1.5f, 0.25f, 2.0f};
  int batch_size = 4;
  int vocab_size = 5;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 5 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", 2);
  params->SetSearchOption("top_p", 0.25f);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));

  // Verify outputs match expected outputs
  generator->GenerateNextToken();
  auto next_tokens = generator->GetNextTokens();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    EXPECT_GT(next_token_score, 1.25f);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPCuda) {
  const int batch_size = 5;
  const float p = 0.95f;
  const int vocab_size = 21;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 21 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_p", p);
  params->SetSearchOption("batch_size", batch_size);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  const int n = 11;  // Number of elements to be set to large values
  const int num_iter = 5000;
  std::map<float, int> logit_to_count;

  // Run test
  for (int i = 0; i < num_iter; i++) {
    std::vector<float> logits_cpu(vocab_size * batch_size);
    // Shuffle integers 1 to n randomly into cpu_span
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < n; j++)
        logits_cpu[indices[j] + vocab_size * b] = float(n - j);
    }

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }

  // Calculate expected distribution of tokens
  std::vector<float> expected_distributions(vocab_size);
  for (int i = 0; i < vocab_size; i++) {
    if (i < n) {
      expected_distributions[i] = float(i + 1);
    } else {
      expected_distributions[i] = 0.0f;
    }
  }
  Softmax(expected_distributions, 1.0f);
  bool first_greater_than_p = true;
  float sum = 0.0;
  for (int i = n - 1; i >= 0; i--) {
    sum += expected_distributions[i];
    if (sum <= p) {
      expected_distributions[i] *= 1.0f / p;
    } else if (first_greater_than_p) {
      float accumulated = (sum - expected_distributions[i]) / p;
      expected_distributions[i] = 1.0f - accumulated;
      first_greater_than_p = false;
    } else {
      expected_distributions[i] = 0.0;
    }
  }
  for (int i = n; i < vocab_size; i++) {
    expected_distributions[i] = 0.0f;
  }
  const int total_count = batch_size * num_iter;
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.01);
  }
}

TEST(SamplingTests, RandomizedSamplingTopKCuda) {
  const int batch_size = 5;
  const int k = 5;
  const int vocab_size = 17;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 17 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", k);
  params->SetSearchOption("batch_size", batch_size);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  const int num_iter = 5000;
  std::map<float, int> logit_to_count;

  // Run test
  for (int i = 0; i < num_iter; i++) {
    std::vector<float> logits_cpu(vocab_size * batch_size);
    // Shuffle integers 1 to k randomly into cpu_span
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < k; j++)
        logits_cpu[indices[j] + vocab_size * b] = float(k - j);
    }

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }
  // Calculate expected distribution of tokens by Softmaxing given logits (integers 1 through k)
  std::vector<float> expected_distributions(k);
  for (int i = 0; i < k; i++)
    expected_distributions[i] = float(i + 1);
  Softmax(expected_distributions, 1.0f);
  const int total_count = batch_size * num_iter;
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.01);
  }
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCuda) {
  const int batch_size = 5;
  const int k = 7;
  const float p = 0.75f;
  const int vocab_size = 21;

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 21 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", k);
  params->SetSearchOption("top_p", p);
  params->SetSearchOption("batch_size", batch_size);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> indices(vocab_size);
  const int num_iter = 5000;
  std::map<float, int> logit_to_count;

  // Run test
  for (int i = 0; i < num_iter; i++) {
    std::vector<float> logits_cpu(vocab_size * batch_size);
    // Shuffle integers 1 to k randomly into cpu_span
    for (int b = 0; b < batch_size; b++) {
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine);
      for (int j = 0; j < k; j++)
        logits_cpu[indices[j] + vocab_size * b] = float(k - j);
    }

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }
  // Calculate expected distribution of tokens by Softmaxing given logits (integers 1 through k)
  std::vector<float> expected_distributions(k);
  for (int i = 0; i < k; i++)
    expected_distributions[i] = float(i + 1);
  Softmax(expected_distributions, 1.0f);
  // Calculate expected distribution of tokens by top_p
  bool first_greater_than_p = true;
  float sum = 0.0;
  for (int i = k - 1; i >= 0; i--) {
    sum += expected_distributions[i];
    if (sum <= p) {
      expected_distributions[i] *= 1.0f / p;
    } else if (first_greater_than_p) {
      float accumulated = (sum - expected_distributions[i]) / p;
      expected_distributions[i] = 1.0f - accumulated;
      first_greater_than_p = false;
    } else {
      expected_distributions[i] = 0.0;
    }
  }
  const int total_count = batch_size * num_iter;
  // Check that the distribution of tokens generated by the model is close to the expected distribution
  for (auto& [logit, count] : logit_to_count) {
    const float expected_distribution = expected_distributions[int(logit) - 1];
    EXPECT_NEAR(count / float(total_count), expected_distribution, 0.01);
  }
}

TEST(SamplingTests, RandomizedSamplingSelectTopCuda) {
  int batch_size = 5;
  int vocab_size = 32000;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};

  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->Overlay(R"({ "model": { "vocab_size" : 32000 } })");
  config->ClearProviders();
  config->AppendProvider("cuda");

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", false);
  params->SetSearchOption("batch_size", batch_size);

  std::vector<float> logits_cpu(vocab_size * batch_size);
  std::vector<int> indices(vocab_size * batch_size);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<> dist(1, 25);
  int num_iter = 100;
  for (int i = 0; i < num_iter; i++) {
    int num_large = dist(engine);
    CreateRandomLogits(logits_cpu.data(), num_large, vocab_size, batch_size, engine);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetLogits(*OgaTensor::Create(logits_cpu.data(), std::array<int64_t, 2>{batch_size, vocab_size}));
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      float max_score = *std::max_element(logits_cpu.begin() + vocab_size * b, logits_cpu.begin() + vocab_size * (b + 1));
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      EXPECT_EQ(next_token_score, max_score);
    }
  }
}
#endif
