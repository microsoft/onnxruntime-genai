// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstring>  // for memcmp
#include <numeric>
#include <random>
#include <limits>
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
  params->SetSearchOption("top_k", 1);
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

/**
 * @brief Helper function to run a randomized sampling test with specified parameters.
 *
 * This function encapsulates the entire test logic, including setting up the model,
 * generating tokens over multiple iterations, calculating the expected probability
 * distribution on the CPU, and verifying the results.
 *
 * @param batch_size The number of sequences to process in parallel.
 * @param k The number of highest probability vocabulary tokens to keep for top-k-filtering.
 * @param p The cumulative probability threshold for nucleus sampling (top-p).
 * @param vocab_size The size of the vocabulary.
 * @param num_iter The number of sampling iterations to run for statistical significance.
 * @param temperature The value used to module the next token probabilities.
 * @param use_cuda Whether to use CUDA for model inference.
 */
void RunSamplingTest(int batch_size, int k, float p, int vocab_size, int num_iter, float temperature, bool use_cuda) {
  // --- 1. Setup Model and Generator Parameters ---
  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::string overlay_json = R"({ "model": { "vocab_size" : )" + std::to_string(vocab_size) + R"( } })";
  config->Overlay(overlay_json.c_str());

  if (use_cuda) {
    config->ClearProviders();
    config->AppendProvider("cuda");
  }

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOptionBool("do_sample", true);
  params->SetSearchOption("top_k", k);
  if (p > 0.0f) {
    params->SetSearchOption("top_p", p);
  }
  params->SetSearchOption("temperature", temperature);
  params->SetSearchOption("batch_size", batch_size);

  // --- 2. Initialize Logits and Random Generators ---
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_int_distribution<int> dist(
      std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max());

  std::vector<int> indices(vocab_size);
  std::map<float, int> logit_to_count;

  std::vector<float> logits_cpu(static_cast<size_t>(vocab_size) * static_cast<size_t>(batch_size));
  // Create a predictable set of logits with the top k values being {k, k-1, ..., 1}
  for (int b = 0; b < batch_size; b++) {
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), engine);
    for (int j = 0; j < k; j++) {
      logits_cpu[indices[j] + vocab_size * b] = float(k - j);
    }
  }

  // --- 3. Run Generation Loop ---
  std::array<int64_t, 2> shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(vocab_size)};
  for (int i = 0; i < num_iter; i++) {
    params->SetSearchOption("random_seed", static_cast<double>(dist(engine)));
    auto generator = OgaGenerator::Create(*model, *params);

    auto logits_tensor = OgaTensor::Create(logits_cpu.data(), shape);
    generator->SetLogits(*logits_tensor);
    generator->GenerateNextToken();
    auto next_tokens = generator->GetNextTokens();

    // Collect statistics on the generated tokens
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      logit_to_count[next_token_score]++;
      EXPECT_GT(next_token_score, 0.0f);
    }
  }

  // --- 4. Calculate Expected Distribution on CPU ---
  std::vector<float> top_k_logits(k);
  std::iota(top_k_logits.rbegin(), top_k_logits.rend(), 1.0f);  // Fills with {k, k-1, ..., 1}

  // Apply temperature to get initial probabilities
  std::vector<float> initial_probs = top_k_logits;
  Softmax(initial_probs, temperature);

  // Apply top-p filtering
  float cumulative_prob = 0.0f;
  std::vector<float> filtered_logits = top_k_logits;
  if (p < 1.0f) {
    for (int i = 0; i < k; ++i) {
      if (cumulative_prob >= p) {
        filtered_logits[i] = std::numeric_limits<float>::lowest();
      }
      cumulative_prob += initial_probs[i];
    }
  }

  // Re-normalize the filtered logits to get the final expected distribution.
  // The final softmax should always use temperature=1.0 according to the sampling logic.
  std::vector<float> expected_distributions = filtered_logits;
  Softmax(expected_distributions, 1.0f);

  // --- 5. Verify Results ---
  const int total_count = batch_size * num_iter;
  for (const auto& [logit, count] : logit_to_count) {
    // Map the logit value back to its index in the sorted distribution array
    const int logit_index = k - static_cast<int>(logit);

    ASSERT_GE(logit_index, 0);
    ASSERT_LT(logit_index, k);

    const float expected_prob = expected_distributions[logit_index];
    const float actual_prob = static_cast<float>(count) / total_count;

    EXPECT_NEAR(actual_prob, expected_prob, 0.015);
  }
}

TEST(SamplingTests, RandomizedSamplingTopKCpu) {
  RunSamplingTest(/*batch_size*/ 5, /*k*/ 5, /*p*/ 1.0f, /*vocab_size*/ 13, /*num_iter*/ 1000, /*temperature*/ 1.0f, /*use_cuda*/ false);
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCpu) {
  RunSamplingTest(/*batch_size*/ 5, /*k*/ 7, /*p*/ 0.75f, /*vocab_size*/ 21, /*num_iter*/ 1000, /*temperature*/ 1.0f, /*use_cuda*/ false);
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
  RunSamplingTest(/*batch_size*/ 5, /*k*/ 11, /*p*/ 0.95f, /*vocab_size*/ 21, /*num_iter*/ 5000, /*temperature*/ 1.0f, /*use_cuda*/ true);
}

TEST(SamplingTests, RandomizedSamplingTopKCuda) {
  RunSamplingTest(/*batch_size*/ 5, /*k*/ 5, /*p*/ 1.0f, /*vocab_size*/ 17, /*num_iter*/ 5000, /*temperature*/ 1.0f, /*use_cuda*/ true);
}

TEST(SamplingTests, RandomizedSamplingTopPAndKCuda) {
  RunSamplingTest(/*batch_size*/ 5, /*k*/ 7, /*p*/ 0.75f, /*vocab_size*/ 21, /*num_iter*/ 5000, /*temperature*/ 1.0f, /*use_cuda*/ true);
}

TEST(SamplingTests, RandomizedSamplingSelectTopCuda) {
  int batch_size = 5;
  int vocab_size = 32000;

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
