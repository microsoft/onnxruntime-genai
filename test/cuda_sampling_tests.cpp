
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if USE_CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <cfloat>

#include "../src/cuda/cuda_sampling.h"
#include "../src/cuda/cuda_common.h"

namespace Generators {
namespace cuda {
namespace test {

// Helper function to perform softmax on the CPU, used for calculating the expected distribution.
void Softmax(std::span<float> scores, float temperature) {
  if (scores.empty()) {
    return;
  }
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score for numerical stability and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) {
    // Avoid division by zero if temperature is 0
    if (temperature == 0.0f) {
      return score == max_score ? 1.0f : 0.0f;
    }
    return std::exp((score - max_score) / temperature);
  });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // If all elements are -inf, exp_sum will be 0. Avoid division by zero.
  if (exp_sum == 0.0f) {
    // Create a uniform distribution among the original max elements
    // This is a rare edge case but important for correctness
    int count = 0;
    for(auto& score : scores) if(score > 0) count++; // exp(0)=1
    for(auto& score : scores) score = (score > 0) ? 1.0f/count : 0.0f;
    return;
  }

  // Divide each score by the sum of exponentials to get probabilities
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score / exp_sum; });
}

// Base class for sampling tests to handle common setup/teardown
class CudaSamplingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaStreamCreate(&stream_);
  }

  void TearDown() override {
    cudaStreamDestroy(stream_);
  }

  cudaStream_t stream_{};
};

// Test fixture parameterized by K to test different code paths
class CudaSamplingTopKTopPTest : public CudaSamplingTest, public ::testing::WithParamInterface<int> {};

TEST_P(CudaSamplingTopKTopPTest, StatisticalVerification) {
  // 1. Test Parameters
  const int batch_size = 4;
  const int vocab_size = 512; // Smaller vocab for faster test execution
  const int k = GetParam();
  const float p = 0.9f;
  const float temperature = 0.7f;
  const int num_iter = 5000;
  const unsigned long long initial_seed = 42;
  const double tolerance = 0.015; // Tolerance for statistical comparison

  // 2. Setup CUDA resources and memory
  auto sampling_data = std::make_unique<SamplingData>(initial_seed, batch_size, vocab_size, stream_);
  auto d_scores = CudaMallocArray<float>(static_cast<size_t>(batch_size) * vocab_size);
  auto d_next_tokens = CudaMallocArray<int32_t>(batch_size);
  std::vector<int32_t> h_next_tokens(batch_size);
  std::map<int, int> token_counts;

  // 3. Prepare input logits on the host
  // We create a simple descending ramp for the first K tokens and -inf for the rest.
  std::vector<float> h_scores(static_cast<size_t>(batch_size) * vocab_size, -FLT_MAX);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      h_scores[static_cast<size_t>(i) * vocab_size + j] = static_cast<float>(k - j);
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(d_scores.get(), h_scores.data(), h_scores.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));

  // 4. Calculate expected probability distribution on the CPU
  std::vector<float> top_k_logits(k);
  std::iota(top_k_logits.rbegin(), top_k_logits.rend(), 1.0f); // Fills with {k, k-1, ..., 1}

  // 4.1. Apply temperature and initial softmax
  std::vector<float> initial_probs = top_k_logits;
  Softmax(initial_probs, temperature);

  // 4.2. Filter logits based on Top-P
  float cumulative_prob = 0.0f;
  for (int i = 0; i < k; ++i) {
    if (cumulative_prob >= p) {
      top_k_logits[i] = -FLT_MAX;
    }
    cumulative_prob += initial_probs[i];
  }

  // 4.3. Re-normalize to get final expected distribution
  std::vector<float> expected_distribution = top_k_logits;
  Softmax(expected_distribution, 1.0f); // Temperature is 1.0 for the final softmax

  // 5. Run the CUDA kernel in a loop to gather statistics
  std::mt19937 engine(initial_seed);
  std::uniform_int_distribution<unsigned long long> dist;

  for (int i = 0; i < num_iter; ++i) {
    sampling_data->ReInitCurandStates(dist(engine), batch_size, stream_);
    GetSample(sampling_data.get(), stream_, d_next_tokens.get(), d_scores.get(),
              vocab_size, batch_size, k, p, temperature);
    CUDA_CHECK(cudaMemcpyAsync(h_next_tokens.data(), d_next_tokens.get(), h_next_tokens.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    for (int32_t token_id : h_next_tokens) {
      token_counts[token_id]++;
    }
  }

  // 6. Verify the observed distribution against the expected one
  const double total_samples = static_cast<double>(batch_size * num_iter);
  double observed_total_prob = 0.0;

  for (int i = 0; i < k; ++i) {
    double expected_prob = expected_distribution[i];
    double observed_prob = 0.0;
    if (token_counts.count(i)) {
      observed_prob = token_counts[i] / total_samples;
    }

    if (expected_prob > 0) {
      EXPECT_NEAR(observed_prob, expected_prob, tolerance) << "Mismatch for token_id: " << i;
    } else {
      // If a token has 0 expected probability, it should not be generated.
      EXPECT_EQ(observed_prob, 0.0) << "Token " << i << " was generated but should have been filtered by Top-P.";
    }
    observed_total_prob += observed_prob;
  }

  // The sum of probabilities for the selected tokens should be close to 1.
  EXPECT_NEAR(observed_total_prob, 1.0, tolerance);
  // Verify no tokens outside the top-K range were ever selected.
  for (auto const& [token_id, count] : token_counts) {
      EXPECT_GE(token_id, 0);
      EXPECT_LT(token_id, k);
  }
}

// Instantiate the test for k=64 (Selection Sort -> Fused Kernel) and k=300 (Full Sort -> Multi-Stage Kernel)
INSTANTIATE_TEST_SUITE_P(
    SamplingKernels,
    CudaSamplingTopKTopPTest,
    ::testing::Values(64, 300)
);


TEST_F(CudaSamplingTest, TopKOnlyVerification) {
  // This test disables Top-P to verify the initial softmax and sampling logic.
  const int batch_size = 4;
  const int vocab_size = 256;
  const int k = 50;
  const float p = 1.0f; // p=1.0 disables Top-P filtering
  const float temperature = 1.0f;
  const int num_iter = 5000;
  const unsigned long long initial_seed = 42;

  auto sampling_data = std::make_unique<SamplingData>(initial_seed, batch_size, vocab_size, stream_);
  auto d_scores = CudaMallocArray<float>(static_cast<size_t>(batch_size) * vocab_size);
  auto d_next_tokens = CudaMallocArray<int32_t>(batch_size);
  std::vector<int32_t> h_next_tokens(batch_size);
  std::map<int, int> token_counts;

  std::vector<float> h_scores(static_cast<size_t>(batch_size) * vocab_size, -FLT_MAX);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      h_scores[static_cast<size_t>(i) * vocab_size + j] = static_cast<float>(k - j);
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(d_scores.get(), h_scores.data(), h_scores.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));

  // Expected distribution is just a softmax over the top K logits.
  std::vector<float> expected_distribution(k);
  std::iota(expected_distribution.rbegin(), expected_distribution.rend(), 1.0f);
  Softmax(expected_distribution, temperature);

  std::mt19937 engine(initial_seed);
  std::uniform_int_distribution<unsigned long long> dist;

  for (int i = 0; i < num_iter; ++i) {
    sampling_data->ReInitCurandStates(dist(engine), batch_size, stream_);
    GetSample(sampling_data.get(), stream_, d_next_tokens.get(), d_scores.get(),
              vocab_size, batch_size, k, p, temperature);
    CUDA_CHECK(cudaMemcpyAsync(h_next_tokens.data(), d_next_tokens.get(), h_next_tokens.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    for (int32_t token_id : h_next_tokens) {
      token_counts[token_id]++;
    }
  }

  const double total_samples = static_cast<double>(batch_size * num_iter);
  for (int i = 0; i < k; ++i) {
    double observed_prob = token_counts.count(i) ? token_counts[i] / total_samples : 0.0;
    EXPECT_NEAR(observed_prob, expected_distribution[i], 0.015);
  }
}

TEST_F(CudaSamplingTest, DeterministicTop1) {
    // With k=1, the result should always be the token with the highest logit, regardless of other parameters.
    const int batch_size = 8;
    const int vocab_size = 1024;
    const int k = 1;
    const float p = 0.5f;
    const float temperature = 0.01f;
    const unsigned long long seed = 1337;

    auto sampling_data = std::make_unique<SamplingData>(seed, batch_size, vocab_size, stream_);
    auto d_scores = CudaMallocArray<float>(static_cast<size_t>(batch_size) * vocab_size);
    auto d_next_tokens = CudaMallocArray<int32_t>(batch_size);
    std::vector<int32_t> h_next_tokens(batch_size);

    // Create random logits, but ensure a unique max for each batch item.
    std::vector<float> h_scores(static_cast<size_t>(batch_size) * vocab_size);
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    std::vector<int> expected_tokens(batch_size);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            h_scores[static_cast<size_t>(i) * vocab_size + j] = dist(engine);
        }
        // Set a guaranteed maximum value at a known index.
        int max_index = engine() % vocab_size;
        h_scores[static_cast<size_t>(i) * vocab_size + max_index] = 101.0f;
        expected_tokens[i] = max_index;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_scores.get(), h_scores.data(), h_scores.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));

    GetSample(sampling_data.get(), stream_, d_next_tokens.get(), d_scores.get(),
              vocab_size, batch_size, k, p, temperature);

    CUDA_CHECK(cudaMemcpyAsync(h_next_tokens.data(), d_next_tokens.get(), h_next_tokens.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(h_next_tokens[i], expected_tokens[i]) << "Mismatch in batch item " << i;
    }
}

} // namespace test
} // namespace cuda
} // namespace Generators
#endif