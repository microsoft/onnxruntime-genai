// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speculative_sampling.h"

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {

// ComputeAcceptProb

TEST(SpeculativeSamplingTest, AcceptProbTargetGreaterAlwaysAccepts) {
  // p_target >= p_draft: clamped to 1.0
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.8f, 0.4f), 1.0f);
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.5f, 0.5f), 1.0f);
}

TEST(SpeculativeSamplingTest, AcceptProbTargetLessGivesRatio) {
  // p_target < p_draft: ratio
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.2f, 0.4f), 0.5f);
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.1f, 1.0f), 0.1f);
}

TEST(SpeculativeSamplingTest, AcceptProbTargetZeroAlwaysRejects) {
  // p_target == 0: never accept
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.0f, 0.5f), 0.0f);
}

TEST(SpeculativeSamplingTest, AcceptProbDraftZeroGuard) {
  // p_draft == 0: guard returns 1.0 instead of dividing
  EXPECT_FLOAT_EQ(ComputeAcceptProb(0.5f, 0.0f), 1.0f);
}

// BuildCorrectionDistribution

TEST(SpeculativeSamplingTest, CorrectionDistributionZerosWhereDraftDominates) {
  // p_target - p_draft at each index: [-0.2, +0.3, -0.1, 0.0]
  // After max(0, .): [0, 0.3, 0, 0], normalized: [0, 1, 0, 0]
  std::array<float, 4> p_target{0.1f, 0.5f, 0.2f, 0.2f};
  std::array<float, 4> p_draft{0.3f, 0.2f, 0.3f, 0.2f};
  std::array<float, 4> out{};

  BuildCorrectionDistribution(p_target, p_draft, out);

  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FLOAT_EQ(out[1], 1.0f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
}

TEST(SpeculativeSamplingTest, CorrectionDistributionSumsToOne) {
  // Multiple positive diffs should be normalized to sum to 1
  std::array<float, 5> p_target{0.4f, 0.1f, 0.3f, 0.1f, 0.1f};
  std::array<float, 5> p_draft{0.1f, 0.3f, 0.1f, 0.4f, 0.1f};
  std::array<float, 5> out{};

  BuildCorrectionDistribution(p_target, p_draft, out);

  float sum = std::accumulate(out.begin(), out.end(), 0.0f);
  EXPECT_NEAR(sum, 1.0f, 1e-6f);

  // Indices where p_draft dominated must be zero
  EXPECT_FLOAT_EQ(out[1], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
  EXPECT_FLOAT_EQ(out[4], 0.0f);
}

TEST(SpeculativeSamplingTest, CorrectionDistributionFallsBackToTargetWhenIdentical) {
  // p_target == p_draft -> all diffs zero -> fall back to p_target
  std::array<float, 3> p_target{0.5f, 0.3f, 0.2f};
  std::array<float, 3> p_draft{0.5f, 0.3f, 0.2f};
  std::array<float, 3> out{};

  BuildCorrectionDistribution(p_target, p_draft, out);

  EXPECT_FLOAT_EQ(out[0], 0.5f);
  EXPECT_FLOAT_EQ(out[1], 0.3f);
  EXPECT_FLOAT_EQ(out[2], 0.2f);
}

// ComputeSampledCategorical (shared with standard decode via search.cpp)

namespace {
// Dense full-vocab view of ComputeSampledCategorical for easy index assertions.
std::vector<float> SampledDense(std::span<const float> logits, int top_k, float top_p, float temperature) {
  SampledCategorical c;
  ComputeSampledCategorical(logits, top_k, top_p, temperature, c);
  return ScatterToFullVocab(c, static_cast<int>(logits.size()));
}
}  // namespace

TEST(SpeculativeSamplingTest, SamplingDistTopKKeepsOnlyTopKAndSumsToOne) {
  // logits ascending; top_k=2 keeps the two largest (indices 4,3), softmax over
  // {5,4}: e/(e+1)=0.7311, 1/(e+1)=0.2689. Others must be exactly zero.
  std::array<float, 5> logits{1.f, 2.f, 3.f, 4.f, 5.f};
  auto out = SampledDense(logits, /*top_k=*/2, /*top_p=*/0.0f, /*temperature=*/1.0f);
  EXPECT_NEAR(out[4], 0.7310586f, 1e-5f);
  EXPECT_NEAR(out[3], 0.2689414f, 1e-5f);
  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FLOAT_EQ(out[1], 0.0f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_NEAR(std::accumulate(out.begin(), out.end(), 0.0f), 1.0f, 1e-6f);
}

TEST(SpeculativeSamplingTest, SamplingDistAppliesTemperature) {
  // top_k=5 >= n=2 keeps all; top_p disabled => full softmax(logits / T).
  // logits {0,2}, T=2 => logits/T {0,1} => {1/(1+e), e/(1+e)}.
  std::array<float, 2> logits{0.f, 2.f};
  auto out = SampledDense(logits, /*top_k=*/5, /*top_p=*/0.0f, /*temperature=*/2.0f);
  EXPECT_NEAR(out[0], 0.2689414f, 1e-5f);
  EXPECT_NEAR(out[1], 0.7310586f, 1e-5f);
}

TEST(SpeculativeSamplingTest, SamplingDistTopPKeepsNucleus) {
  // logits = log(probs) so softmax == {0.5, 0.3, 0.15, 0.05}.
  // top_k=0 -> pure nucleus. top_p=0.6: cumulative 0.5 (<0.6), +0.3=0.8 (>=0.6)
  // -> keep top 2, renormalize -> {0.625, 0.375}.
  std::array<float, 4> logits{std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  auto out = SampledDense(logits, /*top_k=*/0, /*top_p=*/0.6f, /*temperature=*/1.0f);
  EXPECT_NEAR(out[0], 0.625f, 1e-5f);
  EXPECT_NEAR(out[1], 0.375f, 1e-5f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
  EXPECT_NEAR(std::accumulate(out.begin(), out.end(), 0.0f), 1.0f, 1e-6f);
}

TEST(SpeculativeSamplingTest, SamplingDistTopKTopPCombined) {
  // top_k=2 first restricts to {0.5,0.3} -> softmax {0.625,0.375}.
  // top_p=0.6 then cuts at the first token (0.625 >= 0.6) -> {1.0}.
  std::array<float, 5> logits{std::log(0.5f), std::log(0.3f), std::log(0.15f),
                              std::log(0.04f), std::log(0.01f)};
  auto out = SampledDense(logits, /*top_k=*/2, /*top_p=*/0.6f, /*temperature=*/1.0f);
  EXPECT_NEAR(out[0], 1.0f, 1e-5f);
  for (size_t i = 1; i < out.size(); ++i) EXPECT_FLOAT_EQ(out[i], 0.0f);
}

TEST(SpeculativeSamplingTest, SamplingDistFromProbsMatchesLogits) {
  // SamplingDistributionFromProbs(p) must equal the dense distribution from log(p).
  std::array<float, 4> probs{0.5f, 0.3f, 0.15f, 0.05f};
  std::array<float, 4> logits{std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  auto a = SamplingDistributionFromProbs(probs, /*top_k=*/3, /*top_p=*/0.0f, /*temperature=*/1.3f);
  auto b = SampledDense(logits, /*top_k=*/3, /*top_p=*/0.0f, /*temperature=*/1.3f);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-5f);
}

}  // namespace Generators::test
