// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speculative_decoding_strategy.h"

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {

TEST(SpeculativeProposalTest, ModeDoesNotDependOnProbabilityStorage) {
  // These intentionally inconsistent buffers prove that mode, not storage shape, selects behavior.
  // Runtime proposal validation rejects such inconsistencies before verification.
  SpeculativeDecodingStrategy::Proposal greedy{
      SpeculativeDecodingStrategy::ProposalMode::kGreedyMatch};
  greedy.probs.resize(1);
  EXPECT_FALSE(greedy.UsesDraftProbabilities());

  SpeculativeDecodingStrategy::Proposal sampling{
      SpeculativeDecodingStrategy::ProposalMode::kDraftSampling};
  EXPECT_TRUE(sampling.UsesDraftProbabilities());

  SpeculativeDecodingStrategy::Proposal deterministic{
      SpeculativeDecodingStrategy::ProposalMode::kDeterministic};
  deterministic.probs.resize(1);
  EXPECT_FALSE(deterministic.UsesDraftProbabilities());
}

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

// TargetTokenSelection

TEST(SpeculativeSamplingTest, TargetGreedySelectionAppliesMinLengthBeforeArgmax) {
  std::array<float, 4> logits{1.0f, 2.0f, 8.0f, 3.0f};
  std::array<int32_t, 1> eos_ids{2};
  LogitsPenaltyProcessor penalties{
      static_cast<int>(logits.size()), /*repetition_penalty=*/1.0f, /*min_length=*/5, eos_ids};
  SampledCategorical sampled;
  TargetTokenSelection selection;

  ComputeTargetTokenSelection(
      logits, /*current_length=*/4, {}, /*greedy=*/true, /*top_k=*/0, /*top_p=*/0.0f,
      /*temperature=*/1.0f, penalties, sampled, selection);

  EXPECT_EQ(selection.greedy_token, 3);
  EXPECT_TRUE(selection.indices.empty());
  EXPECT_TRUE(selection.probs.empty());
}

TEST(SpeculativeSamplingTest, TargetSamplingAppliesRepetitionPenaltyBeforeTopK) {
  std::array<float, 4> logits{6.0f, 5.0f, 4.0f, 1.0f};
  std::array<int32_t, 1> prefix{0};
  std::array<int32_t, 1> eos_ids{3};
  LogitsPenaltyProcessor penalties{
      static_cast<int>(logits.size()), /*repetition_penalty=*/2.0f, /*min_length=*/0, eos_ids};
  SampledCategorical sampled;
  TargetTokenSelection selection;

  ComputeTargetTokenSelection(
      logits, /*current_length=*/1, prefix, /*greedy=*/false, /*top_k=*/2, /*top_p=*/0.0f,
      /*temperature=*/1.0f, penalties, sampled, selection);

  ASSERT_EQ(selection.indices.size(), 2u);
  ASSERT_EQ(selection.probs.size(), 2u);
  EXPECT_EQ(selection.indices[0], 1);
  EXPECT_EQ(selection.indices[1], 2);
  EXPECT_FLOAT_EQ(GetTargetTokenProbability(selection, 0), 0.0f);
  EXPECT_GT(GetTargetTokenProbability(selection, 1),
            GetTargetTokenProbability(selection, 2));
}

TEST(SpeculativeSamplingTest, TargetSamplingSelectionDensifiesSparseCategorical) {
  std::array<float, 4> logits{1.0f, 4.0f, 3.0f, 2.0f};
  std::array<int32_t, 1> eos_ids{3};
  LogitsPenaltyProcessor penalties{
      static_cast<int>(logits.size()), /*repetition_penalty=*/1.0f, /*min_length=*/0, eos_ids};
  SampledCategorical sampled;
  TargetTokenSelection selection;
  std::vector<float> dense;

  ComputeTargetTokenSelection(
      logits, /*current_length=*/0, {}, /*greedy=*/false, /*top_k=*/2, /*top_p=*/0.0f,
      /*temperature=*/1.0f, penalties, sampled, selection);
  DensifyTargetTokenSelection(selection, static_cast<int>(logits.size()), dense);

  EXPECT_FLOAT_EQ(dense[0], 0.0f);
  EXPECT_GT(dense[1], dense[2]);
  EXPECT_FLOAT_EQ(dense[3], 0.0f);
  EXPECT_NEAR(std::accumulate(dense.begin(), dense.end(), 0.0f), 1.0f, 1e-6f);
}

TEST(SpeculativeSamplingTest, DeterministicProposalAcceptsMatchingSampleAndDrawsBonus) {
  TargetTokenSelection first;
  first.indices = {7};
  first.probs = {1.0f};
  std::array<TargetTokenSelection, 1> subsequent;
  subsequent[0].indices = {9};
  subsequent[0].probs = {1.0f};
  std::array<int32_t, 1> proposal{7};
  std::mt19937 rng{42};

  const auto result = VerifyDeterministicProposal(proposal, first, subsequent, rng);

  EXPECT_EQ(result.accepted_count, 1);
  EXPECT_EQ(result.evaluated_count, 1);
  EXPECT_EQ(result.final_token, 9);
  EXPECT_TRUE(result.used_bonus);
}

TEST(SpeculativeSamplingTest, DeterministicProposalMismatchCommitsSampledTarget) {
  TargetTokenSelection first;
  first.indices = {5};
  first.probs = {1.0f};
  std::array<TargetTokenSelection, 1> subsequent;
  subsequent[0].indices = {9};
  subsequent[0].probs = {1.0f};
  std::array<int32_t, 1> proposal{7};
  std::mt19937 rng{42};

  const auto result = VerifyDeterministicProposal(proposal, first, subsequent, rng);

  EXPECT_EQ(result.accepted_count, 0);
  EXPECT_EQ(result.evaluated_count, 1);
  EXPECT_EQ(result.final_token, 5);
  EXPECT_FALSE(result.used_bonus);
}

TEST(SpeculativeSamplingTest, DeterministicProposalStopsDrawingAfterFirstMismatch) {
  TargetTokenSelection first;
  first.indices = {1, 2};
  first.probs = {0.4f, 0.6f};
  std::array<TargetTokenSelection, 2> subsequent;
  subsequent[0].indices = {3, 4};
  subsequent[0].probs = {0.3f, 0.7f};
  subsequent[1].indices = {5, 6};
  subsequent[1].probs = {0.2f, 0.8f};
  std::array<int32_t, 2> proposal{99, 4};
  std::mt19937 actual_rng{1234};
  std::mt19937 expected_rng{1234};
  SampleTargetToken(first, expected_rng);

  const auto result =
      VerifyDeterministicProposal(proposal, first, subsequent, actual_rng);

  EXPECT_EQ(result.evaluated_count, 1);
  EXPECT_FALSE(result.used_bonus);
  EXPECT_EQ(actual_rng, expected_rng);
}

TEST(SpeculativeSamplingTest, DeterministicProposalDoesNotDrawBonusAfterLaterMismatch) {
  TargetTokenSelection first;
  first.indices = {1};
  first.probs = {1.0f};
  std::array<TargetTokenSelection, 2> subsequent;
  subsequent[0].indices = {2};
  subsequent[0].probs = {1.0f};
  subsequent[1].indices = {3};
  subsequent[1].probs = {1.0f};
  std::array<int32_t, 2> proposal{1, 9};
  std::mt19937 rng{42};

  const auto result = VerifyDeterministicProposal(proposal, first, subsequent, rng);

  EXPECT_EQ(result.accepted_count, 1);
  EXPECT_EQ(result.evaluated_count, 2);
  EXPECT_EQ(result.final_token, 2);
  EXPECT_FALSE(result.used_bonus);
}

// ---------------------------------------------------------------------------
// Boundary settings for ComputeSampledCategorical. Standard decode (search.cpp's
// SampleTop*) and speculative decoding both build their truncated distribution
// through this one shared function, so pinning its behavior at boundary settings
// guards against silent drift in either path after the search.cpp refactor.
// ---------------------------------------------------------------------------

TEST(SpeculativeSamplingTest, TopKOneKeepsOnlyArgmax) {
  std::array<float, 5> logits{1.f, 5.f, 2.f, 4.f, 3.f};  // argmax is index 1
  auto out = SampledDense(logits, /*top_k=*/1, /*top_p=*/0.0f, /*temperature=*/1.0f);
  EXPECT_FLOAT_EQ(out[1], 1.0f);
  for (size_t i = 0; i < out.size(); ++i)
    if (i != 1) EXPECT_FLOAT_EQ(out[i], 0.0f);
}

TEST(SpeculativeSamplingTest, TopPNearZeroKeepsSingleToken) {
  // Nucleus with a tiny p keeps just the single most probable token.
  std::array<float, 4> logits{std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  auto out = SampledDense(logits, /*top_k=*/0, /*top_p=*/0.01f, /*temperature=*/1.0f);
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  for (size_t i = 1; i < out.size(); ++i) EXPECT_FLOAT_EQ(out[i], 0.0f);
}

TEST(SpeculativeSamplingTest, TopKExceedingVocabIsClampedNotOutOfBounds) {
  // top_k far larger than vocab must clamp (no OOB) and behave like top_k == vocab.
  std::array<float, 3> logits{1.f, 2.f, 3.f};
  auto clamped = SampledDense(logits, /*top_k=*/100, /*top_p=*/0.0f, /*temperature=*/1.0f);
  auto full = SampledDense(logits, /*top_k=*/3, /*top_p=*/0.0f, /*temperature=*/1.0f);
  for (size_t i = 0; i < logits.size(); ++i) EXPECT_NEAR(clamped[i], full[i], 1e-6f);
  EXPECT_NEAR(std::accumulate(clamped.begin(), clamped.end(), 0.0f), 1.0f, 1e-6f);
}

TEST(SpeculativeSamplingTest, LowTemperatureSharpensTowardArgmax) {
  std::array<float, 3> logits{0.f, 1.f, 2.f};
  auto out = SampledDense(logits, /*top_k=*/3, /*top_p=*/0.0f, /*temperature=*/0.05f);
  EXPECT_GT(out[2], 0.99f);  // nearly all mass collapses onto the argmax
}

TEST(SpeculativeSamplingTest, HighTemperatureFlattensTowardUniform) {
  std::array<float, 3> logits{0.f, 1.f, 2.f};
  auto out = SampledDense(logits, /*top_k=*/3, /*top_p=*/0.0f, /*temperature=*/1000.0f);
  for (float v : out) EXPECT_NEAR(v, 1.0f / 3.0f, 0.02f);
}

// ---------------------------------------------------------------------------
// Statistical equivalence (Monte Carlo). The core correctness guarantee of
// speculative decoding is that the committed token is distributed exactly as the
// target, regardless of how good or bad the draft proposal is. We exercise the
// real accept/correct primitives (ComputeAcceptProb, BuildCorrectionDistribution)
// and assert empirical token frequencies match the target distribution.
// ---------------------------------------------------------------------------

namespace {

// One accept/correct step using the same primitives and idiom as RunRound: sample
// a draft token, accept it with probability min(1, p_target/p_draft), otherwise
// sample a correction token from normalize(max(0, p_target - p_draft)).
int SimulateAcceptCorrectStep(std::span<const float> p_target,
                              std::span<const float> p_draft, std::mt19937& rng) {
  std::discrete_distribution<int> draft_dist(p_draft.begin(), p_draft.end());
  const int draft_token = draft_dist(rng);

  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  if (uniform(rng) < ComputeAcceptProb(p_target[draft_token], p_draft[draft_token]))
    return draft_token;

  std::vector<float> correction(p_target.size());
  BuildCorrectionDistribution(p_target, p_draft, {correction.data(), correction.size()});
  std::discrete_distribution<int> correction_dist(correction.begin(), correction.end());
  return correction_dist(rng);
}

// Assert the accept/correct step reproduces p_target within tolerance.
void ExpectAcceptCorrectMatchesTarget(std::span<const float> p_target,
                                      std::span<const float> p_draft, uint32_t seed,
                                      float tol = 0.01f, int iterations = 200000) {
  std::mt19937 rng(seed);
  std::vector<int> counts(p_target.size(), 0);
  for (int i = 0; i < iterations; ++i)
    counts[SimulateAcceptCorrectStep(p_target, p_draft, rng)]++;
  for (size_t i = 0; i < p_target.size(); ++i)
    EXPECT_NEAR(static_cast<float>(counts[i]) / iterations, p_target[i], tol)
        << "token " << i << ", seed " << seed;
}

}  // namespace

TEST(SpeculativeEquivalenceTest, DraftEqualsTargetPreservesDistribution) {
  std::array<float, 4> p{0.4f, 0.3f, 0.2f, 0.1f};
  ExpectAcceptCorrectMatchesTarget(p, p, /*seed=*/1);
}

TEST(SpeculativeEquivalenceTest, DraftCloseToTargetPreservesDistribution) {
  std::array<float, 4> p_target{0.4f, 0.3f, 0.2f, 0.1f};
  std::array<float, 4> p_draft{0.35f, 0.32f, 0.23f, 0.10f};
  ExpectAcceptCorrectMatchesTarget(p_target, p_draft, /*seed=*/2);
}

TEST(SpeculativeEquivalenceTest, DraftFarFromTargetStillPreservesDistribution) {
  // Draft is nearly the reverse of target -> most proposals are rejected and the
  // correction distribution carries the load. Output must still match the target.
  std::array<float, 4> p_target{0.4f, 0.3f, 0.2f, 0.1f};
  std::array<float, 4> p_draft{0.1f, 0.2f, 0.3f, 0.4f};
  ExpectAcceptCorrectMatchesTarget(p_target, p_draft, /*seed=*/3);
}

TEST(SpeculativeEquivalenceTest, DraftMissesTargetMassStillPreservesDistribution) {
  // Draft assigns zero probability to token 0, which the target prefers, so the
  // only way token 0 is ever emitted is through the correction path.
  std::array<float, 4> p_target{0.5f, 0.2f, 0.2f, 0.1f};
  std::array<float, 4> p_draft{0.0f, 0.4f, 0.4f, 0.2f};
  ExpectAcceptCorrectMatchesTarget(p_target, p_draft, /*seed=*/4);
}

TEST(SpeculativeEquivalenceTest, DraftProposesTargetZeroMassNeverEmitsIt) {
  // Token 3 has zero target probability; it must never be committed even though
  // the draft proposes it a quarter of the time.
  std::array<float, 4> p_target{0.5f, 0.3f, 0.2f, 0.0f};
  std::array<float, 4> p_draft{0.25f, 0.25f, 0.25f, 0.25f};
  ExpectAcceptCorrectMatchesTarget(p_target, p_draft, /*seed=*/5);
}

TEST(SpeculativeEquivalenceTest, StableAcrossManySeeds) {
  std::array<float, 5> p_target{0.30f, 0.25f, 0.20f, 0.15f, 0.10f};
  std::array<float, 5> p_draft{0.10f, 0.15f, 0.20f, 0.25f, 0.30f};
  for (uint32_t seed = 1; seed <= 8; ++seed)
    ExpectAcceptCorrectMatchesTarget(p_target, p_draft, seed, /*tol=*/0.015f,
                                     /*iterations=*/100000);
}

// ---------------------------------------------------------------------------
// Round-level paths. Explicitly cover (1) first rejection early in the K window,
// (2) rejection late in the window, and (3) all-K-accepted bonus token. In every
// case the committed token at the decisive position must follow the target dist.
// ---------------------------------------------------------------------------

namespace {

// Mirror RunRound's accept/correct/bonus walk for a K-position round with fixed
// per-position target/draft distributions; returns the committed token ids.
std::vector<int> SimulateRound(const std::vector<std::vector<float>>& targets,
                               const std::vector<std::vector<float>>& drafts,
                               std::mt19937& rng) {
  const int K = static_cast<int>(targets.size());
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<int> committed;
  for (int i = 0; i < K; ++i) {
    std::discrete_distribution<int> draft_dist(drafts[i].begin(), drafts[i].end());
    const int draft_token = draft_dist(rng);
    if (uniform(rng) < ComputeAcceptProb(targets[i][draft_token], drafts[i][draft_token])) {
      committed.push_back(draft_token);  // accepted, continue the window
      continue;
    }
    std::vector<float> correction(targets[i].size());
    BuildCorrectionDistribution(targets[i], drafts[i], {correction.data(), correction.size()});
    std::discrete_distribution<int> correction_dist(correction.begin(), correction.end());
    committed.push_back(correction_dist(rng));  // first rejection -> correction, round ends
    return committed;
  }
  // All K accepted: append a bonus token from the target's trailing distribution.
  std::discrete_distribution<int> bonus_dist(targets[K - 1].begin(), targets[K - 1].end());
  committed.push_back(bonus_dist(rng));
  return committed;
}

// Histogram the committed token at a fixed position across many rounds and assert
// it matches the target distribution at that position. The constructed scenarios
// guarantee the position is always present, so there is no selection bias.
void ExpectCommittedPositionMatchesTarget(const std::vector<std::vector<float>>& targets,
                                          const std::vector<std::vector<float>>& drafts,
                                          int position, std::span<const float> expected,
                                          uint32_t seed, float tol = 0.015f,
                                          int iterations = 100000) {
  std::mt19937 rng(seed);
  std::vector<int> counts(expected.size(), 0);
  for (int i = 0; i < iterations; ++i) {
    auto committed = SimulateRound(targets, drafts, rng);
    ASSERT_GT(static_cast<int>(committed.size()), position);
    counts[committed[position]]++;
  }
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(static_cast<float>(counts[i]) / iterations, expected[i], tol) << "token " << i;
}

}  // namespace

TEST(SpeculativeRoundPathTest, EarlyRejectionAtPositionZeroMatchesTarget) {
  // Draft at position 0 is far from the target, so the window usually rejects on
  // the very first token. The committed position-0 token must still match target 0.
  std::vector<std::vector<float>> targets{{0.5f, 0.3f, 0.2f}, {0.4f, 0.4f, 0.2f}};
  std::vector<std::vector<float>> drafts{{0.1f, 0.2f, 0.7f}, {0.4f, 0.4f, 0.2f}};
  ExpectCommittedPositionMatchesTarget(targets, drafts, /*position=*/0, targets[0], /*seed=*/11);
}

TEST(SpeculativeRoundPathTest, LateRejectionAtLastPositionMatchesTarget) {
  // Positions 0 and 1 use draft == target (always accepted), so the window always
  // reaches position 2 where the draft diverges. Committed position-2 token must
  // match target 2 whether it is accepted or corrected.
  std::vector<std::vector<float>> targets{
      {0.5f, 0.3f, 0.2f}, {0.3f, 0.4f, 0.3f}, {0.6f, 0.1f, 0.3f}};
  std::vector<std::vector<float>> drafts{
      {0.5f, 0.3f, 0.2f}, {0.3f, 0.4f, 0.3f}, {0.1f, 0.8f, 0.1f}};
  ExpectCommittedPositionMatchesTarget(targets, drafts, /*position=*/2, targets[2], /*seed=*/12);
}

TEST(SpeculativeRoundPathTest, AllAcceptedAppendsBonusFromTarget) {
  // Draft == target at every position, so all K are accepted and a bonus token is
  // drawn from the target's trailing distribution. Round always commits K+1 tokens.
  std::vector<std::vector<float>> targets{
      {0.5f, 0.3f, 0.2f}, {0.3f, 0.4f, 0.3f}, {0.2f, 0.5f, 0.3f}};
  std::vector<std::vector<float>> drafts = targets;
  std::mt19937 rng(13);
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(SimulateRound(targets, drafts, rng).size(), targets.size() + 1);
  // The bonus token (position K) must follow the target's trailing distribution.
  ExpectCommittedPositionMatchesTarget(targets, drafts, /*position=*/3, targets[2], /*seed=*/13);
}

}  // namespace Generators::test
