// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "span.h"
#include "sampling_distribution.h"

namespace Generators {

// Processed target row used by every speculative verifier. Greedy selection stores only the
// selected token; sampling stores the sparse categorical produced after penalties and truncation.
struct TargetTokenSelection {
  int32_t greedy_token{-1};
  std::vector<int32_t> indices;
  std::vector<float> probs;
};

inline void ComputeTargetTokenSelection(std::span<const float> logits, int current_length,
                                        std::span<const int32_t> prefix, bool greedy,
                                        int top_k, float top_p, float temperature,
                                        LogitsPenaltyProcessor& penalty_processor,
                                        SampledCategorical& sampled,
                                        TargetTokenSelection& selection) {
  const auto processed_logits = penalty_processor.Apply(logits, current_length, prefix);
  if (greedy) {
    selection.greedy_token = static_cast<int32_t>(
        std::max_element(processed_logits.begin(), processed_logits.end()) -
        processed_logits.begin());
    selection.indices.clear();
    selection.probs.clear();
    return;
  }

  ComputeSampledCategorical(processed_logits, top_k, top_p, temperature, sampled);
  selection.greedy_token = -1;
  selection.indices.assign(sampled.indices.begin(), sampled.indices.end());
  selection.probs.assign(sampled.probs.begin(), sampled.probs.end());
}

inline float GetTargetTokenProbability(const TargetTokenSelection& selection, int32_t token) {
  for (size_t i = 0; i < selection.indices.size(); i++) {
    if (selection.indices[i] == token)
      return selection.probs[i];
  }
  return 0.0f;
}

inline int32_t SampleTargetToken(const TargetTokenSelection& selection, std::mt19937& rng) {
  return SampleCategoricalToken(selection.indices, selection.probs, rng);
}

struct DeterministicProposalVerification {
  int accepted_count{};
  int evaluated_count{};
  int32_t final_token{-1};
  bool used_bonus{};
};

inline DeterministicProposalVerification VerifyDeterministicProposal(
    std::span<const int32_t> proposal_tokens,
    const TargetTokenSelection& first_target,
    std::span<const TargetTokenSelection> subsequent_targets,
    std::mt19937& rng,
    std::vector<std::mt19937>* states_after_draw = nullptr) {
  if (proposal_tokens.empty() || subsequent_targets.size() < proposal_tokens.size())
    throw std::invalid_argument(
        "Deterministic proposal verification requires one proposal and next-target row per token.");
  if (states_after_draw) {
    states_after_draw->clear();
    states_after_draw->reserve(proposal_tokens.size() + 1);
  }

  DeterministicProposalVerification result;
  for (size_t i = 0; i < proposal_tokens.size(); i++) {
    const TargetTokenSelection& target = i == 0 ? first_target : subsequent_targets[i - 1];
    result.evaluated_count++;
    result.final_token = SampleTargetToken(target, rng);
    if (states_after_draw)
      states_after_draw->push_back(rng);
    if (result.final_token != proposal_tokens[i])
      return result;
    result.accepted_count++;
  }

  result.final_token = SampleTargetToken(subsequent_targets[proposal_tokens.size() - 1], rng);
  if (states_after_draw)
    states_after_draw->push_back(rng);
  result.used_bonus = true;
  return result;
}

inline std::vector<float>& DensifyTargetTokenSelection(const TargetTokenSelection& selection,
                                                       int vocab_size,
                                                       std::vector<float>& dense) {
  dense.assign(static_cast<size_t>(vocab_size), 0.0f);
  for (size_t i = 0; i < selection.indices.size(); i++)
    dense[static_cast<size_t>(selection.indices[i])] = selection.probs[i];
  return dense;
}

// Inputs: probability vector probs (e.g. draft softmax), top_k, top_p, temperature
// Output: full-vocab sampling distribution with zeros outside the kept set
inline std::vector<float> SamplingDistributionFromProbs(std::span<const float> probs,
                                                        int top_k, float top_p,
                                                        float temperature) {
  std::vector<float> scores(probs.size());
  for (size_t i = 0; i < probs.size(); i++) {
    scores[i] = std::log(std::max(probs[i], std::numeric_limits<float>::min()));
  }
  SampledCategorical c;
  ComputeSampledCategorical({scores.data(), scores.size()}, top_k, top_p, temperature, c);
  return ScatterToFullVocab(c, static_cast<int>(probs.size()));
}

// Inputs: two scalar probabilities p_target, p_draft
// Output: acceptance probability for draft token, scalar in [0, 1]
inline float ComputeAcceptProb(float p_target, float p_draft) {
  // Avoid divide-by-zero
  if (p_draft <= 0.0f) {
    return 1.0f;
  }
  return std::min(1.0f, p_target / p_draft);
}

// Inputs: two distributions p_target, p_draft; output buffer out
// Output: normalized correction distribution written in out
inline void BuildCorrectionDistribution(std::span<const float> p_target,
                                        std::span<const float> p_draft,
                                        std::span<float> out) {
  // Zero out tokens the draft over-represented and accumulate sum for normalization
  float sum = 0.0f;
  for (size_t i = 0; i < out.size(); i++) {
    float diff = p_target[i] - p_draft[i];
    if (diff > 0.0f) {
      out[i] = diff;
    } else {
      out[i] = 0.0f;
    }
    sum += out[i];
  }

  // Normalize to sum to 1
  if (sum > 0.0f) {
    float inv_sum = 1.0f / sum;
    for (auto& v : out) {
      v *= inv_sum;
    }
  } else {
    // Distributions matched exactly (shouldn't happen on a rejection): fall back to target
    for (size_t i = 0; i < out.size(); i++) {
      out[i] = p_target[i];
    }
  }
}

}  // namespace Generators
