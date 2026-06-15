// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "span.h"
#include "sampling_distribution.h"

namespace Generators {

// Numerically-stable softmax over a contiguous fp32 row.
inline std::vector<float> Softmax(std::span<const float> logits) {
  if (logits.empty()) return {};
  float mx = *std::max_element(logits.begin(), logits.end());
  std::vector<float> p(logits.size());
  float sum = 0.f;
  for (size_t i = 0; i < logits.size(); i++) {
    p[i] = std::exp(logits[i] - mx);
    sum += p[i];
  }
  if (sum > 0.f)
    for (auto& v : p) v /= sum;
  return p;
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

// Input: distribution probs; rng state
// Output: index of sampled token
inline int SampleFromDistribution(std::span<const float> probs, std::mt19937& rng) {
  // Produce a random float r in [0, 1) and draw one sample
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);

  // Walk the CDF and return first index where running sum exceeds r
  float cum = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    cum += probs[i];
    if (r < cum) return static_cast<int>(i);
  }

  return static_cast<int>(probs.size() - 1);
}

}  // namespace Generators
