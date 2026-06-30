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
