// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "draft_verification.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace Generators {

DraftVerificationTokenSelector::DraftVerificationTokenSelector(
    size_t vocab_size, const Config::Search& search,
    std::span<const int32_t> eos_token_ids)
    : search_{search},
      eos_token_ids_{eos_token_ids},
      penalty_processor_{static_cast<int>(vocab_size), search.repetition_penalty,
                         search.min_length, search.no_repeat_ngram_size,
                         eos_token_ids} {}

bool DraftVerificationTokenSelector::MinLengthMasksEosAt(
    size_t current_length) const {
  return search_.min_length > 0 &&
         current_length < static_cast<size_t>(search_.min_length);
}

bool DraftVerificationTokenSelector::ContainsEos(
    std::span<const int32_t> tokens) const {
  return std::any_of(tokens.begin(), tokens.end(), [this](int32_t token) {
    return std::find(eos_token_ids_.begin(), eos_token_ids_.end(), token) !=
           eos_token_ids_.end();
  });
}

bool DraftVerificationTokenSelector::RequiresProcessedRow(
    size_t current_length, std::span<const int32_t> raw_candidates) const {
  if (search_.repetition_penalty != 1.0f ||
      search_.no_repeat_ngram_size > 0) {
    return true;
  }
  return MinLengthMasksEosAt(current_length) &&
         (raw_candidates.empty() || ContainsEos(raw_candidates));
}

std::span<const float> DraftVerificationTokenSelector::ProcessRow(
    DeviceSpan<float> logits, size_t current_length,
    std::span<const int32_t> prefix) {
  const auto cpu_logits = logits.CopyDeviceToCpu();
  return penalty_processor_.Apply(cpu_logits, static_cast<int>(current_length),
                                  prefix);
}

int32_t DraftVerificationTokenSelector::SelectGreedy(
    DeviceSpan<float> logits, int32_t raw_argmax, size_t current_length,
    std::span<const int32_t> prefix) {
  const std::array raw_candidate{raw_argmax};
  if (!RequiresProcessedRow(current_length, raw_candidate)) {
    return raw_argmax;
  }

  const auto processed = ProcessRow(logits, current_length, prefix);
  return static_cast<int32_t>(
      std::max_element(processed.begin(), processed.end()) - processed.begin());
}

TargetTokenSelection DraftVerificationTokenSelector::BuildSampled(
    DeviceSpan<float> logits, DraftVerificationTopKRow raw_topk,
    size_t current_length, std::span<const int32_t> prefix) {
  TargetTokenSelection selection;
  if (raw_topk.tokens.empty() ||
      RequiresProcessedRow(current_length, raw_topk.tokens)) {
    const auto processed = ProcessRow(logits, current_length, prefix);
    ComputeSampledCategorical(processed, search_.top_k, search_.top_p,
                              search_.temperature, sampling_scratch_);
    selection.indices = sampling_scratch_.indices;
    selection.probs = sampling_scratch_.probs;
    return selection;
  }

  const int k = std::min(search_.top_k,
                         static_cast<int>(raw_topk.tokens.size()));
  const float max_score = raw_topk.scores.front();
  const float inverse_temperature = 1.0f / search_.temperature;
  std::vector<float> probabilities(static_cast<size_t>(k));
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probabilities[static_cast<size_t>(i)] =
        std::exp((raw_topk.scores[static_cast<size_t>(i)] - max_score) *
                 inverse_temperature);
    sum += probabilities[static_cast<size_t>(i)];
  }
  for (float& probability : probabilities) {
    probability /= sum;
  }

  int keep = k;
  if (search_.top_p > 0.0f && search_.top_p < 1.0f) {
    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
      cumulative += probabilities[static_cast<size_t>(i)];
      if (cumulative >= search_.top_p) {
        keep = i + 1;
        break;
      }
    }
  }

  float kept_sum = 0.0f;
  for (int i = 0; i < keep; ++i) {
    kept_sum += probabilities[static_cast<size_t>(i)];
  }
  selection.indices.assign(raw_topk.tokens.begin(),
                           raw_topk.tokens.begin() + keep);
  selection.probs.assign(probabilities.begin(), probabilities.begin() + keep);
  for (float& probability : selection.probs) {
    probability /= kept_sum;
  }
  return selection;
}

}  // namespace Generators
