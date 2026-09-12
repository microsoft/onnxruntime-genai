// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <span>

#include "../config.h"
#include "../decoding/speculative_sampling.h"
#include "../sampling_distribution.h"
#include "../smartptrs.h"

namespace Generators {

struct DraftVerificationTopKRow {
  std::span<const int32_t> tokens;
  std::span<const float> scores;
};

// Applies the target request's ordinary logits policy to speculative verification rows.
// Transaction state, RNG draws, and draft acceptance remain owned by ScheduledRequests.
class DraftVerificationTokenSelector {
 public:
  DraftVerificationTokenSelector(size_t vocab_size, const Config::Search& search,
                                 std::span<const int32_t> eos_token_ids);

  int32_t SelectGreedy(DeviceSpan<float> logits, int32_t raw_argmax,
                       size_t current_length, std::span<const int32_t> prefix);

  TargetTokenSelection BuildSampled(DeviceSpan<float> logits,
                                    DraftVerificationTopKRow raw_topk,
                                    size_t current_length,
                                    std::span<const int32_t> prefix);

 private:
  bool MinLengthMasksEosAt(size_t current_length) const;
  bool ContainsEos(std::span<const int32_t> tokens) const;
  bool RequiresProcessedRow(size_t current_length,
                            std::span<const int32_t> raw_candidates) const;
  std::span<const float> ProcessRow(DeviceSpan<float> logits, size_t current_length,
                                    std::span<const int32_t> prefix);

  const Config::Search& search_;
  std::span<const int32_t> eos_token_ids_;
  LogitsPenaltyProcessor penalty_processor_;
  SampledCategorical sampling_scratch_;
};

}  // namespace Generators
