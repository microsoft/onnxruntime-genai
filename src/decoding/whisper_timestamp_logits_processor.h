// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "span.h"

namespace Generators {

struct Search;

struct WhisperTimestampLogitsConfig {
  int timestamp_begin{};
  int eot_token{};
  std::optional<int> no_timestamps_token;
  std::optional<int> max_initial_timestamp_index;
};

class WhisperTimestampLogitsProcessor {
 public:
  explicit WhisperTimestampLogitsProcessor(WhisperTimestampLogitsConfig config);

  // Applies Whisper's timestamp pairing and probability-mass rules to one logits row.
  // sample_begin identifies the first generated token, excluding the decoder prompt.
  void Apply(std::span<float> logits, std::span<const int32_t> tokens, size_t sample_begin) const;
  void ValidateTokens(std::span<const int32_t> tokens, size_t sample_begin, size_t vocab_size) const;
  const WhisperTimestampLogitsConfig& GetConfig() const { return config_; }

 private:
  WhisperTimestampLogitsConfig config_;
};

void ApplyWhisperTimestampRulesToSearch(Search& search,
                                        const WhisperTimestampLogitsProcessor& processor,
                                        size_t sample_begin);

}  // namespace Generators
