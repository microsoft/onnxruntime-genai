#pragma once

#include "span.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace Generators {

struct TokenSpeakerEvidence {
  double start_probability_sums[8]{};
  double stop_probability_sums[8]{};
  int64_t start_frame_count{};
  int64_t stop_frame_count{};
};

class TokenOutputs {
 public:
  explicit TokenOutputs(std::span<const int32_t> tokens,
                        std::optional<std::span<const std::pair<int64_t, int64_t>>> acoustic_frame_intervals = std::nullopt,
                        std::optional<std::span<const TokenSpeakerEvidence>> speaker_evidence = std::nullopt);
  std::span<const int32_t> GetTokens() const;
  const std::span<const std::pair<int64_t, int64_t>>* GetAcousticFrameIntervals() const;
  const std::span<const TokenSpeakerEvidence>* GetSpeakerEvidence() const;

 private:
  std::span<const int32_t> tokens_;
  std::optional<std::span<const std::pair<int64_t, int64_t>>> acoustic_frame_intervals_;
  std::optional<std::span<const TokenSpeakerEvidence>> speaker_evidence_;
};

}  // namespace Generators