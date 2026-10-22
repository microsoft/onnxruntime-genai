#pragma once

#include "span.h"

#include <cstdint>
#include <optional>

namespace Generators {

struct AcousticFrameInterval {
  int64_t start;
  int64_t stop;
};

struct TokenSpeakerEvidence {
  double start_probability_sums[8]{};
  double stop_probability_sums[8]{};
  int64_t start_frame_count{};
  int64_t stop_frame_count{};
};

class TokenOutputs {
 public:
  explicit TokenOutputs(std::span<const int32_t> tokens,
                        std::optional<std::span<const AcousticFrameInterval>> timestamps = std::nullopt,
                        std::optional<std::span<const TokenSpeakerEvidence>> speaker_evidence = std::nullopt);
  std::span<const int32_t> GetTokens() const;
  const std::span<const AcousticFrameInterval>* GetAcousticFrameIntervals() const;
  const std::span<const TokenSpeakerEvidence>* GetSpeakerEvidence() const;

 private:
  std::span<const int32_t> tokens_;
  std::optional<std::span<const AcousticFrameInterval>> timestamps_;
  std::optional<std::span<const TokenSpeakerEvidence>> speaker_evidence_;
};

}  // namespace Generators