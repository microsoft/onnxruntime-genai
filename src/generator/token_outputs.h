#pragma once

#include "ort_genai_c.h"
#include "span.h"

#include <memory>

namespace Generators {

struct Generator;
struct TokenSpeakerEvidence {
  double start_probability_sums[8]{};
  double stop_probability_sums[8]{};
  int64_t start_frame_count{};
  int64_t stop_frame_count{};
};

class TokenOutputs {
  friend struct Generator;

 public:
  ~TokenOutputs();
  TokenOutputs(TokenOutputs&&) noexcept;
  std::span<const int32_t> GetTokens() const;
  const std::span<const OgaTokenMetadataAcousticFrameInterval>* GetTimestamps() const;
  const std::span<const TokenSpeakerEvidence>* GetSpeakerEvidence() const;

 private:
  explicit TokenOutputs(std::span<const int32_t> tokens);
  void SetTimestamps(std::span<const OgaTokenMetadataAcousticFrameInterval> intervals);
  void SetSpeakerEvidence(std::span<const TokenSpeakerEvidence> evidence);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Generators