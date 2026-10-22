#include "token_outputs.h"

namespace Generators {

TokenOutputs::TokenOutputs(std::span<const int32_t> tokens,
                           std::optional<std::span<const AcousticFrameInterval>> timestamps,
                           std::optional<std::span<const TokenSpeakerEvidence>> speaker_evidence)
    : tokens_{tokens}, timestamps_{timestamps}, speaker_evidence_{speaker_evidence} {}

std::span<const int32_t> TokenOutputs::GetTokens() const {
  return tokens_;
}

const std::span<const AcousticFrameInterval>* TokenOutputs::GetAcousticFrameIntervals() const {
  if (!timestamps_) return nullptr;
  return &*timestamps_;
}

const std::span<const TokenSpeakerEvidence>* TokenOutputs::GetSpeakerEvidence() const {
  return speaker_evidence_ ? &*speaker_evidence_ : nullptr;
}

}  // namespace Generators