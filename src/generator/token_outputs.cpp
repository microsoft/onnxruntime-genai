#include "token_outputs.h"

namespace Generators {

TokenOutputs::TokenOutputs(std::span<const int32_t> tokens,
                           std::optional<std::span<const std::pair<int64_t, int64_t>>> acoustic_frame_intervals)
    : tokens_{tokens}, acoustic_frame_intervals_{acoustic_frame_intervals} {}

std::span<const int32_t> TokenOutputs::GetTokens() const {
  return tokens_;
}

const std::span<const std::pair<int64_t, int64_t>>* TokenOutputs::GetAcousticFrameIntervals() const {
  if (!acoustic_frame_intervals_) return nullptr;
  return &*acoustic_frame_intervals_;
}

}  // namespace Generators