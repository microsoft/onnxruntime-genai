#include "token_outputs.h"

namespace Generators {

TokenOutputs::TokenOutputs(std::span<const int32_t> tokens,
                           std::optional<std::span<const std::pair<int64_t, int64_t>>> timestamps)
    : tokens_{tokens}, timestamps_{timestamps} {}

std::span<const int32_t> TokenOutputs::GetTokens() const {
  return tokens_;
}

const std::span<const std::pair<int64_t, int64_t>>* TokenOutputs::GetAcousticFrameIntervals() const {
  if (!timestamps_) return nullptr;
  return &*timestamps_;
}

}  // namespace Generators