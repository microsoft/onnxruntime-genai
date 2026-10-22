#include "token_outputs.h"

namespace Generators {

TokenOutputs::TokenOutputs(std::span<const int32_t> tokens,
                           std::optional<std::span<const AcousticFrameInterval>> timestamps)
    : tokens_{tokens}, timestamps_{timestamps} {}

std::span<const int32_t> TokenOutputs::GetTokens() const {
  return tokens_;
}

const std::span<const AcousticFrameInterval>* TokenOutputs::GetTimestamps() const {
  if (!timestamps_) return nullptr;
  return &*timestamps_;
}

}  // namespace Generators