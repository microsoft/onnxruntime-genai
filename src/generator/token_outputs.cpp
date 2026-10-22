#include "token_outputs.h"

#include <optional>

namespace Generators {

struct TokenOutputs::Impl {
  std::span<const int32_t> tokens;
  std::optional<std::span<const OgaTokenMetadataAcousticFrameInterval>> timestamps;
};

TokenOutputs::TokenOutputs(std::span<const int32_t> tokens)
    : impl_{std::make_unique<Impl>(Impl{tokens, std::nullopt})} {}

TokenOutputs::~TokenOutputs() = default;
TokenOutputs::TokenOutputs(TokenOutputs&&) noexcept = default;

std::span<const int32_t> TokenOutputs::GetTokens() const {
  return impl_->tokens;
}

const std::span<const OgaTokenMetadataAcousticFrameInterval>* TokenOutputs::GetTimestamps() const {
  if (!impl_->timestamps) return nullptr;
  return &*impl_->timestamps;
}

void TokenOutputs::SetTimestamps(std::span<const OgaTokenMetadataAcousticFrameInterval> intervals) {
  impl_->timestamps = intervals;
}

}  // namespace Generators