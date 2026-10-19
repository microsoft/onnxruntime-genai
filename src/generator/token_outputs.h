#pragma once

#include "ort_genai_c.h"
#include "span.h"

#include <memory>

namespace Generators {

struct Generator;
class TokenOutputs {
  friend struct Generator;

 public:
  ~TokenOutputs();
  TokenOutputs(TokenOutputs&&) noexcept;
  std::span<const int32_t> GetTokens() const;
  const std::span<const OgaTokenMetadataAcousticFrameInterval>* GetTimestamps() const;

 private:
  explicit TokenOutputs(std::span<const int32_t> tokens);
  void SetTimestamps(std::span<const OgaTokenMetadataAcousticFrameInterval> intervals);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Generators