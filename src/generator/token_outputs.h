#pragma once

#include "span.h"

#include <cstdint>
#include <optional>

namespace Generators {

struct AcousticFrameInterval {
  int64_t start;
  int64_t stop;
};

class TokenOutputs {
 public:
  explicit TokenOutputs(std::span<const int32_t> tokens,
                        std::optional<std::span<const AcousticFrameInterval>> timestamps = std::nullopt);
  std::span<const int32_t> GetTokens() const;
  const std::span<const AcousticFrameInterval>* GetAcousticFrameIntervals() const;

 private:
  std::span<const int32_t> tokens_;
  std::optional<std::span<const AcousticFrameInterval>> timestamps_;
};

}  // namespace Generators