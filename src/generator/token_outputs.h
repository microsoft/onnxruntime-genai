#pragma once

#include "span.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace Generators {

class TokenOutputs {
 public:
  explicit TokenOutputs(std::span<const int32_t> tokens,
                        std::optional<std::span<const std::pair<int64_t, int64_t>>> acoustic_frame_intervals = std::nullopt);
  std::span<const int32_t> GetTokens() const;
  const std::span<const std::pair<int64_t, int64_t>>* GetAcousticFrameIntervals() const;

 private:
  std::span<const int32_t> tokens_;
  std::optional<std::span<const std::pair<int64_t, int64_t>>> acoustic_frame_intervals_;
};

}  // namespace Generators