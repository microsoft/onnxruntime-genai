#pragma once

#include "ort_genai_c.h"
#include "span.h"

namespace Generators {

struct Generator;
class TokenPostProcessor;

class TokenOutputs {
  friend struct Generator;
  friend class TokenPostProcessor;

 private:
  TokenOutputs(std::span<const int32_t> tokens,
               std::span<const OgaTokenMetadataAcousticFrameInterval> intervals)
      : tokens_{tokens}, intervals_{intervals} {}
  std::span<const int32_t> tokens_;
  std::span<const OgaTokenMetadataAcousticFrameInterval> intervals_;
};

}  // namespace Generators