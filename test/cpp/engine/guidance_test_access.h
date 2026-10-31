// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include "constrained_logits_processor.h"
#include "engine/request.h"

namespace Generators {
namespace test {

#if USE_GUIDANCE
struct GuidanceProcessorTestAccess {
  static void InstallFailedMaskFuture(GuidanceLogitsProcessor& processor) {
    std::promise<std::vector<uint32_t>> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Injected real guidance future failure.")));
    processor.pending_masks_ = promise.get_future().share();
    processor.mask_dirty_ = false;
  }

  static bool MaskDirty(const GuidanceLogitsProcessor& processor) {
    return processor.mask_dirty_;
  }

  static bool MaskAllowsOnlyTokens(
      std::span<const uint32_t> mask, size_t vocab_size,
      std::span<const int> tokens) {
    return GuidanceLogitsProcessor::MaskAllowsOnlyTokens(
        mask, vocab_size, tokens);
  }
};
#endif

struct RequestGuidanceTestAccess {
  static void Install(
      Request& request,
      std::unique_ptr<ConstrainedLogitsProcessor> processor) {
    request.guidance_logits_processor_ = std::move(processor);
  }

  static ConstrainedLogitsProcessor* Get(Request& request) {
    return request.guidance_logits_processor_.get();
  }
};

}  // namespace test
}  // namespace Generators
