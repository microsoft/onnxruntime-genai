// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"

namespace Generators {

// Extends the packed variable-length decoder contract with the fixed request-state tensors gathered
// and staged by the current composite cache reservation.
struct HybridDecoderIO : DecoderIO {
  HybridDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                  ScheduledRequests& scheduled_requests,
                  std::shared_ptr<CacheManager> cache_manager,
                  const ExecutionContext& execution_context,
                  size_t position_planes);

  std::vector<DeviceSpan<float>> ProcessLogits() override;
  Tensor* HiddenStates() const override { return varlen_io_.HiddenStates(); }

 private:
  // Takes the context by parameter: this IO is moved into ScheduledRequests and outlives the
  // ExecutionContext that Engine::Step keeps on its stack.
  void BindFixedState(const ExecutionContext& execution_context);

  VarlenDecoderIO varlen_io_;
};

}  // namespace Generators
