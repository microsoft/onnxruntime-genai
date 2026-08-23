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
                  const ExecutionContext& execution_context);

  std::vector<DeviceSpan<float>> ProcessLogits() override;
  Tensor* HiddenStates() const override { return varlen_io_.HiddenStates(); }

 private:
  void BindFixedState();

  const ExecutionContext& execution_context_;
  VarlenDecoderIO varlen_io_;
};

}  // namespace Generators
