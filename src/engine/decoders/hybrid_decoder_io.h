// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"

namespace Generators {

struct HybridDecoderIO : DecoderIO {
  HybridDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                  ScheduledRequests& scheduled_requests,
                  std::shared_ptr<CacheManager> cache_manager,
                  const ExecutionContext& execution_context);

  std::vector<DeviceSpan<float>> ProcessLogits() override;

 private:
  void BindFixedState();

  const ExecutionContext& execution_context_;
  VarlenDecoderIO varlen_io_;
};

}  // namespace Generators
