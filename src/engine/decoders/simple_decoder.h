// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"

namespace Generators {

struct SimpleDecoder : public Decoder {
  SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model, std::shared_ptr<CacheManager> cache_manager);

  void Decode(ScheduledRequests& scheduled_requests,
              ExecutionContext& context) override;

 private:
  std::shared_ptr<DecoderOnly_Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  bool has_fixed_state_groups_{};
  size_t position_planes_{};
  // Allocated only when the model asked for CUDA graphs. Owning it here, rather than in the per-step
  // decoder IO, is what gives the captured graph stable buffer addresses to replay against.
  std::unique_ptr<VarlenGraphBuffers> graph_buffers_;
};

}  // namespace Generators
