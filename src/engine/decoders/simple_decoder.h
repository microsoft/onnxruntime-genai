// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "../../models/decoder_only.h"
#include "../cache_manager.h"

namespace Generators {

struct StaticBatchDecoderIO : ModelIO {
  StaticBatchDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                       ScheduledRequests& scheduled_requests,
                       std::shared_ptr<CacheManager> cache_manager);

 private:
  void PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareAttentionMask(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);

  std::vector<std::unique_ptr<Tensor>> owned_inputs_;
  std::vector<std::unique_ptr<Tensor>> owned_outputs_;
};

struct VarlenDecoderIO : ModelIO {
  VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                  ScheduledRequests& scheduled_requests,
                  std::shared_ptr<CacheManager> cache_manager);

 private:
  void PrepareInputIds();
  void PreparePositionIds();
};

struct SimpleDecoder : public Decoder {
  SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model, std::shared_ptr<CacheManager> cache_manager);

  void Decode(ScheduledRequests& scheduled_requests) override;

 private:
  std::shared_ptr<DecoderOnly_Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
};

}  // namespace Generators
