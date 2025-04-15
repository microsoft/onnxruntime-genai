// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../models/decoder_only.h"
#include "decoders/decoder.h"
#include "scheduled_requests.h"
#include "cache_manager.h"

namespace Generators {

struct ModelExecutor {
  ModelExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  void Encode(ScheduledRequests& scheduled_requests);

  void Decode(ScheduledRequests& scheduled_requests);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Decoder> decoder_;
};

}  // namespace Generators
