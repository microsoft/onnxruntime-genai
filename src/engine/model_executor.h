// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../models/decoder_only.h"
#include "scheduled_requests.h"

namespace Generators {

struct Decoder {
  Decoder() = default;

  virtual void Decode(ScheduledRequests& scheduled_requests) = 0;
};

struct SimpleDecoder : public Decoder {
  SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model);

  void Decode(ScheduledRequests& scheduled_requests) override;

 private:
  std::shared_ptr<DecoderOnly_Model> model_;
};

struct ModelExecutor {
  ModelExecutor(std::shared_ptr<Model> model);

  void Encode(ScheduledRequests& scheduled_requests);

  void Decode(ScheduledRequests& scheduled_requests);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Decoder> decoder_;
};

}  // namespace Generators
