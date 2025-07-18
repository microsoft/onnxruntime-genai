// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"

namespace Generators {

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model)
    : requests_{requests}, model_{model} {
}

std::unique_ptr<OrtRunOptions> ScheduledRequests::RunOptions() {
  return OrtRunOptions::Create();
}

std::shared_ptr<GeneratorParams> ScheduledRequests::Params() {
  if (!params_) {
    params_ = std::make_shared<GeneratorParams>(*model_);
  }
  return params_;
}

void ScheduledRequests::AddDecoderState(std::unique_ptr<DecoderIO> decoder_state) {
  decoder_state_ = std::move(decoder_state);
}

void ScheduledRequests::GenerateNextTokens() {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot generate next tokens without the decoder state.");
  }

  std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
  if (logits.size() != requests_.size()) {
    throw std::runtime_error("Logits size does not match the number of requests.");
  }

  for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
    if (requests_[request_idx]->status_ != RequestStatus::Completed) {
      requests_[request_idx]->GenerateNextTokens(logits[request_idx]);
    }
  }
}

}  // namespace Generators
