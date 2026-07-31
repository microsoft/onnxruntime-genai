// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"

namespace Generators {

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model,
                                     bool allow_chunked_prefill)
    : requests_{requests}, model_{model} {
  // Fixes what each request contributes to this step before anything reads UnprocessedTokens():
  // the cache sizing, the decoder inputs and the logits row selection all have to agree on it.
  for (auto& request : requests_) {
    request->ScheduleTokens(allow_chunked_prefill);
  }
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
    auto& request = requests_[request_idx];
    if (request->status_ == RequestStatus::Completed) {
      continue;
    }

    // A request whose prompt is still being chunked ends this step in the middle of its own prompt,
    // so its last logits row predicts a token the prompt already supplies. It selects nothing and
    // only moves its cursor, which is what makes the next step resume where this one stopped.
    if (request->IsChunkComplete()) {
      request->GenerateNextTokens(logits[request_idx]);
    } else {
      request->AdvanceChunk();
    }
  }
}

}  // namespace Generators
