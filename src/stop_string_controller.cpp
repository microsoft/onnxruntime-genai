// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stop_string_controller.h"

#include <stdexcept>

#include "models/preprocessing/genai_tokenizer.h"

namespace Generators {

StopStringController::StopStringController(std::shared_ptr<const Tokenizer> tokenizer,
                                           std::vector<std::string> stop_strings)
    : tokenizer_{std::move(tokenizer)},
      matcher_{std::move(stop_strings)} {
  if (!tokenizer_)
    throw std::runtime_error("StopStringController requires a tokenizer.");
  RebuildStream();
}

StopStringController::~StopStringController() = default;

const std::optional<StopStringMatch>& StopStringController::ObserveToken(int32_t token) {
  turn_tokens_.push_back(token);
  // Consume() is a no-op after a match, but the decode is not: the stream has to stay aligned with
  // the raw token history so a later rollback replay reproduces the same bytes.
  DecodeIntoMatcher(token);
  // The Engine publishes raw tokens, not decoded text, so nothing ever reads the safe-output
  // buffer. Drain and discard it here so it cannot grow across a whole generated turn.
  matcher_.TakeSafeOutput();
  return matcher_.Match();
}

void StopStringController::RollbackTo(size_t committed_token_count) {
  if (committed_token_count > turn_tokens_.size())
    throw std::logic_error("Stop-string rollback checkpoint exceeds staged token history.");

  // Nothing was staged since the checkpoint (e.g. a failed new-turn admission that never observed a
  // token through this controller): the stream and matcher are already at the checkpoint, so skip
  // rebuilding the stream and replaying tokens that are already reflected in the current state.
  if (committed_token_count == turn_tokens_.size())
    return;

  turn_tokens_.resize(committed_token_count);
  matcher_.Reset();
  RebuildStream();
  for (int32_t token : turn_tokens_) {
    DecodeIntoMatcher(token);
    matcher_.TakeSafeOutput();
  }
}

void StopStringController::RebuildStream() {
  // The detokenizer cache is not cloneable, so construction and every rollback recreate the stream.
  stream_ = tokenizer_->CreateStream();
}

void StopStringController::DecodeIntoMatcher(int32_t token) {
  matcher_.Consume(stream_->Decode(token));
}

}  // namespace Generators
