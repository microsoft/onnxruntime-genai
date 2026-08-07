// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Shared base class for streaming transducer models (RNNT, TDT).
//
// Transducer models bypass the standard search/logits pipeline: each call
// to StepToken() advances the decoder by one emitted symbol and appends it
// to the running transcript.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "model.h"

namespace Generators {

struct TransducerState : State {
  using State::State;

  // Advance the decoder by exactly one emitted (non-blank) symbol when
  // possible. Implementations must:
  //   * clear last_tokens_ at entry,
  //   * append every emitted token to both last_tokens_ and all_tokens_,
  //   * set chunk_done_ = true when the current chunk/utterance is fully
  //     consumed.
  virtual void StepToken() = 0;

  bool IsChunkDone() const { return chunk_done_; }
  std::span<const int32_t> GetStepTokens() const { return last_tokens_; }
  std::span<const int32_t> GetAllTokens() const { return all_tokens_; }
  size_t TokenCount() const { return all_tokens_.size(); }

 protected:
  // Full transcript accumulated across all StepToken() calls (and chunks).
  std::vector<int32_t> all_tokens_;
  // Tokens emitted by the most recent StepToken() call.
  std::vector<int32_t> last_tokens_;
  // Set to true when the current chunk has been fully consumed.
  bool chunk_done_{false};
};

}  // namespace Generators
