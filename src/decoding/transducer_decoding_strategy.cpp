// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "decoding/transducer_decoding_strategy.h"

#include "generator/generators.h"
#include "models/transducer_state.h"

namespace Generators {

// Validate the downcast before binding the non-null reference member
static TransducerState& RequireTransducerState(Generator& g) {
  auto* transducer_state = dynamic_cast<TransducerState*>(g.state_.get());
  if (!transducer_state) {
    throw std::runtime_error("TransducerDecodingStrategy requires TransducerState");
  }
  return *transducer_state;
}

TransducerDecodingStrategy::TransducerDecodingStrategy(Generator& g)
    : transducer_state_{RequireTransducerState(g)} {}

void TransducerDecodingStrategy::StepImpl(Generator& g) {
  g.state_->SetExtraInputs(g.extra_inputs_);
  g.extra_inputs_.clear();
  transducer_state_.StepToken();
  g.RecordStepTokens(transducer_state_.GetStepTokens());
  if (g.model_->config_->model.timestamp_level != Config::TimestampLevel::Off)
    g.SetStepTimestamps(transducer_state_.GetStepTokenIntervals());
}

}  // namespace Generators
