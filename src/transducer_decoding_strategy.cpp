// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "transducer_decoding_strategy.h"

#include "generators.h"
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

void TransducerDecodingStrategy::Step(Generator& g) {
  g.state_->SetExtraInputs(g.extra_inputs_);
  g.extra_inputs_.clear();
  transducer_state_.StepToken();
}

}  // namespace Generators
