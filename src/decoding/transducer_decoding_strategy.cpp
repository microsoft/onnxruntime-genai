// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "decoding/transducer_decoding_strategy.h"

#include "generator/generators.h"
#include "models/step_state.h"

namespace Generators {

// Validate the downcast before binding the non-null reference member
static StepState& RequireStepState(Generator& g) {
  auto* transducer_state = dynamic_cast<StepState*>(g.state_.get());
  if (!transducer_state) {
    throw std::runtime_error("TransducerDecodingStrategy requires a StepState");
  }
  return *transducer_state;
}

TransducerDecodingStrategy::TransducerDecodingStrategy(Generator& g)
    : transducer_state_{RequireStepState(g)} {}

void TransducerDecodingStrategy::Step(Generator& g) {
  // Take the inputs first, so inputs that the state rejects are not passed again on the next step.
  const auto extra_inputs = std::move(g.extra_inputs_);
  g.extra_inputs_.clear();
  g.state_->SetExtraInputs(extra_inputs);
  transducer_state_.StepToken();
}

}  // namespace Generators
