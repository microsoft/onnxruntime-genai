// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "transducer_decoding_strategy.h"

#include "generators.h"
#include "models/transducer_state.h"

namespace Generators {

TransducerDecodingStrategy::TransducerDecodingStrategy(Generator& g)
    : transducer_state_{*dynamic_cast<TransducerState*>(g.state_.get())} {}

void TransducerDecodingStrategy::Step(Generator& g) {
  g.state_->SetExtraInputs(g.extra_inputs_);
  g.extra_inputs_.clear();
  transducer_state_.StepToken();
}

}  // namespace Generators
