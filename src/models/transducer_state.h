// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "step_state.h"

namespace Generators {

// Base of the speech-recognition states (RNNT, TDT, streaming encoder-decoder ASR). Generic code uses
// StepState, so other step-driven model types do not need a transducer name.
struct TransducerState : StepState {
  using StepState::StepState;
};

}  // namespace Generators
