// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "decoding_strategy.h"

namespace Generators {

struct Generator;
struct TransducerState;

// TransducerDecodingStrategy
// Drives RNNT / TDT models that bypass the search / logits pipeline entirely.
struct TransducerDecodingStrategy final : DecodingStrategy {
  explicit TransducerDecodingStrategy(Generator& g);
  void Step(Generator& g) override;

 private:
  TransducerState& transducer_state_;
};

}  // namespace Generators
