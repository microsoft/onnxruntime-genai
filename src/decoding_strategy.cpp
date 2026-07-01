// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "decoding_strategy.h"

#include <memory>

#include "generators.h"
#include "standard_decoding_strategy.h"
#include "transducer_decoding_strategy.h"
#include "base_speculative_strategy.h"
#include "models/model.h"
#include "models/model_type.h"

namespace Generators {

// Default: no stats. Speculative strategies override.
SpeculativeStats DecodingStrategy::GetStats() const {
  return SpeculativeStats{};
}

// Factory
std::unique_ptr<DecodingStrategy> MakeDecodingStrategy(Generator& generator) {
  const auto& model_type = generator.model_->config_->model.type;
  if (ModelType::IsTransducer(model_type))
    return std::make_unique<TransducerDecodingStrategy>(generator);
  if (ModelType::IsSpeculative(model_type))
    return std::make_unique<BaseSpeculativeStrategy>(generator);
  return std::make_unique<StandardDecodingStrategy>();
}

}  // namespace Generators
