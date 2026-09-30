// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "decoding/decoding_strategy.h"

#include <memory>

#include "generator/generators.h"
#include "search.h"
#include "decoding/standard_decoding_strategy.h"
#include "decoding/transducer_decoding_strategy.h"
#include "decoding/base_speculative_strategy.h"
#include "decoding/n_gram_decoding_strategy.h"
#include "models/model.h"
#include "models/model_type.h"

namespace Generators {

// Default: no stats. Speculative strategies override.
SpeculativeStats DecodingStrategy::GetStats() const {
  return SpeculativeStats{};
}

// Factory
std::unique_ptr<DecodingStrategy> MakeDecodingStrategy(Generator& generator) {
  const auto& model = generator.model_->config_->model;
  // Transducer (RNNT/TDT) and streaming encoder-decoder ASR (e.g. Moonshine)
  // models drive token emission via TransducerState::StepToken() and never
  // build the search_/logits pipeline, so they must not read generator.search_.
  if (ModelType::IsTransducer(model.type) || ModelType::IsStreamingEncDecASR(model.type))
    return std::make_unique<TransducerDecodingStrategy>(generator);
  const bool uses_ngram = generator.search_->params_->speculative.ngram_size > 0;
  if (model.draft)
    return std::make_unique<BaseSpeculativeStrategy>(generator);
  if (uses_ngram)
    return std::make_unique<NGramDecodingStrategy>(generator);
  return std::make_unique<StandardDecodingStrategy>();
}

}  // namespace Generators
