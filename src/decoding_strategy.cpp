// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "decoding_strategy.h"

#include <memory>

#include "generators.h"
#include "search.h"
#include "standard_decoding_strategy.h"
#include "transducer_decoding_strategy.h"
#include "base_speculative_strategy.h"
#include "n_gram_decoding_strategy.h"
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
  if (ModelType::IsTransducer(model.type))
    return std::make_unique<TransducerDecodingStrategy>(generator);
  const bool uses_draft_model = ModelType::UsesDraftModelSpeculation(model.type, model.draft.filename);
  const bool uses_ngram = generator.search_->params_->speculative.ngram_size > 0;
  if (uses_draft_model && uses_ngram)
    throw std::runtime_error(
        "N-gram decoding cannot be combined with draft-model speculative decoding.");
  if (uses_draft_model)
    return std::make_unique<BaseSpeculativeStrategy>(generator);
  if (uses_ngram)
    return std::make_unique<NGramDecodingStrategy>(generator);
  return std::make_unique<StandardDecodingStrategy>();
}

}  // namespace Generators
