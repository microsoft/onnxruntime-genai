// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/embedding/gemma4_embedding_state.h"

#include "models/multi_modal.h"
#include "models/decoder/multi_modal_decoder.h"

namespace Generators {

Gemma4EmbeddingState::Gemma4EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : EmbeddingState(model, params) {
  if (!model_.config_->model.embedding.outputs.per_layer_inputs.empty()) {
    auto shape = model_.session_info_.GetOutputShape(model_.config_->model.embedding.outputs.per_layer_inputs);
    const int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
    per_layer_inputs_ = std::make_unique<Embeddings>(*this, Embeddings::Mode::Output,
                                                     model_.config_->model.embedding.outputs.per_layer_inputs, per_layer_dim);
    per_layer_inputs_->Add();
  }
}

void Gemma4EmbeddingState::ReuseBuffersInDecoder(DecoderState& decoder) {
  EmbeddingState::ReuseBuffersInDecoder(decoder);
  if (per_layer_inputs_) {
    if (auto* decoder_per_layer_inputs = decoder.GetPerLayerInputs()) {
      per_layer_inputs_->ReuseEmbeddingsBuffer(*decoder_per_layer_inputs);
    }
  }
}

DeviceSpan<float> Gemma4EmbeddingState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                            DeviceSpan<int32_t> next_indices) {
  auto result = EmbeddingState::Run(current_length, next_tokens, next_indices);
  if (per_layer_inputs_) per_layer_inputs_->CopyToConsumer();
  return result;
}

}  // namespace Generators
