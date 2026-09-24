// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/embedding/gemma4_embedding_state.h"

#include "models/multi_modal.h"
#include "models/multi_modal_decoder.h"

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

void Gemma4EmbeddingState::SetExtraInputs(int64_t num_images, int64_t num_image_tokens, int64_t num_audio_tokens) {
  EmbeddingState::SetExtraInputs(num_images, num_image_tokens, num_audio_tokens);
  if (!model_.speech_session_ &&
      model_.session_info_.HasInput(model_.config_->model.embedding.inputs.audio_features)) {
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           -1, 0);
    audio_features_->Add();
    audio_features_->AllocateEmptyFeatures();
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

}  // namespace Generators
