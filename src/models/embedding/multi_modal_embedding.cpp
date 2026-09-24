// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/multi_modal.h"
#include "models/embedding/multi_modal_embedding.h"
#include "models/multi_modal_decoder.h"
#include "models/embedding/gemma4_embedding_state.h"

namespace Generators {

EmbeddingState::EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      inputs_embeds_{*this, Embeddings::Mode::Output, model.config_->model.embedding.outputs.embeddings} {
  input_ids_.Add();
  inputs_embeds_.Add();
}

void EmbeddingState::SetExtraInputs(const int64_t num_images, const int64_t num_image_tokens, const int64_t num_audio_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_audio_tokens_ = num_audio_tokens;

  if (model_.vision_session_) {
    image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.image_features,
                                                           num_images, num_image_tokens_);
    image_features_->Add();
  }
  if (model_.speech_session_) {
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           -1, num_audio_tokens_);
    audio_features_->Add();
  }
}

void EmbeddingState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  input_ids_.Update(next_tokens);
  if (model_.vision_session_) image_features_->Update(is_prompt);
  if (audio_features_) audio_features_->Update(is_prompt);
}

DeviceSpan<float> EmbeddingState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.embedding.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.embedding.run_options.value());
  }
  State::Run(*model_.embedding_session_);
  return {};
}

void EmbeddingState::ReuseBuffersInDecoder(DecoderState& decoder) {
  inputs_embeds_.ReuseEmbeddingsBuffer(decoder.GetInputsEmbeds());
}

std::unique_ptr<EmbeddingState> CreateEmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  // Gemma4: the embedding model produces per_layer_inputs alongside inputs_embeds, which the
  // decoder consumes as per-layer conditioning. Dispatch on the config field itself (rather than a
  // model-name literal) so any model whose embedding graph declares this output gets the subclass.
  if (!model.config_->model.embedding.outputs.per_layer_inputs.empty()) {
    return std::make_unique<Gemma4EmbeddingState>(model, params);
  }
  return std::make_unique<EmbeddingState>(model, params);
}

}  // namespace Generators
