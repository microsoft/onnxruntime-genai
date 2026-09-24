// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal.h"
#include "gemma4_multi_modal.h"

namespace Generators {

Gemma4EmbeddingState::Gemma4EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : EmbeddingState(model, params) {
  // Gemma4: embedding model produces per_layer_inputs alongside inputs_embeds
  if (!model_.config_->model.embedding.outputs.per_layer_inputs.empty()) {
    auto shape = model_.session_info_.GetOutputShape(model_.config_->model.embedding.outputs.per_layer_inputs);
    int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
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

Gemma4DecoderState::Gemma4DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : DecoderState(model, sequence_lengths, params) {
  // Gemma4: decoder accepts per_layer_inputs from the embedding model
  if (!model_.config_->model.decoder.inputs.per_layer_inputs.empty()) {
    auto shape = model_.session_info_.GetInputShape(model_.config_->model.decoder.inputs.per_layer_inputs);
    int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
    per_layer_inputs_ = std::make_unique<Embeddings>(*this, Embeddings::Mode::Input,
                                                     model_.config_->model.decoder.inputs.per_layer_inputs, per_layer_dim);
    per_layer_inputs_->Add();
  }
}

void Gemma4DecoderState::UpdateExtraSequenceLength(size_t new_length) {
  if (per_layer_inputs_) per_layer_inputs_->UpdateSequenceLength(new_length);
}

void Gemma4DecoderState::UseExtraChunkView(size_t offset, size_t count) {
  if (per_layer_inputs_) per_layer_inputs_->UseChunkView(offset, count);
}

void Gemma4DecoderState::RestoreExtraFullView() {
  if (per_layer_inputs_) per_layer_inputs_->RestoreFullView();
}

}  // namespace Generators
