// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/decoder/gemma4_decoder_state.h"
#include "models/multi_modal.h"

namespace Generators {

Gemma4DecoderState::Gemma4DecoderState(const MultiModalLanguageModel& model,
                                       DeviceSpan<int32_t> sequence_lengths,
                                       const GeneratorParams& params)
    : DecoderState(model, sequence_lengths, params) {
  if (!model.config_->model.decoder.inputs.per_layer_inputs.empty()) {
    const auto shape = model.session_info_.GetInputShape(
        model.config_->model.decoder.inputs.per_layer_inputs);
    const int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
    per_layer_inputs_ = std::make_unique<Embeddings>(
        *this, Embeddings::Mode::Input,
        model.config_->model.decoder.inputs.per_layer_inputs, per_layer_dim);
    per_layer_inputs_->Add();
  }
}

void Gemma4DecoderState::UpdateExtraSequenceLength(size_t new_length) {
  if (per_layer_inputs_)
    per_layer_inputs_->UpdateSequenceLength(new_length);
}

void Gemma4DecoderState::UseExtraChunkView(size_t offset, size_t count) {
  if (per_layer_inputs_)
    per_layer_inputs_->UseChunkView(offset, count);
}

void Gemma4DecoderState::RestoreExtraFullView() {
  if (per_layer_inputs_)
    per_layer_inputs_->RestoreFullView();
}

}  // namespace Generators
