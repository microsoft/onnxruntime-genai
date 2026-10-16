// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/speech/gemma4_speech_state.h"

#include <numeric>

#include "models/multi_modal.h"

namespace Generators {

int64_t Gemma4SpeechState::GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs) const {
  const auto& audio_sizes_name = model_.config_->model.speech.inputs.audio_sizes;
  for (const auto& input : extra_inputs) {
    if (input.name == audio_sizes_name) {
      assert(input.tensor->ort_tensor_);
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("Unsupported data type " +
                                 std::to_string(static_cast<int64_t>(info->GetElementType())) +
                                 " for audio_sizes tensor. Only int64 is supported.");
      }
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(data, data + info->GetElementCount(), 0LL);
    }
  }
  return 0;
}

void Gemma4SpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, int64_t num_audio_tokens) {
  num_audio_tokens_ = num_audio_tokens;
  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         params_->BatchBeamSize(), num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());
}

void Gemma4SpeechState::ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) {
  auto& speech_shape = audio_features_->GetShape();
  if (speech_shape.size() == 3) {
    audio_features_->ReshapeFeatures({speech_shape[0] * speech_shape[1], speech_shape[2]});
  }
  SpeechState::ReuseFeaturesBuffer(embedding_features);
}

}  // namespace Generators
