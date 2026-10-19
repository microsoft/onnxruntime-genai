// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/speech/multi_modal_speech.h"
#include "models/multi_modal.h"
#include "models/model_type.h"
#include "models/speech/gemma4_speech_state.h"
#include "models/speech/lfm2_audio_speech_state.h"

#include <numeric>

namespace Generators {

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model, model.config_->model.speech.run_options, model.speech_device_},
      model_{model} {}

void SpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  num_audio_tokens_ = num_audio_tokens;

  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         -1, num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());
}

DeviceSpan<float> SpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.speech_session_);
  return {};
}

int64_t SpeechState::GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs) const {
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

void SpeechState::ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) {
  embedding_features.ReuseFeaturesBuffer(*audio_features_);
}

std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (model.config_->model.type == "gemma4" ||
      model.config_->model.type == "gemma4_unified") {
    return std::make_unique<Gemma4SpeechState>(model, params);
  }
  if (ModelType::IsLfm2Audio(model.config_->model.type)) {
    return std::make_unique<Lfm2AudioSpeechState>(model, params);
  }
  return std::make_unique<SpeechState>(model, params);
}

}  // namespace Generators
