// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal_speech.h"
#include "multi_modal.h"
#include "models/model_type.h"
#include "models/speech/lfm2_audio_speech_state.h"

#include <numeric>

namespace Generators {

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void SpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  num_audio_tokens_ = num_audio_tokens;

  // Allocate 3D [batch, num_audio_tokens, hidden_size] matching the speech ONNX model's
  // output rank. Will be reshaped to 2D before passing to the embedding model.
  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         params_->BatchBeamSize(), num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());
}

DeviceSpan<float> SpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.speech.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.speech.run_options.value());
  }
  State::Run(*model_.speech_session_);
  return {};
}

int64_t GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs, const std::string& audio_sizes_name) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == audio_sizes_name) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      auto type_and_shape_info = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto element_count = type_and_shape_info->GetElementCount();
      if (type_and_shape_info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* audio_sizes_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
        return std::accumulate(audio_sizes_data, audio_sizes_data + element_count, 0LL);
      } else {
        throw std::runtime_error("Unsupported data type " + std::to_string(static_cast<int64_t>(type_and_shape_info->GetElementType())) + " for audio_sizes tensor. Only int64 is supported.");
      }
    }
  }

  return 0;
}

std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (ModelType::IsLfm2Audio(model.config_->model.type)) {
    return std::make_unique<Lfm2AudioSpeechState>(model, params);
  }
  return std::make_unique<SpeechState>(model, params);
}

void ValidateMultiModalSessionDevices(const Config& config, DeviceType decoder_device, DeviceType inputs_device) {
  if (ModelType::IsLfm2Audio(config.model.type)) {
    CheckLfm2AudioSessionDevices(config, decoder_device, inputs_device, /*with_audio=*/false);
  }
}

}  // namespace Generators
