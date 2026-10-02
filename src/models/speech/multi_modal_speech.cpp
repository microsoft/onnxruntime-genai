// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/speech/multi_modal_speech.h"
#include "models/multi_modal.h"
#include "models/model_type.h"
#include "models/speech/gemma4_speech_state.h"
#include "models/speech/lfm2_audio_speech_state.h"
#include "models/speech/phi4_multimodal_speech_state.h"

namespace Generators {

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model, model.speech_device_},
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
  if (model_.config_->model.speech.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.speech.run_options.value());
  }
  State::Run(*model_.speech_session_);
  return {};
}

int64_t SpeechState::GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs) const {
  return 0;
}

void SpeechState::ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) {
  embedding_features.ReuseFeaturesBuffer(*audio_features_);
}

std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (model.config_->model.type == "gemma4") {
    return std::make_unique<Gemma4SpeechState>(model, params);
  }
  if (model.config_->model.type == "phi4mm") {
    return std::make_unique<Phi4MultimodalSpeechState>(model, params);
  }
  if (ModelType::IsLfm2Audio(model.config_->model.type)) {
    return std::make_unique<Lfm2AudioSpeechState>(model, params);
  }
  return std::make_unique<SpeechState>(model, params);
}

}  // namespace Generators
