// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "audio_features.h"

namespace Generators {

AudioFeatures::AudioFeatures(State& state, const std::string& name, const std::vector<ExtraInput>& extra_inputs)
    : state_{state},
      model_{state.model_},
      name_{name} {
  // Get audio features
  for (const auto& [input_name, value] : extra_inputs) {
    if (input_name == Config::Defaults::AudioFeaturesName) {
      audio_features_ = model_.ExpandInputs(value->ort_tensor_, state_.params_->search.num_beams);
    }
  }
  if (audio_features_ == nullptr) {
    throw std::runtime_error("audio_features must be provided via SetInputs");
  }

  auto audio_features_info = audio_features_->GetTensorTypeAndShapeInfo();
  shape_ = audio_features_info->GetShape();
  type_ = audio_features_info->GetElementType();
}

void AudioFeatures::Add() {
  index_ = state_.inputs_.size();
  state_.inputs_.push_back(audio_features_.get());
  state_.input_names_.push_back(name_.c_str());
}

}  // namespace Generators
