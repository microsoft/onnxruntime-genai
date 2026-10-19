// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/speech/multi_modal_speech.h"

namespace Generators {

struct Gemma4SpeechState : SpeechState {
  using SpeechState::SpeechState;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, int64_t num_audio_tokens) override;
  void ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) override;
};

}  // namespace Generators
