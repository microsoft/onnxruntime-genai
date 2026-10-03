// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/speech/multi_modal_speech.h"

namespace Generators {

struct Phi4MultimodalSpeechState : SpeechState {
  using SpeechState::SpeechState;

  int64_t GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs) const override;
};

}  // namespace Generators
