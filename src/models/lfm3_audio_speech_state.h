// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "multi_modal.h"

namespace Generators {

// LFM3-Audio's encoder takes one clip per run: [1, num_mels, frames], no length input, and writes
// [tokens, hidden]. The processor stages the clips in one zero-padded tensor, so with several clips this state
// runs the encoder on each clip's own frames and copies its features into the shared buffer in clip order.
struct Lfm3AudioSpeechState : SpeechState {
  Lfm3AudioSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> frames_per_clip_;
  std::vector<int64_t> tokens_per_clip_;
};

}  // namespace Generators
