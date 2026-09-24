// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model.h"
#include "models/io/multi_modal_features.h"
#include "models/io/extra_inputs.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;

struct SpeechState : State {
  SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  SpeechState(const SpeechState&) = delete;
  SpeechState& operator=(const SpeechState&) = delete;

  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 protected:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_audio_tokens_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> audio_features_;
};

// Returns the total number of audio tokens across the clips in the current batch, read from the
// `audio_sizes_name` extra input (each clip's contribution to the decoder's token sequence).
int64_t GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs, const std::string& audio_sizes_name);

// Factory: pick the right SpeechState subclass based on model type.
std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);

// Validates model-specific device/session placement constraints for the multi-modal pipeline
// (e.g. LFM2-Audio requires every session whose buffers it shares with the decoder to live on a
// device the decoder's session can use). A no-op for models with no such requirement.
void ValidateMultiModalSessionDevices(const Config& config, DeviceType decoder_device, DeviceType inputs_device);

}  // namespace Generators
