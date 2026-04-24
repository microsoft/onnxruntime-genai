// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "whisper.h"

namespace Generators {

// Cohere Transcribe encoder: variable-length mel or raw audio with stride-based frame computation
struct CohereEncoderState : AudioEncoderState {
  using AudioEncoderState::AudioEncoderState;

 protected:
  void ComputeNumFrames(const std::vector<int64_t>& shape) override;
  void AddModelSpecificInputs(const std::vector<ExtraInput>& extra_inputs) override;

 private:
  std::unique_ptr<OrtValue> mel_length_;  // { batch_size } — required for Cohere mel encoder
};

// Cohere model: reuses WhisperModel but creates CohereEncoderState
struct CohereModel : WhisperModel {
  using WhisperModel::WhisperModel;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;
};

}  // namespace Generators
