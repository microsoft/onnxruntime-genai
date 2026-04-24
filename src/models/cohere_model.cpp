// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

namespace Generators {

void CohereEncoderState::ComputeNumFrames(const std::vector<int64_t>& shape) {
  int audio_stride = model_.config_->model.encoder.audio_stride;

  if (audio_stride > 0 && shape.size() == 2) {
    // Raw audio input [batch, samples]
    // Compute encoder output frames: T_enc = samples / audio_stride + 1
    int T_enc = static_cast<int>(shape[1]) / audio_stride + 1;
    // Store as num_frames_ * 2 so that GetNumFrames() / 2 == T_enc (matches CrossCache allocation)
    num_frames_ = T_enc * 2;
  } else if (audio_stride > 0 && shape.size() == 3) {
    // Mel input [batch, n_mels, T_mel]
    // T_enc = (T_mel - 1) / subsampling_factor + 1 (subsampling_factor = 8 for Cohere conformer)
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  } else {
    throw std::runtime_error("Cohere Transcribe requires audio_stride > 0 in the config. "
                             "Got audio_stride=" + std::to_string(audio_stride) +
                             " with input rank=" + std::to_string(shape.size()));
  }
}

void CohereEncoderState::AddModelSpecificInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Add mel_length input if provided
  for (const auto& [name, value] : extra_inputs) {
    if (name == "mel_length") {
      mel_length_ = std::move(reinterpret_cast<Tensor*>(value.get())->ort_tensor_);
      input_names_.push_back("mel_length");
      inputs_.push_back(mel_length_.get());
      break;
    }
  }
}

std::unique_ptr<State> CohereModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  auto encoder = std::make_unique<CohereEncoderState>(*this, params);
  return std::make_unique<WhisperState>(*this, params, sequence_lengths, std::move(encoder));
}

}  // namespace Generators
