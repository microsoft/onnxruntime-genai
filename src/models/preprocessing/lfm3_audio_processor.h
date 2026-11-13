// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// The speech placeholder of an answer already spoken, which only the speech output path can fill.
inline constexpr char kLfm3AudioOutMarker[] = "<|audio_out|>";

// Log-mel front end of LFM3-Audio. The window is periodic Hann of fft_size and the rest of the formula is fixed, so
// the builder refuses a checkpoint whose feature extractor differs.
struct Lfm3AudioMelConfig {
  int num_mels{};
  int fft_size{};
  int hop_length{};
  int sample_rate{};
};

// Right-padded to whole hops and to at least one window, with no centering.
int64_t Lfm3AudioNumMelFrames(int64_t num_samples, const Lfm3AudioMelConfig& config);

// log10 of the Slaney mel power clamped at -12, then (x + 4) / 4, mel-major [num_mels, num_frames].
std::vector<float> ComputeLfm3AudioMel(const float* pcm, int64_t num_samples, const Lfm3AudioMelConfig& config,
                                       int64_t& num_frames);

// Stages the clips of one prompt in a zero-padded [num_clips, num_mels, longest] tensor with their frame and token
// counts. Lfm3AudioSpeechState runs the encoder on each clip's own frames.
struct Lfm3AudioProcessor : Processor {
  Lfm3AudioProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ONNXTensorElementDataType mel_type_;
  Lfm3AudioMelConfig mel_config_;
  int64_t subsampling_factor_{};
  int32_t audio_token_id_{};
};

}  // namespace Generators
