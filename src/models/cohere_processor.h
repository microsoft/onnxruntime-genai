// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct CohereProcessor : Processor {
  CohereProcessor(Config& config, const SessionInfo& session_info);

  CohereProcessor() = delete;
  CohereProcessor(const CohereProcessor&) = delete;
  CohereProcessor& operator=(const CohereProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  // Split waveform into chunks at quiet energy boundaries.
  std::vector<std::pair<size_t, size_t>> SplitWaveformIntoChunks(
      const float* samples, size_t num_samples, int sample_rate) const;

  // Convert float32 samples to in-memory WAV bytes (16-bit PCM).
  std::vector<uint8_t> SamplesToWavBytes(const float* samples, size_t num_samples, int sample_rate) const;

  // Run mel extraction on WAV bytes, returns (mel_tensor, mel_length).
  std::pair<std::unique_ptr<OrtValue>, int64_t> ExtractMel(const std::vector<uint8_t>& wav_bytes) const;

  ONNXTensorElementDataType audio_features_type_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> processor_;

  // Chunking parameters
  float max_audio_clip_s_{35.0f};
  float overlap_chunk_s_{5.0f};
  int min_energy_window_samples_{1600};
};

}  // namespace Generators
