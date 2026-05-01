// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "nemo_mel_spectrogram.h"

namespace Generators {

struct CohereProcessor : Processor {
  CohereProcessor(Config& config, const SessionInfo& session_info);

  CohereProcessor() = delete;
  CohereProcessor(const CohereProcessor&) = delete;
  CohereProcessor& operator=(const CohereProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  // Split waveform into non-overlapping chunks of at most max_audio_clip_s_
  // seconds each, with cut points snapped to the quietest 100ms window inside
  // the last boundary_chunk_s_ seconds (energy-based silence boundaries).
  // Mirrors exactly CohereAsrFeatureExtractor._split_audio_chunks_energy.
  std::vector<std::pair<size_t, size_t>> SplitWaveformIntoChunks(
      const float* samples, size_t num_samples, int sample_rate) const;

  // Compute mel + normalize directly from PCM float32 samples. Returns OrtValue [1, num_mels, num_frames].
  std::pair<std::unique_ptr<OrtValue>, int64_t> ComputeMelFromPCM(const float* samples, size_t num_samples) const;

  ONNXTensorElementDataType audio_features_type_;

  // Mel + normalize config — populated from genai_config.json (model.* fields)
  // at construction time. Defaults match Cohere Transcribe's published preset.
  nemo_mel::NemoMelConfig mel_cfg_{128, 512, 160, 400, 16000, 0.97f, 5.96046448e-08f};
  float norm_eps_{1e-5f};

  // Chunking parameters (from genai_config.json model section)
  float max_audio_clip_s_{};
  float boundary_chunk_s_{};
};

}  // namespace Generators
