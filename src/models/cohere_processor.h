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
  // Split waveform into chunks at quiet energy boundaries.
  std::vector<std::pair<size_t, size_t>> SplitWaveformIntoChunks(
      const float* samples, size_t num_samples, int sample_rate) const;

  // Compute mel + normalize directly from PCM float32 samples. Returns OrtValue [1, num_mels, num_frames].
  std::pair<std::unique_ptr<OrtValue>, int64_t> ComputeMelFromPCM(const float* samples, size_t num_samples) const;

  ONNXTensorElementDataType audio_features_type_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> processor_;  // Fallback for non-chunked path

  // Mel config (NeMo-compatible defaults)
  nemo_mel::NemoMelConfig mel_cfg_{128, 512, 160, 400, 16000, 0.97f, 5.96046448e-08f};
  float norm_eps_{1e-5f};

  // Chunking parameters
  float max_audio_clip_s_{35.0f};
  float overlap_chunk_s_{5.0f};
  int min_energy_window_samples_{1600};
};

}  // namespace Generators
