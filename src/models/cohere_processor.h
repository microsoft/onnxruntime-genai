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
  // Split waveform into non-overlapping chunks of length `max_audio_clip_s_`,
  // refining each chunk's tail by searching for the lowest-energy point inside
  // the last `boundary_chunk_s_` of the chunk. This places splits at quiet
  // points (between words/sentences) so the model rarely chops mid-word.
  // Mirrors `split_audio_chunks_energy` / `_find_split_point_energy` in
  // CohereLabs/cohere-transcribe-03-2026 (modeling_cohere_asr.py).
  std::vector<std::pair<size_t, size_t>> SplitWaveformIntoChunks(
      const float* samples, size_t num_samples, int sample_rate) const;

  // Compute mel + normalize directly from PCM float32 samples. Returns OrtValue [1, num_mels, num_frames].
  std::pair<std::unique_ptr<OrtValue>, int64_t> ComputeMelFromPCM(const float* samples, size_t num_samples) const;

  ONNXTensorElementDataType audio_features_type_;

  // Mel + normalize config — populated from genai_config.json (model.* fields)
  // at construction time.
  nemo_mel::NemoMelConfig mel_cfg_{};
  float norm_eps_{};

  // Chunking parameters (from genai_config.json model section)
  float max_audio_clip_s_{};
  float boundary_chunk_s_{};
  int min_energy_window_samples_{};
};

}  // namespace Generators
