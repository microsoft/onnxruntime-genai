// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "nemo_mel_spectrogram.h"

namespace Generators {

struct SileroVad;

struct CohereProcessor : Processor {
  CohereProcessor(Config& config, const SessionInfo& session_info, Model& model);
  ~CohereProcessor();

  CohereProcessor() = delete;
  CohereProcessor(const CohereProcessor&) = delete;
  CohereProcessor& operator=(const CohereProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  // Silero-VAD-based chunking (used when config has VAD section).
  // Runs the Silero VAD over the full waveform in non-overlapping windows, then
  // converts per-frame speech probabilities into chunks. Each chunk is a list
  // of speech sub-regions (start,end) in sample units; the caller concatenates
  // the sub-regions into a single contiguous buffer, dropping the silence
  // between them. This prevents the model from hallucinating across long
  // intra-chunk silences when short utterances are merged to satisfy
  // min_speech_ms.
  std::vector<std::vector<std::pair<size_t, size_t>>> SplitWaveformByVad(
      const float* samples, size_t num_samples, int sample_rate) const;

  // Compute mel + normalize directly from PCM float32 samples. Returns OrtValue [1, num_mels, num_frames].
  std::pair<std::unique_ptr<OrtValue>, int64_t> ComputeMelFromPCM(const float* samples, size_t num_samples) const;

  ONNXTensorElementDataType audio_features_type_;

  nemo_mel::NemoMelConfig mel_cfg_{};
  float norm_eps_{};

  const Config* config_{};

  // VAD instance — created in the constructor when the config opts in.
  // Mutable because Process() is const but SileroVad mutates internal state.
  mutable std::unique_ptr<SileroVad> vad_;
};

}  // namespace Generators
