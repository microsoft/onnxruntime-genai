// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "nemo_mel_spectrogram.h"

namespace Generators {

// Forward decl — silero_vad.h includes model.h which includes this header,
// so we can't include silero_vad.h directly without a circular include.
struct SileroVad;

struct CohereProcessor : Processor {
  CohereProcessor(Config& config, const SessionInfo& session_info);
  ~CohereProcessor();  // Defined in .cpp where SileroVad is complete.

  CohereProcessor() = delete;
  CohereProcessor(const CohereProcessor&) = delete;
  CohereProcessor& operator=(const CohereProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

  // Optional dependency injection: allows the processor to construct a SileroVad
  // (which needs a non-const Model&) once the owning MultiModalProcessor has the
  // model reference. If the model's genai_config.json has a non-empty `model.vad`
  // section, VAD-based chunking replaces the energy-based splitter.
  void SetModel(Model& model);

 private:
  // Silero-VAD-based chunking (used when SetModel is called and config has VAD section).
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

  // Mel + normalize config — populated from genai_config.json (model.* fields)
  // at construction time.
  nemo_mel::NemoMelConfig mel_cfg_{};
  float norm_eps_{};

  // VAD-based chunking parameters (from genai_config.json model section).
  // Populated in SetModel from model.cohere_vad_* fields.
  int   vad_min_silence_ms_{};
  int   vad_min_speech_ms_{};
  float vad_max_speech_s_{};
  int   vad_speech_pad_ms_{};

  // VAD instance — created lazily in SetModel when the config opts in.
  // Mutable because Process() is const but SileroVad mutates internal state.
  mutable std::unique_ptr<SileroVad> vad_;
};

}  // namespace Generators
