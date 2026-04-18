// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NemotronStreamingProcessor, Nemotron-specific streaming mel spectrogram extraction.
#pragma once

#include "streaming_processor.h"
#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

/// Nemotron-specific streaming processor that converts raw PCM audio into
/// mel spectrogram tensors for the cache-aware FastConformer encoder.
struct NemotronStreamingProcessor : StreamingProcessor {
  explicit NemotronStreamingProcessor(Model& model);
  ~NemotronStreamingProcessor() override;

  std::unique_ptr<NamedTensors> Process(const float* audio_data, size_t num_samples) override;
  std::unique_ptr<NamedTensors> Flush() override;

  int GetChunkSamples() const { return nemotron_config_.chunk_samples; }
  int GetSampleRate() const { return nemotron_config_.sample_rate; }

 private:
  Model& model_;
  NemotronConfig nemotron_config_;

  // Log-mel feature extraction
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Mel pre-encode cache: ring buffer of last pre_encode_cache_size frames.
  std::vector<float> mel_pre_encode_cache_;
  int cache_pos_{0};

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  std::unique_ptr<OrtValue> BuildMelTensor(const float* audio_chunk, size_t chunk_samples);
};

}  // namespace Generators
