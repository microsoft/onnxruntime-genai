// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "batch_asr.h"
#include "nemo_mel_spectrogram.h"
#include "models/nemotron_speech.h"

namespace Generators {

/// Batch (offline) ASR implementation for NeMo cache-aware FastConformer
/// encoder with RNNT greedy decoding.
///
/// Computes mel spectrogram on the full audio at once (avoiding chunk-boundary
/// artifacts), then feeds it through the encoder in config-sized chunks (required
/// by the ONNX export which truncates output per call). Each Transcribe() call
/// is independent — no state carried between calls.
struct NemoBatchASR : BatchASR {
  explicit NemoBatchASR(Model& model);
  ~NemoBatchASR() override;

  std::string Transcribe(const float* audio_data, size_t num_samples) override;

 private:
  Model& model_;
  NemotronCacheConfig cache_config_;

  OrtSession* encoder_session_{};
  OrtSession* decoder_session_{};
  OrtSession* joiner_session_{};

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  void LoadVocab();

  /// Run RNNT greedy decoding on the full encoder output.
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len,
                             NemotronDecoderState& decoder_state);
};

}  // namespace Generators
