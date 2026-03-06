// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../streaming_asr.h"
#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

/// Encoder State: manages inputs/outputs for the FastConformer encoder session.
struct NemotronEncoderState : State {
  NemotronEncoderState(const NemotronSpeechModel& model, std::shared_ptr<GeneratorParams> params);

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  void SetInputs(OrtValue* audio_signal, OrtValue* length,
                 OrtValue* cache_channel, OrtValue* cache_time, OrtValue* cache_channel_len);

  OrtValue* GetEncoded();
  int64_t GetEncodedLength();
  std::unique_ptr<OrtValue> TakeCacheChannel();
  std::unique_ptr<OrtValue> TakeCacheTime();
  std::unique_ptr<OrtValue> TakeCacheChannelLen();

 private:
  const NemotronSpeechModel& model_;
  std::vector<std::unique_ptr<OrtValue>> owned_outputs_;
};

/// Decoder State: manages inputs/outputs for the LSTM prediction network session.
struct NemotronPredNetState : State {
  NemotronPredNetState(const NemotronSpeechModel& model, std::shared_ptr<GeneratorParams> params);

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  void SetInputs(OrtValue* targets, OrtValue* target_length,
                 OrtValue* lstm_hidden, OrtValue* lstm_cell);

  OrtValue* GetDecoderOutput() { return outputs_[0]; }
  std::unique_ptr<OrtValue> TakeLstmHidden();
  std::unique_ptr<OrtValue> TakeLstmCell();

 private:
  const NemotronSpeechModel& model_;
  std::vector<std::unique_ptr<OrtValue>> owned_outputs_;
};

/// Joiner State: manages inputs/outputs for the joint network session.
struct NemotronJoinerState : State {
  NemotronJoinerState(const NemotronSpeechModel& model, std::shared_ptr<GeneratorParams> params);

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  void SetInputs(OrtValue* encoder_out, OrtValue* decoder_out);

  const float* GetLogitsData();
  int GetLogitsSize();

 private:
  const NemotronSpeechModel& model_;
  std::vector<std::unique_ptr<OrtValue>> owned_outputs_;
};

/// Streaming ASR implementation for NeMo cache-aware FastConformer encoder
/// with RNNT greedy decoding. Routes inference through ORT GenAI State subclasses.
struct NemoStreamingASR : StreamingASR {
  explicit NemoStreamingASR(Model& model);
  ~NemoStreamingASR() override;

  std::string TranscribeChunk(const float* audio_data, size_t num_samples) override;
  std::string Flush() override;
  const std::string& GetTranscript() const override { return full_transcript_; }
  void Reset() override;

 private:
  NemotronSpeechModel& model_;
  NemotronCacheConfig cache_config_;
  std::shared_ptr<GeneratorParams> params_;

  // State subclasses for each session
  std::unique_ptr<NemotronEncoderState> encoder_state_;
  std::unique_ptr<NemotronPredNetState> prednet_state_;
  std::unique_ptr<NemotronJoinerState> joiner_state_;

  // Streaming state
  NemotronEncoderCache encoder_cache_;
  NemotronDecoderState decoder_state_;
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Log-mel feature extraction
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Mel pre-encode cache: ring buffer of last pre_encode_cache_size frames.
  std::vector<float> mel_pre_encode_cache_;
  int cache_pos_{0};

  // Audio accumulation buffer
  std::vector<float> audio_buffer_;

  void LoadVocab();
  std::string TranscribeMelChunk(const std::vector<float>& mel_data, int num_frames);
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);
};

}  // namespace Generators
