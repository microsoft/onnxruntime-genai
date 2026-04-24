// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "whisper.h"

namespace Generators {

// Cohere Transcribe encoder — standalone State, does NOT inherit AudioEncoderState.
// Handles variable-length mel/raw audio with stride-based frame computation.
struct CohereEncoderState : State {
  CohereEncoderState(const WhisperModel& model, const GeneratorParams& params);
  CohereEncoderState(const CohereEncoderState&) = delete;
  CohereEncoderState& operator=(const CohereEncoderState&) = delete;

  void AddCrossCache(std::unique_ptr<CrossCache>& cross_cache) { cross_cache->AddOutputs(*this); }
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  int GetNumFrames() { return num_frames_; }
  bool HasCrossKVCacheOutputs() { return model_.session_info_.HasOutput(ComposeKeyValueName(model_.config_->model.encoder.outputs.cross_present_key_names, 0)); }
  OrtValue* GetHiddenStates() { return hidden_states_.get(); }

 private:
  const WhisperModel& model_;

  std::unique_ptr<AudioFeatures> audio_features_;
  std::unique_ptr<OrtValue> hidden_states_;
  std::unique_ptr<OrtValue> mel_length_;
  int num_frames_{0};
};

// Cohere orchestrator — standalone State, reuses WhisperDecoderState.
struct CohereState : State {
  CohereState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths);
  CohereState(const CohereState&) = delete;
  CohereState& operator=(const CohereState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const WhisperModel& model_;

  std::unique_ptr<CohereEncoderState> encoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;
  std::unique_ptr<WhisperDecoderState> decoder_state_;
  std::unique_ptr<OrtValue> transpose_k_cache_buffer_;
};

// Cohere model — inherits WhisperModel, overrides CreateState.
struct CohereModel : WhisperModel {
  using WhisperModel::WhisperModel;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;
};

}  // namespace Generators
