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

  // Re-initialize encoder with new audio features for next chunk
  void SetChunkAudioFeatures(std::shared_ptr<Tensor> audio_features, std::shared_ptr<Tensor> mel_length);

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

  // Multi-chunk support
  bool HasMoreChunks() const { return current_chunk_ + 1 < total_chunks_; }
  int CurrentChunk() const { return current_chunk_; }
  int TotalChunks() const { return total_chunks_; }
  bool AdvanceToNextChunk();  // Returns true if advanced, false if no more chunks
  const std::vector<int32_t>& GetPromptTokens() const { return prompt_tokens_; }
  void SetPromptTokens(cpu_span<int32_t> tokens) { prompt_tokens_.assign(tokens.begin(), tokens.end()); }

  // Per-chunk token accumulation for clean text joining
  void SaveChunkTokens(const int32_t* tokens, size_t count);
  const std::vector<std::vector<int32_t>>& GetCompletedChunkTokens() const { return completed_chunk_tokens_; }

  // PyTorch-parity finalization: decode each chunk's tokens, strip ASCII whitespace,
  // filter empty parts, and join with `separator`. Mirrors CohereAsr's
  // `join_chunk_texts(texts, separator=get_chunk_separator(language))` (modeling_cohere_asr.py:1525).
  std::string GetJoinedChunkText(const Tokenizer& tokenizer, const std::string& separator) const;

 private:
  const WhisperModel& model_;

  std::unique_ptr<CohereEncoderState> encoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;
  std::unique_ptr<WhisperDecoderState> decoder_state_;
  std::unique_ptr<OrtValue> transpose_k_cache_buffer_;

  // Multi-chunk state
  int current_chunk_{0};
  int total_chunks_{1};
  std::vector<std::shared_ptr<Tensor>> chunk_mels_;         // Remaining chunk mel tensors
  std::vector<std::shared_ptr<Tensor>> chunk_mel_lengths_;   // Remaining chunk mel_length tensors
  std::vector<int32_t> prompt_tokens_;                       // Saved prompt tokens for re-feeding
  std::vector<std::vector<int32_t>> completed_chunk_tokens_;  // Per-chunk generated tokens (excl prompt/EOS)
};

// Cohere model — inherits WhisperModel, overrides CreateState.
struct CohereModel : WhisperModel {
  using WhisperModel::WhisperModel;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;
};

}  // namespace Generators
