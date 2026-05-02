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

  // Update num_frames_ and (re)create hidden_states_ output to match audio_features_'s shape/type.
  void UpdateForCurrentAudio();

  std::unique_ptr<OrtValue> audio_features_;  // device-resident mel for the current chunk
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
  bool AdvanceToNextChunk();  // Returns true if advanced, false if no more chunks
  const std::vector<int32_t>& GetPromptTokens() const { return prompt_tokens_; }
  void SetPromptTokens(cpu_span<int32_t> tokens) { prompt_tokens_.assign(tokens.begin(), tokens.end()); }

  // ----- Streaming (committed) state, all internal -----
  // The Generator uses these to expose an incremental token stream through
  // the standard API (GetSequence + GenerateNextToken). Chunks are emitted by
  // the processor without audio overlap
  void CommitChunkText(const std::vector<int32_t>& chunk_tokens, bool is_final, const Tokenizer& tokenizer);
  size_t StreamedTokensCount() const { return streamed_tokens_count_; }
  void AdvanceStreamedTokensCount() { ++streamed_tokens_count_; }
  const std::vector<int32_t>& CommittedTokens() const { return committed_tokens_; }
  bool FullyDone() const { return fully_done_; }
  void MarkFullyDone() { fully_done_ = true; }
  // Wrap committed_tokens_[0:streamed_tokens_count_] as a CPU-backed DeviceSpan.
  DeviceSpan<int32_t> GetCommittedSpan() const;

 private:
  const WhisperModel& model_;

  // (Re)create decoder_state_ for the current encoder num_frames and wire it
  // to the encoder via either cross-KV cache or encoder_hidden_states.
  void RebuildDecoderForCurrentChunk();

  std::unique_ptr<CohereEncoderState> encoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;
  std::unique_ptr<WhisperDecoderState> decoder_state_;

  // Multi-chunk state
  int current_chunk_{0};
  int total_chunks_{1};
  std::vector<std::shared_ptr<Tensor>> chunk_mels_;         // All chunk mel tensors (indexed by current_chunk_)
  std::vector<std::shared_ptr<Tensor>> chunk_mel_lengths_;   // All chunk mel_length tensors (indexed by current_chunk_)
  std::vector<int32_t> prompt_tokens_;                       // Saved prompt tokens for re-feeding

  // Streaming state
  std::vector<int32_t> committed_tokens_;  // Token sequence visible via GetSequence
  size_t streamed_tokens_count_{0};        // # of committed tokens already yielded by GenerateNextToken
  bool fully_done_{false};                 // True once the last chunk has been committed
};

// Cohere model — inherits WhisperModel, overrides CreateState.
struct CohereModel : WhisperModel {
  using WhisperModel::WhisperModel;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;
};

}  // namespace Generators
