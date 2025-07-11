// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "audio_features.h"
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"

namespace Generators {

struct WhisperModel : Model {
  WhisperModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // audio_features -> encoder_hidden_states, cross_kv_cache
  std::unique_ptr<OrtSession> session_decoder_;  // input_ids, self_kv_cache, cross_kv_cache -> logits, self_kv_cache

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
};

struct AudioEncoderState : State {
  AudioEncoderState(const WhisperModel& model, const GeneratorParams& params);
  AudioEncoderState(const AudioEncoderState&) = delete;
  AudioEncoderState& operator=(const AudioEncoderState&) = delete;

  void AddCrossCache(std::unique_ptr<CrossCache>& cross_cache) { cross_cache->AddOutputs(*this); }
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  int GetNumFrames() { return num_frames_; }

 private:
  friend struct WhisperState;

  const WhisperModel& model_;

  std::unique_ptr<AudioFeatures> audio_features_;  // { batch_size, num_mels, num_frames }
  std::unique_ptr<OrtValue> hidden_states_;        // { batch_size, num_frames / 2, hidden_size }
  int num_frames_{3000};                           // Whisper uses a default value of 3000
};

struct WhisperDecoderState : State {
  WhisperDecoderState(const WhisperModel& model, const GeneratorParams& params, const int num_frames);
  WhisperDecoderState(const WhisperDecoderState&) = delete;
  WhisperDecoderState& operator=(const WhisperDecoderState&) = delete;

  void AddCrossCache(std::unique_ptr<CrossCache>& cross_cache) { cross_cache->AddInputs(*this); }
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  bool HasPastSequenceLengthInput() { return model_.session_info_.HasInput(model_.config_->model.decoder.inputs.past_sequence_length); }
  bool HasCacheIndirectionInput() { return model_.session_info_.HasInput(model_.config_->model.decoder.inputs.cache_indirection); }
  bool UsesDecoderMaskedMHA() { return HasPastSequenceLengthInput() && HasCacheIndirectionInput(); }

 private:
  // clang-format off
  friend struct WhisperState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool first_update);

  const WhisperModel& model_;

  DefaultInputIDs input_ids_{*this};                        // Model input
  DefaultKeyValueCache kv_cache_{*this};                    // Model input and output
  
  // Inputs for beam search attention
  std::unique_ptr<OrtValue> past_sequence_length_;          // Model input
  std::unique_ptr<OrtValue> cache_indirection_;             // Model input { batch_size, num_beams, max_sequence_length }

  Logits logits_{*this};                                    // Model output
  std::vector<std::unique_ptr<OrtValue>> output_cross_qk_;  // Model output { batch_size, num_heads, sequence_length, num_frames / 2 }

  // Properties about cross attention's QK outputs
  std::vector<std::string> output_cross_qk_names_;          // Formatted names to check if cross attention's QK outputs exist in model
  std::string output_cross_qk_name_;                        // Format for name of cross attention's QK output
  std::array<int64_t, 4> output_cross_qk_shape_;            // Shape of cross attention's QK outputs
  ONNXTensorElementDataType output_cross_qk_type_;          // Type of cross attention's QK outputs

  const int num_frames_{};
  size_t cache_indirection_index_{~0U};
  size_t output_cross_qk_index_{~0U};
  // clang-format on
};

struct WhisperState : State {
  WhisperState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths);
  WhisperState(const WhisperState&) = delete;
  WhisperState& operator=(const WhisperState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  // clang-format off
  void TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches);
  template <typename T> void UpdateCrossQKSearchBuffer(int current_length);
  template <typename T> void FinalizeCrossQK(int current_length);
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length);
  void Finalize(int current_length) override;

  const WhisperModel& model_;
  int prompt_length_{};

  std::unique_ptr<AudioEncoderState> encoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;             // Model output for encoder, constant input for decoder
  std::unique_ptr<WhisperDecoderState> decoder_state_;

  // Temporary buffer for transpoing self attention K caches and cross attention K caches
  std::unique_ptr<OrtValue> transpose_k_cache_buffer_;  // { batch_size, num_heads, num_frames / 2, head_size }

  // To create and hold a reference to the GPU vector of T* pointers
  DeviceSpan<void*> cross_qk_ptrs_;                     // { num_decoder_layers }

  std::unique_ptr<OrtValue> alignment_heads_;           // { num_alignment_heads, 2 }
  std::unique_ptr<OrtValue> cross_qk_search_buffer_;    // { batch_size, num_alignment_heads, max_sequence_length, num_frames / 2 }
  std::unique_ptr<OrtValue> cross_qk_final_;            // { batch_size, num_return_sequences, num_alignment_heads, decoded_length, num_frames / 2 }
  // clang-format on
};
}  // namespace Generators
