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

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // audio_features -> encoder_hidden_states, cross_kv_cache
  std::unique_ptr<OrtSession> session_decoder_;  // input_ids, self_kv_cache, cross_kv_cache -> logits, self_kv_cache
};

struct AudioEncoderState : State {
  AudioEncoderState(const WhisperModel& model, const GeneratorParams& params);
  AudioEncoderState(const AudioEncoderState&) = delete;
  AudioEncoderState& operator=(const AudioEncoderState&) = delete;

  void AddCrossCache(std::unique_ptr<Cross_Cache>& cross_cache) { cross_cache->AddOutputs(*this); }
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  int GetNumFrames() { return audio_features_.GetShape()[2]; }

 private:
  friend struct WhisperState;

  const WhisperModel& model_;

  AudioFeatures audio_features_{*this, model_.config_->model.encoder.inputs.audio_features};  // { batch_size, num_mels, num_frames }
  std::unique_ptr<OrtValue> hidden_states_;                                                   // { batch_size, num_frames / 2, hidden_size }
};

struct WhisperDecoderState : State {
  WhisperDecoderState(const WhisperModel& model, const GeneratorParams& params, const int num_frames);
  WhisperDecoderState(const WhisperDecoderState&) = delete;
  WhisperDecoderState& operator=(const WhisperDecoderState&) = delete;
  
  void AddCrossCache(std::unique_ptr<Cross_Cache>& cross_cache) { cross_cache->AddInputs(*this); }
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  friend struct WhisperState;

  void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length, bool first_update);

  const WhisperModel& model_;

  InputIDs input_ids_{*this};                               // Model input
  KV_Cache kv_cache_{*this};                                // Model input and output
  
  // Inputs for beam search attention
  std::unique_ptr<OrtValue> past_sequence_length_;          // Model input
  std::unique_ptr<OrtValue> cache_indirection_;             // Model input { batch_size, num_beams, max_sequence_length }

  Logits logits_{*this};                                    // Model output
  std::vector<std::unique_ptr<OrtValue>> output_cross_qk_;  // Model output { batch_size, num_heads, sequence_length, num_frames / 2 }

  // Properties about cross attention's QK outputs
  std::vector<std::string> output_cross_qk_names_;          // Formatted names to check if cross attention's QK outputs exist in model
  std::array<int64_t, 4> output_cross_qk_shape_;            // Shape of cross attention's QK outputs
  ONNXTensorElementDataType output_cross_qk_type_;          // Type of cross attention's QK outputs

  size_t cache_indirection_index_{~0U};
  size_t output_cross_qk_index_{~0U};
};

struct WhisperState : State {
  WhisperState(const WhisperModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths);
  WhisperState(const WhisperState&) = delete;
  WhisperState& operator=(const WhisperState&) = delete;

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  OrtValue* GetOutput(const char* name) override;

private:
  void TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches);
  template <typename T> void UpdateCrossQKSearchBuffer(int current_length);
  template <typename T> void FinalizeCrossQK();
  void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);
  void Finalize() override;

  const WhisperModel& model_;

  std::unique_ptr<AudioEncoderState> encoder_state_;
  std::unique_ptr<Cross_Cache> cross_cache_;            // Model output for encoder, constant input for decoder
  std::unique_ptr<WhisperDecoderState> decoder_state_;

  // Temporary buffer for transpoing self attention K caches and cross attention K caches
  std::unique_ptr<OrtValue> transpose_k_cache_buffer_;  // { batch_size, num_heads, num_frames / 2, head_size }

#if USE_CUDA
  // Buffers for calculating word-level timestamps
  cuda_unique_ptr<void*> cross_qk_ptrs_buffer_;        // To create and hold a reference to the GPU memory so it isn't freed
  gpu_span<void*> output_cross_qk_ptrs_gpu_;           // To use for copying the CPU vector of float* pointers into
#endif

  std::unique_ptr<OrtValue> alignment_heads_;           // { num_alignment_heads, 2 }
  std::unique_ptr<OrtValue> cross_qk_search_buffer_;    // { batch_size, num_alignment_heads, max_sequence_length, num_frames / 2 }
  std::unique_ptr<OrtValue> cross_qk_final_;            // { batch_size, num_return_sequences, num_alignment_heads, decoded_length, num_frames / 2 }
};
}  // namespace Generators
