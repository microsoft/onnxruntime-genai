// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"

namespace Generators {

struct Whisper_Model : Model {
  Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx
  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx
};

struct Whisper_State : State {
  Whisper_State(const Whisper_Model& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length, bool search_buffers);
  void Finalize() override;

  const Whisper_Model& model_;
  enum struct RunState {
    Encoder_Decoder_Init,
    Decoder_First,
    Decoder,
  } run_state_{RunState::Encoder_Decoder_Init};

  InputIDs decoder_input_ids_{model_, *this};
  Logits logits_{model_, *this};
  KV_Cache kv_cache_{model_, *this};
  Cross_Cache cross_cache_{model_, *this};
  std::unique_ptr<OrtValue> encoder_input_ids_;
  std::unique_ptr<OrtValue> encoder_hidden_states_;

  std::unique_ptr<OrtValue> past_sequence_length_;
  std::unique_ptr<OrtValue> beam_width_;
  std::unique_ptr<OrtValue> cache_indirection_;

  // Temporary hack to have different sized outputs from the encoder that we then expand into the decoder buffers
  std::vector<std::unique_ptr<OrtValue>> init_presents_;  // Hacked sized encoder_decoder_init presents
  std::vector<OrtValue*> presents_;                       // The original present buffers we must resize init_presents_ into after the first run

  std::vector<std::string> output_cross_qk_names_;
  std::vector<std::unique_ptr<OrtValue>> output_cross_qk_;  // { batch_size, num_heads, 1, 1500 }

#if USE_CUDA
  // Buffers for calculating word-level timestamps
  cuda_unique_ptr<float*> cross_qk_ptrs_buffer_;  // To create and hold a reference to the GPU memory so it isn't freed
  gpu_span<float*> output_cross_qk_ptrs_gpu_;     // To use for copying the CPU vector of float* pointers into
#endif
  std::unique_ptr<OrtValue> alignment_heads_;         // { num_alignment_heads, 2 }
  std::unique_ptr<OrtValue> cross_qk_search_buffer_;  // { batch_beam_size, num_alignment_heads, max_length, 1500 }
  std::unique_ptr<OrtValue> cross_qk_final_;          // { batch_size, num_return_sequences, num_alignment_heads, decoded_length, 1500 }

  size_t cache_indirection_index_{~0U};
};
}  // namespace Generators
