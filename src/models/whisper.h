#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"

namespace Generators {

struct Whisper_Model : Arch {
  Whisper_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  Model& model_;

  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;

  Ort::Allocator* allocator_device_{};  // Can be CUDA or CPU based on the DeviceType in the model

  std::unique_ptr<OrtSession> session_decoder_; // decoder.onnx
  std::unique_ptr<OrtSession> session_encoder_; // encoder_decoder_init.onnx

  // Model parameters:
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

  std::array<const char*, 2> past_names_{"past_key_self_%d", "past_value_self_%d"};
  std::array<const char*, 2> present_names_{"present_key_self_%d", "present_value_self_%d"};
  std::array<const char*, 2> past_cross_names_{"past_key_cross_%d", "past_value_cross_%d"};
  std::array<const char*, 2> present_cross_names_{"present_key_cross_%d", "present_value_cross_%d"};

 private:
  void InitModelParams();
};

struct Whisper_State : State {
  Whisper_State(Whisper_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  Whisper_Model* model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  InputIDs decoder_input_ids_;
  Logits logits_;
  KV_Cache kv_cache_;
  std::unique_ptr<OrtValue> encoder_hidden_states_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;
};
}
