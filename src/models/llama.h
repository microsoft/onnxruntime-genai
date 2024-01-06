#pragma once
#include "model.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Llama_Model : Arch {
  Llama_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  std::unique_ptr<OrtSession> session_decoder_;

  // Model parameters:
  Model& model_;

  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;

  Ort::Allocator* allocator_device_{};  // Can be CUDA or CPU based on the DeviceType in the model

  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

  std::array<const char*, 2> past_names_{"past_key_values.%d.key", "past_key_values.%d.value"};
  std::array<const char*, 2> present_names_{"present.%d.key", "present.%d.value"};

 private:
  void InitModelParams();
};


struct Llama_State : State {
  Llama_State(Llama_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  Llama_Model* model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  Logits logits_;
  KV_Cache kv_cache_;
  PositionIDs<int64_t> position_ids_;

  std::array<int64_t, 2> input_ids_shape_;
  std::unique_ptr<OrtValue> input_ids_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;
};

}
