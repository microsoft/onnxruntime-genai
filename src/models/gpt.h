#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Gpt_Model : Arch {
  Gpt_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  Model& model_;

  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;

  Ort::Allocator* allocator_device_ {}; // Can be CUDA or CPU based on the DeviceType in the model

  std::unique_ptr<OrtSession> session_decoder_;

  // Model parameters:
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

 private:
  void InitModelParams();
};

struct Gpt_State : State {
  Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& search_params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> beam_indices, int current_length);

  const SearchParams& search_params_;
  bool first_run_{true};

  Gpt_Model* model_;

  InputIDs input_ids_;
  Logits logits_;
  KV_Cache_Combined kv_cache_;
  PositionIDs<int32_t> position_ids_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;
};
}
