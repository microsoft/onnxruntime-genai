#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Llama_Model : Model {
  Llama_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  std::unique_ptr<OrtSession> session_decoder_;

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

  Llama_Model& model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  InputIDs<int64_t> input_ids_{model_, search_params_};
  Logits logits_{model_, search_params_};
  KV_Cache kv_cache_{model_, search_params_, model_.past_names_, model_.present_names_};
  PositionIDs<int64_t> position_ids_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;
};

}
