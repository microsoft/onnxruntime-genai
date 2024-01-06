#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Gpt_Model : Model {
  Gpt_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  std::unique_ptr<OrtSession> session_decoder_;

 private:
  void InitModelParams();
};

struct Gpt_State : State {
  Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& search_params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> beam_indices, int current_length);

  Gpt_Model& model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  InputIDs<int32_t> input_ids_{model_, search_params_};
  Logits logits_{model_, search_params_};
  KV_Cache_Combined kv_cache_{model_, search_params_};
  PositionIDs<int32_t> position_ids_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;
};
}
