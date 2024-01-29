#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_ids.h"

namespace Generators {

struct Phi2_Model : Model {
  Phi2_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct Phi2_State : State {
  Phi2_State(Phi2_Model& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  Phi2_Model& model_;
  bool first_run_{true};

  InputIDs<int32_t> input_ids_{model_, *this};
  PositionIDs<int32_t> position_ids_;
  Logits logits_{model_, *this};
  KV_Cache_Combined kv_cache_{model_, *this};
};

}  // namespace Generators
