#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct Gpt_Model : Model {
  Gpt_Model(std::unique_ptr<Config> config, std::shared_ptr<OrtEnv> ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct Gpt_State : State {
  Gpt_State(const Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> beam_indices, int current_length);

  const Gpt_Model& model_;
  bool first_run_{true};

  InputIDs input_ids_{model_, *this};
  Logits logits_{model_, *this};
  KV_Cache_Combined kv_cache_{model_, *this};
  PositionInputs position_inputs_;
  ExtraInputs extra_inputs_{model_, *this};
};
}  // namespace Generators
