#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct Gpt_Model : Model {
  Gpt_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct Gpt_State : State {
  Gpt_State(const Gpt_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  void RewindTo(size_t index) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length);

  const Gpt_Model& model_;

  InputIDs input_ids_{*this};
  Logits logits_{*this};
  KV_Cache_Combined kv_cache_{*this};
  PositionInputs position_inputs_;
  ExtraInputs extra_inputs_{*this};
};
}  // namespace Generators
