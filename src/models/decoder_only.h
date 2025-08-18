#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct DecoderOnly_Model : Model {
  DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct DecoderOnly_State : State {
  DecoderOnly_State(const DecoderOnly_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  void RewindTo(size_t index) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length);

  const DecoderOnly_Model& model_;

  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  DefaultPositionInputs position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
