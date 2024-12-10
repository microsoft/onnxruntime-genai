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
  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_.get(); };

  void RewindTo(size_t index) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length);

  const DecoderOnly_Model& model_;
  CapturedGraphInfoPtr captured_graph_info_;

  InputIDs input_ids_{*this};
  Logits logits_{*this};
  KV_Cache kv_cache_{*this};
  PositionInputs position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
