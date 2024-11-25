// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct DecoderOnlyPipelineModel : Model {
  DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  DecoderOnlyPipelineModel(const DecoderOnlyPipelineModel&) = delete;
  DecoderOnlyPipelineModel& operator=(const DecoderOnlyPipelineModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::vector<std::unique_ptr<OrtSession>> sessions_;
};

struct IntermediatePipelineState : State {
  IntermediatePipelineState(const DecoderOnlyPipelineModel& model, const GeneratorParams& params,
                            size_t pipeline_state_index);

  IntermediatePipelineState(const IntermediatePipelineState&) = delete;
  IntermediatePipelineState& operator=(const IntermediatePipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  bool HasInput(std::string_view name) const;

  bool HasOutput(std::string_view name) const;

  bool SupportsPrimaryDevice() const;

  size_t id_;

 private:
  const DecoderOnlyPipelineModel& model_;
};

struct DecoderOnlyPipelineState : State {
  DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model, DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  DecoderOnlyPipelineState(const DecoderOnlyPipelineState&) = delete;
  DecoderOnlyPipelineState& operator=(const DecoderOnlyPipelineState&) = delete;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int total_length);

  const DecoderOnlyPipelineModel& model_;
  std::vector<std::unique_ptr<IntermediatePipelineState>> pipeline_states_;

  // Stores all the outputs from the previous pipeline state(s)
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> ortvalue_store_;

  InputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KV_Cache> kv_cache_;
  PositionInputs position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
