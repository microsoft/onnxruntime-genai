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

// Pipeline-as-Config model: loads sessions from pipeline config,
// dispatches based on config version rather than model_type strings.
//
// For v2 configs, the PipelineConfigModel replaces string-based dispatch
// (DecoderOnly_Model, MultiModalLanguageModel, etc.) with a generic
// model that reads its session layout from the config file.
//
// This minimal implementation supports decoder-only models and reuses
// existing components (DefaultKeyValueCache, DefaultPositionInputs,
// Logits) to produce identical output to DecoderOnly_Model.
struct PipelineConfigModel : Model {
  PipelineConfigModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(
      DeviceSpan<int32_t> sequence_lengths,
      const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct PipelineConfigState : State {
  PipelineConfigState(const PipelineConfigModel& model,
                      DeviceSpan<int32_t> sequence_lengths,
                      const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;
  void RewindTo(size_t index) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                           DeviceSpan<int32_t> beam_indices,
                           int total_length);

  const PipelineConfigModel& model_;

  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<PositionInputs> position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
