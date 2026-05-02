// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "decoder_only.h"
#include "extra_inputs.h"
#include "flow_interpreter.h"

namespace Generators {

// Pipeline-as-Config model: dispatches based on config version rather
// than model_type strings.
//
// For v2 configs, CreateModel() routes here instead of the string-based
// dispatch chain.
//
// For single-session (decoder-only) pipelines, PipelineConfigState directly
// implements the same logic as DecoderOnly_State (same components, same order).
//
// For multi-session pipelines (VLM, encoder-decoder), loads all sessions
// and uses FlowInterpreter to orchestrate execution order and dataflow.
struct PipelineConfigModel : Model {
  PipelineConfigModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(
      DeviceSpan<int32_t> sequence_lengths,
      const GeneratorParams& params) const override;

  // Named sessions loaded from pipeline_config.sessions.
  std::map<std::string, std::unique_ptr<OrtSession>> sessions_;

  // Per-session options for non-decoder sessions (graph capture disabled).
  std::map<std::string, std::unique_ptr<OrtSessionOptions>> non_decoder_session_options_;

  // Flow interpreter constructed from pipeline_config.
  std::unique_ptr<FlowInterpreter> flow_interpreter_;
};

// State for Pipeline-as-Config model execution.
//
// Single-session mode (decoder-only):
//   Behaves identically to DecoderOnly_State — same components, same flow.
//
// Multi-session mode (VLM/encoder-decoder):
//   Non-decoder sessions (vision, embedding, encoder) are driven by
//   FlowInterpreter which manages execution order and intermediate wiring.
//   The decoder session uses existing components (KV cache, position inputs,
//   logits) for full parity with DecoderOnly_State.
struct PipelineConfigState : State {
  PipelineConfigState(const PipelineConfigModel& model,
                      DeviceSpan<int32_t> sequence_lengths,
                      const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;
  void RewindTo(size_t index) override;

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                           DeviceSpan<int32_t> beam_indices,
                           int total_length);

  void RunFlowStep(const PipelineConfig::FlowStep& step,
                   int total_length,
                   DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices);

  // Run a non-decoder session: set up I/O, execute, capture outputs.
  void RunNonDecoderSession(const std::string& session_name);

  const PipelineConfigModel& model_;

  // Decoder session components (reuse existing proven implementations)
  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<PositionInputs> position_inputs_;
  ExtraInputs extra_inputs_{*this};

  // Per-session extra inputs for non-decoder sessions.
  struct NonDecoderSessionIO {
    std::vector<const char*> input_names;
    std::vector<OrtValue*> inputs;
    std::vector<const char*> output_names;
    std::vector<OrtValue*> outputs;
  };
  std::map<std::string, NonDecoderSessionIO> non_decoder_io_;

  // Intermediate tensor store for wiring between sessions.
  // Stores outputs from non-decoder sessions so they can be
  // passed as inputs to downstream sessions.
  // Key: "session_name.tensor_name"
  std::map<std::string, std::unique_ptr<OrtValue>> intermediate_store_;

  // Persistent storage for wired input names to avoid dangling c_str() pointers.
  std::vector<std::string> wired_input_names_;

  bool is_prompt_{true};
};

}  // namespace Generators
