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
// Inherits from DecoderOnly_Model which loads the decoder session and
// provides session_decoder_.  For multi-session pipelines (VLM, encoder-
// decoder), additional sessions are loaded from pipeline_config.sessions.
//
// For single-session configs, CreateState() returns DecoderOnly_State
// directly — zero code duplication, full decoder parity.
//
// For multi-session configs, CreateState() returns PipelineConfigState
// which uses FlowInterpreter to orchestrate non-decoder sessions around
// a DecoderOnly_State that handles all decoder internals.
struct PipelineConfigModel : DecoderOnly_Model {
  PipelineConfigModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(
      DeviceSpan<int32_t> sequence_lengths,
      const GeneratorParams& params) const override;

  // Additional sessions for multi-session pipelines (vision, embedding, etc.)
  // The decoder session is inherited from DecoderOnly_Model::session_decoder_.
  std::map<std::string, std::unique_ptr<OrtSession>> extra_sessions_;

  // Per-session options for non-decoder sessions (graph capture disabled).
  std::map<std::string, std::unique_ptr<OrtSessionOptions>> non_decoder_session_options_;

  // Flow interpreter constructed from pipeline_config.
  std::unique_ptr<FlowInterpreter> flow_interpreter_;
};

// State for multi-session Pipeline-as-Config model execution.
//
// The decoder session is fully delegated to DecoderOnly_State (KV cache,
// position inputs, logits, sliding window, chunking — all handled).
// Non-decoder sessions (vision, embedding, encoder) are orchestrated by
// FlowInterpreter which manages execution order and dataflow wiring.
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
  // Run a non-decoder session: set up I/O, execute, capture outputs.
  void RunNonDecoderSession(const std::string& session_name);

  // Run a flow step (decoder delegates to decoder_state_, others to RunNonDecoderSession).
  void RunFlowStep(const PipelineConfig::FlowStep& step,
                   int total_length,
                   DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices);

  const PipelineConfigModel& model_;

  // Decoder session is fully delegated to DecoderOnly_State.
  std::unique_ptr<DecoderOnly_State> decoder_state_;

  // Per-session extra inputs for non-decoder sessions.
  struct NonDecoderSessionIO {
    std::vector<const char*> input_names;
    std::vector<OrtValue*> inputs;
  };
  std::map<std::string, NonDecoderSessionIO> non_decoder_io_;

  // Intermediate tensor store for wiring between sessions.
  // Key: "session_name.tensor_name"
  std::map<std::string, std::unique_ptr<OrtValue>> intermediate_store_;

  // Persistent storage for wired input names to avoid dangling c_str() pointers.
  std::vector<std::string> wired_decoder_input_names_;

  // Logits from the last decoder run, saved by RunFlowStep("decoder").
  DeviceSpan<float> last_logits_;

  bool is_prompt_{true};
};

}  // namespace Generators
