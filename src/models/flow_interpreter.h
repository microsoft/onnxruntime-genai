// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Flow interpreter for Pipeline-as-Config multi-session pipelines.
//
// The FlowInterpreter is a thin orchestration layer that:
//   1. Partitions flow steps into prompt-phase and decode-phase groups
//   2. Stores intermediate tensors produced by non-decoder sessions
//   3. Wires intermediate tensors between sessions based on dataflow config
//
// It does NOT own session execution or manage decoder-specific state (KV cache,
// position inputs, logits).  Each session runs through its own State object.

#pragma once
#include "../pipeline_config_schema.h"
#include "model.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Generators {

struct FlowInterpreter {
  explicit FlowInterpreter(const PipelineConfig& pipeline_config);

  // Partitioned flow steps.  Populated at construction from pipeline_config.flow[].
  //
  // prompt_steps: steps with when=="prompt" or when=="once" — run only on first invocation.
  // always_steps: steps with when=="always" — run every invocation (prompt + decode).
  //
  // On the prompt invocation, execution order is: prompt_steps then always_steps.
  // On decode invocations, only always_steps run.
  const std::vector<PipelineConfig::FlowStep>& prompt_steps() const { return prompt_steps_; }
  const std::vector<PipelineConfig::FlowStep>& always_steps() const { return always_steps_; }

  // Returns true when the pipeline has more than just a decoder session.
  bool IsMultiSession() const { return is_multi_session_; }

  // Store an intermediate tensor produced by a session.
  // Key format: "session_name.tensor_name"
  void StoreIntermediate(const std::string& session_name,
                         const std::string& tensor_name,
                         OrtValue* value);

  // Look up an intermediate tensor.  Returns nullptr if not found.
  OrtValue* GetIntermediate(const std::string& session_name,
                            const std::string& tensor_name) const;

  // For a given target session, collect any wired inputs from intermediates.
  // Returns pairs of (input_tensor_name, OrtValue*) to be bound as session inputs.
  std::vector<std::pair<std::string, OrtValue*>> GetWiredInputs(
      const std::string& session_name) const;

  // Clear all prompt-only intermediates (called after prompt phase completes).
  void ClearPromptIntermediates();

  // Clear all intermediates (called on reset).
  void ClearAll();

  // Access the dataflow wires for inspection/debugging.
  const std::vector<PipelineConfig::DataflowWire>& dataflow() const { return dataflow_; }

 private:
  std::vector<PipelineConfig::FlowStep> prompt_steps_;
  std::vector<PipelineConfig::FlowStep> always_steps_;
  std::vector<PipelineConfig::DataflowWire> dataflow_;
  bool is_multi_session_{false};

  // Set of session names that only run during prompt phase.
  // Their intermediates are cleared after the prompt completes.
  std::set<std::string> prompt_only_sessions_;

  // Intermediate tensors keyed by "session_name.tensor_name".
  // These are non-owning pointers — the OrtValue lifetime is managed by
  // the session's output buffer (State::outputs_).
  std::map<std::string, OrtValue*> intermediates_;
};

}  // namespace Generators
