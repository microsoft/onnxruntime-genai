// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Flow interpreter for Pipeline-as-Config multi-session pipelines.
//
// The FlowInterpreter is a STATELESS orchestration layer that:
//   1. Partitions flow steps into prompt-phase and decode-phase groups
//   2. Resolves dataflow wiring between sessions
//
// It does NOT store intermediate tensors or manage session execution.
// Intermediate tensor storage is owned by PipelineConfigState (per-State,
// not per-Model) to avoid clobbering when multiple States share a Model.

#pragma once
#include "../pipeline_config_schema.h"
#include "model.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace Generators {

struct FlowInterpreter {
  explicit FlowInterpreter(const PipelineConfig& pipeline_config);

  // Partitioned flow steps.  Populated at construction from pipeline_config.flow[].
  const std::vector<PipelineConfig::FlowStep>& prompt_steps() const { return prompt_steps_; }
  const std::vector<PipelineConfig::FlowStep>& always_steps() const { return always_steps_; }

  // Returns true when the pipeline has more than just a decoder session.
  bool IsMultiSession() const { return is_multi_session_; }

  // For a given target session, look up dataflow wires and resolve inputs
  // from the provided intermediates map.
  // Returns pairs of (input_tensor_name, OrtValue*) to be bound as session inputs.
  std::vector<std::pair<std::string, OrtValue*>> GetWiredInputs(
      const std::string& session_name,
      const std::map<std::string, OrtValue*>& intermediates) const;

  // Access the dataflow wires for inspection/debugging.
  const std::vector<PipelineConfig::DataflowWire>& dataflow() const { return dataflow_; }

  // Access prompt-only session names (sessions that only run during prompt phase).
  const std::set<std::string>& prompt_only_sessions() const { return prompt_only_sessions_; }

 private:
  std::vector<PipelineConfig::FlowStep> prompt_steps_;
  std::vector<PipelineConfig::FlowStep> always_steps_;
  std::vector<PipelineConfig::DataflowWire> dataflow_;
  bool is_multi_session_{false};
  std::set<std::string> prompt_only_sessions_;
};

}  // namespace Generators
