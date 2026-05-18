// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "flow_interpreter.h"
#include <set>

namespace Generators {

FlowInterpreter::FlowInterpreter(const PipelineConfig& pipeline_config)
    : dataflow_{pipeline_config.dataflow} {
  // Partition flow steps by execution phase
  for (const auto& step : pipeline_config.flow) {
    if (step.when == "init") {
      init_steps_.push_back(step);
    } else if (step.when == "step") {
      step_steps_.push_back(step);
    } else if (step.when == "final") {
      final_steps_.push_back(step);
    }
  }

  is_multi_session_ = pipeline_config.sessions.size() > 1;

  // Identify init-only sessions (sessions that never appear in step_steps or final_steps)
  std::set<std::string> non_init_session_names;
  for (const auto& step : step_steps_) {
    non_init_session_names.insert(step.run);
  }
  for (const auto& step : final_steps_) {
    non_init_session_names.insert(step.run);
  }
  for (const auto& step : init_steps_) {
    if (non_init_session_names.find(step.run) == non_init_session_names.end()) {
      init_only_sessions_.insert(step.run);
    }
  }
}

std::vector<std::pair<std::string, OrtValue*>> FlowInterpreter::GetWiredInputs(
    const std::string& session_name,
    const std::map<std::string, OrtValue*>& intermediates) const {
  std::vector<std::pair<std::string, OrtValue*>> result;
  for (const auto& wire : dataflow_) {
    if (wire.to_session == session_name) {
      auto key = wire.from_session + "." + wire.from_output;
      auto it = intermediates.find(key);
      if (it != intermediates.end() && it->second) {
        result.emplace_back(wire.to_input, it->second);
      }
    }
  }
  return result;
}

}  // namespace Generators
