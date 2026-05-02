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
    if (step.when == "prompt" || step.when == "once") {
      prompt_steps_.push_back(step);
    }
    if (step.when == "always") {
      always_steps_.push_back(step);
    }
  }

  is_multi_session_ = pipeline_config.sessions.size() > 1;

  // Identify prompt-only sessions (sessions that never appear in always_steps)
  std::set<std::string> always_session_names;
  for (const auto& step : always_steps_) {
    always_session_names.insert(step.run);
  }
  for (const auto& step : prompt_steps_) {
    if (always_session_names.find(step.run) == always_session_names.end()) {
      prompt_only_sessions_.insert(step.run);
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
