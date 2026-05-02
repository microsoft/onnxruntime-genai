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
    // Note: "prompt" steps go into prompt_steps_ only.
    // "always" steps go into always_steps_ only — they run on both prompt and decode.
    // "once" steps go into prompt_steps_ only — they run once before generation.
  }

  // Determine if this is a multi-session pipeline
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

void FlowInterpreter::StoreIntermediate(const std::string& session_name,
                                         const std::string& tensor_name,
                                         OrtValue* value) {
  intermediates_[session_name + "." + tensor_name] = value;
}

OrtValue* FlowInterpreter::GetIntermediate(const std::string& session_name,
                                            const std::string& tensor_name) const {
  auto key = session_name + "." + tensor_name;
  auto it = intermediates_.find(key);
  return it != intermediates_.end() ? it->second : nullptr;
}

std::vector<std::pair<std::string, OrtValue*>> FlowInterpreter::GetWiredInputs(
    const std::string& session_name) const {
  std::vector<std::pair<std::string, OrtValue*>> result;
  for (const auto& wire : dataflow_) {
    if (wire.to_session == session_name) {
      auto* value = GetIntermediate(wire.from_session, wire.from_output);
      if (value) {
        result.emplace_back(wire.to_input, value);
      }
    }
  }
  return result;
}

void FlowInterpreter::ClearPromptIntermediates() {
  for (auto it = intermediates_.begin(); it != intermediates_.end();) {
    // Extract session name from key "session.tensor"
    auto dot = it->first.find('.');
    if (dot != std::string::npos) {
      auto session_name = it->first.substr(0, dot);
      if (prompt_only_sessions_.count(session_name)) {
        it = intermediates_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

void FlowInterpreter::ClearAll() {
  intermediates_.clear();
}

}  // namespace Generators
