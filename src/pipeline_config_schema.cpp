// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "pipeline_config_schema.h"
#include <stdexcept>

namespace Generators {

void ValidatePipelineConfig(const PipelineConfig& config) {
  // Validate flow steps reference existing sessions
  for (const auto& step : config.flow) {
    if (config.sessions.find(step.run) == config.sessions.end()) {
      throw std::runtime_error(
          "Pipeline config error: flow step references unknown session '" + step.run + "'");
    }
    if (step.when != "always" && step.when != "prompt" && step.when != "once") {
      throw std::runtime_error(
          "Pipeline config error: invalid 'when' value '" + step.when +
          "' for flow step '" + step.run + "'. Expected 'always', 'prompt', or 'once'.");
    }
    if (!step.loop.empty() && step.loop != "per_image" && step.loop != "batched") {
      throw std::runtime_error(
          "Pipeline config error: invalid 'loop' value '" + step.loop +
          "' for flow step '" + step.run + "'. Expected '', 'per_image', or 'batched'.");
    }
  }

  // Validate dataflow wires reference existing sessions
  for (const auto& wire : config.dataflow) {
    if (config.sessions.find(wire.from_session) == config.sessions.end()) {
      throw std::runtime_error(
          "Pipeline config error: dataflow from_session '" + wire.from_session + "' not found");
    }
    if (config.sessions.find(wire.to_session) == config.sessions.end()) {
      throw std::runtime_error(
          "Pipeline config error: dataflow to_session '" + wire.to_session + "' not found");
    }
    if (wire.from_output.empty()) {
      throw std::runtime_error(
          "Pipeline config error: dataflow from_output is empty for wire from '" +
          wire.from_session + "'");
    }
    if (wire.to_input.empty()) {
      throw std::runtime_error(
          "Pipeline config error: dataflow to_input is empty for wire to '" +
          wire.to_session + "'");
    }
  }
}

}  // namespace Generators
