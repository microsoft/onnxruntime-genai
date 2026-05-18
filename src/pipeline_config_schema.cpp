// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "pipeline_config_schema.h"
#include <stdexcept>

namespace Generators {

void NormalizePipelineConfig(PipelineConfig& config) {
  for (auto& step : config.flow) {
    if (step.when == "prompt" || step.when == "once") {
      step.when = "init";
    } else if (step.when == "always") {
      step.when = "step";
    }
  }
}

void ValidatePipelineConfig(const PipelineConfig& config) {
  // Validate flow steps reference existing sessions
  for (const auto& step : config.flow) {
    if (config.sessions.find(step.run) == config.sessions.end()) {
      throw std::runtime_error(
          "Pipeline config error: flow step references unknown session '" + step.run + "'");
    }
    if (step.when != "init" && step.when != "step" && step.when != "final" &&
        step.when != "prompt" && step.when != "always" && step.when != "once") {
      throw std::runtime_error(
          "Pipeline config error: invalid 'when' value '" + step.when +
          "' for flow step '" + step.run + "'. Expected 'init', 'step', or 'final'.");
    }
    if (!step.loop.empty() && step.loop != "per_image" && step.loop != "batched") {
      throw std::runtime_error(
          "Pipeline config error: invalid 'loop' value '" + step.loop +
          "' for flow step '" + step.run + "'. Expected '', 'per_image', or 'batched'.");
    }
  }

  // Validate generation_loop (resolve default if unset)
  const auto& loop = config.generation_loop.value_or("autoregressive");
  if (loop == "autoregressive" || loop == "single_pass") {
    // Valid values — no-op
  } else if (loop == "denoising") {
    throw std::runtime_error(
        "Pipeline config error: generation_loop 'denoising' is not yet implemented.");
  } else {
    throw std::runtime_error(
        "Pipeline config error: unknown generation_loop value '" + loop +
        "'. Expected 'autoregressive', 'single_pass', or 'denoising'.");
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
