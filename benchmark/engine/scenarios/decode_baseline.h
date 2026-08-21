// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "scenarios/scenario_base.h"

namespace engine_benchmark {

class DecodeBaselineScenario final : public ScenarioBase {
 protected:
  std::string Name() const override { return "decode_baseline"; }
  void ValidateConfig(const ScenarioConfig& config) const override;
  ScenarioExecutionOutput Execute(const ScenarioConfig& config, const BenchmarkContext& context) const override;
};

}  // namespace engine_benchmark
