// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "common/benchmark_types.h"

namespace engine_benchmark {

class ScenarioBase {
 public:
  virtual ~ScenarioBase() = default;

  nlohmann::json Run(const ScenarioConfig& config, const BenchmarkContext& context) const;

 protected:
  virtual std::string Name() const = 0;
  virtual void ValidateConfig(const ScenarioConfig& config) const;
  virtual ScenarioExecutionOutput Execute(const ScenarioConfig& config, const BenchmarkContext& context) const = 0;
};

}  // namespace engine_benchmark
