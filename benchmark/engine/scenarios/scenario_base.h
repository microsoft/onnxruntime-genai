// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "scenarios/utils.h"

namespace engine_benchmark {

class ScenarioBase {
 public:
  virtual ~ScenarioBase() = default;

  nlohmann::json Run(const ScenarioConfig& config, const BenchmarkContext& context) const;

  /// Constructs the scenario registered under `name` (see "Adding a scenario" in README.md.)
  static std::unique_ptr<ScenarioBase> Create(const std::string& name);

  template <typename T>
  struct Registrar {
    explicit Registrar(const std::string& name) {
      Factories()[name] = [] { return std::make_unique<T>(); };
    }
  };

 protected:
  virtual std::string Name() const = 0;
  virtual void ValidateConfig(const ScenarioConfig& config) const;
  virtual ScenarioExecutionOutput Execute(const ScenarioConfig& config, const BenchmarkContext& context) const = 0;

 private:
  using Factory = std::function<std::unique_ptr<ScenarioBase>()>;
  static std::map<std::string, Factory>& Factories();
};

}  // namespace engine_benchmark
