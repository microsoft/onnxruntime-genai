// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/decode_baseline.h"

#include <stdexcept>

#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8;
}

}  // namespace

void DecodeBaselineScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("decode_baseline requires concurrency in [1,2,4,8]");
  }
  if (!config.synthetic) {
    throw std::invalid_argument("decode_baseline requires synthetic=true");
  }
}

ScenarioExecutionOutput DecodeBaselineScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  return RunEngineWorkload(config, Name());
}

}  // namespace engine_benchmark
