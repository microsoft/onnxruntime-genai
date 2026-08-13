// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/long_prefill.h"

#include <stdexcept>

#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8;
}

// Prompt length buckets from the RULER subset the scenario is intended to mirror.
bool IsAllowedPromptLength(int prompt_length_k) {
  switch (prompt_length_k) {
    case 4:
    case 16:
    case 18:
    case 32:
    case 48:
    case 64:
    case 96:
    case 128:
      return true;
    default:
      return false;
  }
}

}  // namespace

void LongPrefillScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("long_prefill requires concurrency in [1,2,4,8]");
  }
  if (!IsAllowedPromptLength(config.prompt_length_k)) {
    throw std::invalid_argument("long_prefill requires prompt_length_k in [4,16,18,32,48,64,96,128]");
  }
  if (!config.synthetic) {
    // Dataset-backed prompts (data/ruler/prompts.json) are not wired up yet.
    throw std::invalid_argument("long_prefill currently requires synthetic=true");
  }
}

ScenarioExecutionOutput LongPrefillScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  return RunEngineWorkload(config, Name());
}

}  // namespace engine_benchmark
