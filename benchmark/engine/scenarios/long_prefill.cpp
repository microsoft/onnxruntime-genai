// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/long_prefill.h"

#include <stdexcept>

#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

// Prompt length buckets this scenario targets (see benchmark-requirements.md).
bool IsAllowedPromptLength(int prompt_length_k) {
  return prompt_length_k == 32 || prompt_length_k == 64 || prompt_length_k == 128;
}

}  // namespace

void LongPrefillScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (config.concurrency != 1) {
    throw std::invalid_argument("long_prefill requires concurrency=1 so prefill is measured in isolation");
  }
  if (!IsAllowedPromptLength(config.prompt_length_k)) {
    throw std::invalid_argument("long_prefill requires prompt_length_k in [32,64,128]");
  }
  if (!config.synthetic) {
    // Dataset-backed prompts (data/ruler/prompts.json) are not wired up yet.
    throw std::invalid_argument("long_prefill currently requires synthetic=true");
  }
}

ScenarioExecutionOutput LongPrefillScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  ScenarioExecutionOutput output = RunEngineWorkload(config, Name());
  // Decode throughput is not meaningful when only a handful of tokens are generated.
  output.scenario_metrics.erase("tokens_per_s");
  return output;
}

}  // namespace engine_benchmark
