// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine_options.h"

#include <stdexcept>

namespace Generators {

void ValidateExperimentalEnginePolicy(ExperimentalEnginePolicy policy) {
  switch (policy) {
    case ExperimentalEnginePolicy::Default:
    case ExperimentalEnginePolicy::AgeBasedRecovery:
      return;
  }
  throw std::invalid_argument(
      "Unsupported experimental Engine policy; supported choices are Default and AgeBasedRecovery.");
}

ExperimentalEngineOptions::ExperimentalEngineOptions(uint32_t version) {
  if (version != 1) {
    throw std::invalid_argument("Unsupported experimental Engine options version; this runtime supports version 1.");
  }
}

void ExperimentalEngineOptions::SetPolicy(ExperimentalEnginePolicy value) {
  ValidateExperimentalEnginePolicy(value);
  policy = value;
}

void ExperimentalEngineOptions::SetRecomputeLimits(size_t max_victims_per_step) {
  if (max_victims_per_step == 0) {
    throw std::invalid_argument("recompute_limits.max_victims_per_step must be > 0.");
  }
  recompute_limits = RecomputeLimits{max_victims_per_step};
}

void ExperimentalEngineOptions::SetAgeBasedRecovery(size_t min_decode_steps_before_preemption) {
  if (min_decode_steps_before_preemption == 0) {
    throw std::invalid_argument("age_based_recovery.min_decode_steps_before_preemption must be > 0.");
  }
  age_based_recovery = AgeBasedRecoveryOptions{min_decode_steps_before_preemption};
}

EnginePreemptionOptions ResolveEnginePreemptionOptions(
    const Config& config, const ExperimentalEngineOptions& options) {
  EnginePreemptionOptions resolved;
  const auto& batching = config.engine.dynamic_batching;
  if (batching) {
    resolved.policy = batching->enable_recompute_preemption
                          ? ExperimentalEnginePolicy::AgeBasedRecovery
                          : ExperimentalEnginePolicy::Default;
    resolved.recompute_limits.max_victims_per_step = batching->max_preemptions_per_step;
    resolved.age_based_recovery.min_decode_steps_before_preemption =
        batching->min_decode_steps_before_preemption;
  }
  if (options.policy) {
    ValidateExperimentalEnginePolicy(*options.policy);
    // A programmatically enabled legacy flag is also a selection. Parsed false needs provenance
    // so an omitted default does not veto the construction-time opt-in.
    if (batching &&
        (batching->recompute_preemption_explicitly_set || batching->enable_recompute_preemption) &&
        *options.policy != resolved.policy) {
      throw std::invalid_argument(
          "Experimental Engine policy conflicts with explicitly configured enable_recompute_preemption.");
    }
    resolved.policy = *options.policy;
  }
  if (!resolved.Enabled() && (options.recompute_limits || options.age_based_recovery)) {
    throw std::invalid_argument("Recompute limits and age_based_recovery options require AgeBasedRecovery.");
  }
  if (options.recompute_limits) {
    resolved.recompute_limits = *options.recompute_limits;
  }
  if (options.age_based_recovery) {
    resolved.age_based_recovery = *options.age_based_recovery;
  }
  if (resolved.Enabled() &&
      (resolved.recompute_limits.max_victims_per_step == 0 ||
       resolved.age_based_recovery.min_decode_steps_before_preemption == 0)) {
    throw std::invalid_argument("Recompute preemption limits must be positive.");
  }
  return resolved;
}

}  // namespace Generators
