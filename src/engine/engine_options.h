// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../config.h"

namespace Generators {

enum class ExperimentalEnginePolicy : uint32_t {
  Default = 0,
  AgeBasedRecovery = 1,
};

struct RecomputeLimits {
  size_t max_victims_per_step{1};
};

struct AgeBasedRecoveryOptions {
  size_t min_decode_steps_before_preemption{8};
};

struct ExperimentalEngineOptions {
  explicit ExperimentalEngineOptions(uint32_t version = 1);

  void SetPolicy(ExperimentalEnginePolicy value);
  void SetRecomputeLimits(size_t max_victims_per_step);
  void SetAgeBasedRecovery(size_t min_decode_steps_before_preemption);

  std::optional<ExperimentalEnginePolicy> policy;
  std::optional<RecomputeLimits> recompute_limits;
  std::optional<AgeBasedRecoveryOptions> age_based_recovery;
};

struct EnginePreemptionOptions {
  ExperimentalEnginePolicy policy{ExperimentalEnginePolicy::Default};
  RecomputeLimits recompute_limits;
  AgeBasedRecoveryOptions age_based_recovery;

  bool Enabled() const noexcept { return policy == ExperimentalEnginePolicy::AgeBasedRecovery; }
};

void ValidateExperimentalEnginePolicy(ExperimentalEnginePolicy policy);
EnginePreemptionOptions ResolveEnginePreemptionOptions(
    const Config& config, const ExperimentalEngineOptions& options = ExperimentalEngineOptions{});

}  // namespace Generators
