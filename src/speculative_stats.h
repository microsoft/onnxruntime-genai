// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>

namespace Generators {

// Instrumentation snapshot returned by Generator::GetSpeculativeStats().
// All fields are zero for non-speculative models.
struct SpeculativeStats {
  size_t rounds{};
  size_t draft_tokens_proposed{};
  size_t draft_tokens_accepted{};
  size_t correction_tokens{};
  size_t bonus_tokens{};
  float avg_draft_ms_per_token{};
  float avg_target_ms_per_token{};
  float acceptance_rate{};
  float mean_accepted_tokens{};
  float effective_speedup{};
};

}  // namespace Generators
