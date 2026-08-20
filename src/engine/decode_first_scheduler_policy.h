// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "../span.h"

namespace Generators {

struct DecodeFirstBudgetCandidate {
  bool is_prefill{};
  size_t pending_token_count{};
  std::optional<size_t> prefill_token_cap;
};

std::vector<size_t> DecodeFirstCandidateOrder(
    std::span<const DecodeFirstBudgetCandidate> candidates);

size_t DecodeFirstProvisionalRequestLimit(
    size_t max_scheduled_tokens,
    size_t max_batch_size);

std::vector<size_t> AllocateDecodeFirstTokenBudget(
    std::span<const DecodeFirstBudgetCandidate> selected_candidates,
    size_t max_scheduled_tokens);

}  // namespace Generators
