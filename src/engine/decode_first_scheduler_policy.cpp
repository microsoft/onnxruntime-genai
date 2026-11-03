// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decode_first_scheduler_policy.h"

#include <algorithm>
#include <stdexcept>

namespace Generators {

namespace {

size_t ChunkLimit(const DecodeFirstBudgetCandidate& candidate) {
  if (candidate.chunk_token_cap.value_or(0) == 0)
    return candidate.pending_token_count;
  return std::min(candidate.pending_token_count, *candidate.chunk_token_cap);
}

}  // namespace

std::vector<size_t> DecodeFirstCandidateOrder(
    std::span<const DecodeFirstBudgetCandidate> candidates) {
  std::vector<size_t> order;
  order.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].is_prefill)
      order.push_back(i);
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].is_prefill)
      order.push_back(i);
  }
  return order;
}

size_t DecodeFirstProvisionalRequestLimit(
    size_t max_scheduled_tokens,
    size_t max_batch_size) {
  if (max_scheduled_tokens == 0 || max_batch_size == 0)
    throw std::invalid_argument("Decode-first scheduling limits must be positive.");
  return std::min(max_scheduled_tokens, max_batch_size);
}

std::vector<size_t> AllocateDecodeFirstTokenBudget(
    std::span<const DecodeFirstBudgetCandidate> selected_candidates,
    size_t max_scheduled_tokens) {
  if (max_scheduled_tokens == 0)
    throw std::invalid_argument("The scheduled token budget must be positive.");
  if (selected_candidates.size() > max_scheduled_tokens)
    throw std::invalid_argument("The selected requests exceed the scheduled token budget.");

  std::vector<size_t> token_counts(selected_candidates.size(), 1);
  size_t remaining_budget = max_scheduled_tokens - selected_candidates.size();
  for (size_t i = 0; i < selected_candidates.size(); ++i) {
    const auto& candidate = selected_candidates[i];
    if (candidate.pending_token_count == 0)
      throw std::invalid_argument("A scheduling candidate must have pending tokens.");
    if (candidate.is_prefill)
      continue;

    if (candidate.decode_extra_token_count != 0 && candidate.pending_token_count > 1)
      throw std::invalid_argument("A decode row cannot verify drafts before reading its committed tokens.");

    // Beyond its first token, a decode row reads its drafts or up to one chunk of its unread
    // committed tokens.
    const size_t additional_tokens =
        std::min(candidate.decode_extra_token_count + ChunkLimit(candidate) - 1,
                 remaining_budget);
    token_counts[i] += additional_tokens;
    remaining_budget -= additional_tokens;
  }
  for (size_t i = 0; i < selected_candidates.size(); ++i) {
    const auto& candidate = selected_candidates[i];
    if (!candidate.is_prefill)
      continue;

    const size_t additional_tokens =
        std::min(ChunkLimit(candidate) - 1, remaining_budget);
    token_counts[i] += additional_tokens;
    remaining_budget -= additional_tokens;
  }
  return token_counts;
}

}  // namespace Generators
