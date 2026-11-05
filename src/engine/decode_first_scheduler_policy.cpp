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
    size_t max_scheduled_tokens,
    std::vector<bool>* chunk_service) {
  if (max_scheduled_tokens == 0)
    throw std::invalid_argument("The scheduled token budget must be positive.");
  if (selected_candidates.size() > max_scheduled_tokens)
    throw std::invalid_argument("The selected requests exceed the scheduled token budget.");

  std::vector<size_t> token_counts(selected_candidates.size(), 1);
  size_t remaining_budget = max_scheduled_tokens - selected_candidates.size();
  bool verifies_drafts = false;
  for (size_t i = 0; i < selected_candidates.size(); ++i) {
    const auto& candidate = selected_candidates[i];
    if (candidate.pending_token_count == 0)
      throw std::invalid_argument("A scheduling candidate must have pending tokens.");
    if (candidate.is_prefill)
      continue;
    if (candidate.decode_extra_token_count != 0 && candidate.pending_token_count > 1)
      throw std::invalid_argument("A decode row cannot verify drafts before reading its committed tokens.");

    const size_t additional_tokens =
        std::min(candidate.decode_extra_token_count, remaining_budget);
    token_counts[i] += additional_tokens;
    remaining_budget -= additional_tokens;
    verifies_drafts |= additional_tokens != 0;
  }

  // Chunk work shares what drafts leave, in plan order, so prompts finish in the order they came.
  // With a decode row catching up on fast-forward tokens among them, the row that has waited
  // longest goes first instead, so neither catch-up nor prompt work can keep the other from spare
  // tokens. A catch-up row reads only one token while a peer verifies drafts, so it never holds
  // that back.
  std::vector<size_t> chunk_rows;
  bool has_catch_up = false;
  for (size_t i = 0; i < selected_candidates.size(); ++i) {
    const auto& candidate = selected_candidates[i];
    const bool catches_up =
        !candidate.is_prefill && candidate.pending_token_count > 1 && !verifies_drafts;
    if (candidate.is_prefill || catches_up)
      chunk_rows.push_back(i);
    has_catch_up |= catches_up;
  }
  if (has_catch_up) {
    std::stable_sort(chunk_rows.begin(), chunk_rows.end(), [&](size_t left, size_t right) {
      return selected_candidates[left].last_chunk_service < selected_candidates[right].last_chunk_service;
    });
  }
  if (chunk_service)
    chunk_service->assign(selected_candidates.size(), false);
  bool first_share = true;
  for (const size_t i : chunk_rows) {
    const size_t wanted_tokens = ChunkLimit(selected_candidates[i]) - 1;
    const size_t additional_tokens = std::min(wanted_tokens, remaining_budget);
    if (additional_tokens == 0)
      continue;
    token_counts[i] += additional_tokens;
    remaining_budget -= additional_tokens;
    if (chunk_service)
      (*chunk_service)[i] = first_share || additional_tokens == wanted_tokens;
    first_share = false;
  }
  return token_counts;
}

}  // namespace Generators
