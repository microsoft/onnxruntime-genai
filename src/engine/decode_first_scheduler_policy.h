// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../span.h"

namespace Generators {

struct DecodeFirstBudgetCandidate {
  bool is_prefill{};
  size_t pending_token_count{};
  // Bounds the tokens one step reads from a prefill, or from a decode row whose committed tokens
  // the model has not read yet.
  std::optional<size_t> chunk_token_cap;
  size_t decode_extra_token_count{};
  // When this row last read spare tokens as chunk work, or was admitted; smaller waited longer.
  uint64_t last_chunk_service{};
};

std::vector<size_t> DecodeFirstCandidateOrder(
    std::span<const DecodeFirstBudgetCandidate> candidates);

size_t DecodeFirstProvisionalRequestLimit(
    size_t max_scheduled_tokens,
    size_t max_batch_size);

// Gives every selected row one token, then drafts their extra tokens, then shares the rest among
// prefill chunks and decode rows reading unread committed tokens, longest-waiting first.
std::vector<size_t> AllocateDecodeFirstTokenBudget(
    std::span<const DecodeFirstBudgetCandidate> selected_candidates,
    size_t max_scheduled_tokens);

}  // namespace Generators
