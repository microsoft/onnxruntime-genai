// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Generators {

// Absolute-position arithmetic shared by the varlen decoder and the paged KV cache.
//
// PagedAttention writes token j of sequence i at absolute position `past_sequence_lengths[i] + j`,
// so the decoder and the cache have to agree on what that base is. Both derive it from the number
// of tokens whose keys and values are already in the cache -- never from the sequence length, which
// is one longer on a decode step because the search has already appended the token that is about to
// be processed.

// Slots the cache must own once the pending step has run: the step writes at absolute positions
// [processed_length, processed_length + scheduled_tokens).
constexpr size_t SlotsAfterStep(int64_t processed_length, size_t scheduled_tokens) {
  return static_cast<size_t>(processed_length) + scheduled_tokens;
}

// Slots the request will need once its whole sequence is in the cache. Used for admission, so that
// a request is never accepted and then left stalled part way through its prefill.
constexpr size_t SlotsForWholeSequence(int64_t sequence_length) {
  return static_cast<size_t>(sequence_length);
}

// Tokens a request contributes to the step that is about to run, out of the `unprocessed` tokens it
// still has to push through the model.
//
// With chunking enabled and a `search.chunk_size` configured, a prompt longer than the chunk size
// is spread over several steps, which bounds the activation footprint and the latency of a single
// step. Everything downstream -- the cache sizing, the decoder inputs and the logits row selection
// -- has to agree on this count, hence one definition of it.
constexpr size_t ScheduledTokenCount(size_t unprocessed, std::optional<size_t> chunk_size, bool allow_chunking) {
  if (!allow_chunking || !chunk_size.has_value() || *chunk_size == 0) {
    return unprocessed;
  }
  return std::min(*chunk_size, unprocessed);
}

}  // namespace Generators
