// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

/**
 * @file admission.h
 * @brief Pure admission arithmetic used when deciding how much KV-cache room a request needs.
 *
 * These helpers take plain counters rather than a Request so they can be unit-tested without a
 * model, a device, or a Search. The paged cache manager wraps them with request-derived values.
 */

namespace Generators {

/**
 * @brief Number of KV slots a request will have addressed once its pending step has run.
 *
 * The decoder writes a request's unprocessed tokens at absolute positions
 * [processed_sequence_length, processed_sequence_length + unprocessed_token_count).
 *
 * @param processed_sequence_length Tokens already represented in the KV cache.
 * @param unprocessed_token_count Tokens that will be written by the pending step.
 * @return The number of KV slots the request must own after its pending step.
 */
inline size_t RequiredSlots(size_t processed_sequence_length, size_t unprocessed_token_count) {
  return processed_sequence_length + unprocessed_token_count;
}

}  // namespace Generators
