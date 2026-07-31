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
 * [past, past + unprocessed), so the cache must own `past + unprocessed` slots after the step.
 * `sequence_length` (the request's current sequence length) already counts the unprocessed tokens
 * for a prefill request and excludes them for a generation (decode) request, hence the branch:
 *
 * - prefill: every token in the sequence is unprocessed, so the sequence length is the slot count;
 * - decode: the sequence length counts only already-processed tokens, so the freshly generated
 *   `unprocessed_count` tokens must be added on top.
 *
 * Counting appended tokens instead would leave the cache exactly one slot short on every decode
 * step -- invisible while the last block still has room, but an out-of-bounds block-table entry the
 * moment a sequence length lands on a block boundary.
 *
 * @param sequence_length  Current sequence length of the request.
 * @param unprocessed_count Tokens not yet processed by the model (relevant only when decoding).
 * @param is_prefill        True while the request is still prefilling its prompt.
 * @return The number of KV slots the request must own after its pending step.
 */
inline size_t RequiredSlots(size_t sequence_length, size_t unprocessed_count, bool is_prefill) {
  return is_prefill ? sequence_length : sequence_length + unprocessed_count;
}

}  // namespace Generators
