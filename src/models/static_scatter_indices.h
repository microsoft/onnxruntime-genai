// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace Generators {

// The per-step index pair a static-scatter (TensorScatter) KV cache needs.
//
// A mobius-exported static-cache decoder consumes a pre-allocated KV buffer of
// shape [batch, max_seq_len, kv_hidden] and writes each step's new key/value
// rows into it in place via TensorScatter, then reads them back through
// Attention. Two int64 [batch] inputs drive that:
//   * write_indices    - the cache row offset TensorScatter writes this step's
//                        rows at (i.e. how many valid tokens are already cached
//                        BEFORE this step).
//   * nonpad_kv_seqlen - the number of valid cached tokens AFTER this step,
//                        which Attention reads as the per-batch seqlens_k.
struct StaticScatterIndices {
  int64_t write_index;     // valid cache tokens before this step (scatter offset)
  int64_t nonpad_seqlen;   // valid cache tokens after this step (Attention seqlens_k)
};

// Tracks the running static-scatter cache indices for a single (batch==1)
// generation stream. Kept as a standalone, dependency-free helper so the
// off-by-one / init behaviour can be unit-tested without standing up a Model.
//
// Sequencing contract (the crux mobius and genai must agree on):
//   * The very first step (prefill) writes at row 0 and reports nonpad equal to
//     the number of prefill tokens.
//   * Each subsequent step's write_index is the PREVIOUS step's nonpad_seqlen,
//     so rows are appended contiguously with no gap or overlap.
// This deliberately does NOT reuse genai's existing past_sequence_length scalar
// (which inits to -1 and is consumed differently); mixing the two would yield
// nonpad = 2N-1 after a length-N prefill instead of N.
class StaticScatterIndexTracker {
 public:
  // Advance one generation step that appended new_unpadded_tokens valid tokens
  // (the prompt length on prefill, normally 1 per decode step). Returns the
  // index pair to bind for THIS step, then folds the new tokens into the
  // running total for the next step.
  StaticScatterIndices Advance(int64_t new_unpadded_tokens) {
    const int64_t write_index = valid_tokens_;
    valid_tokens_ += new_unpadded_tokens;
    return {write_index, valid_tokens_};
  }

  // Valid cached tokens before the next step. Zero before any Advance().
  int64_t valid_tokens() const { return valid_tokens_; }

  // Reset the stream back to an empty cache (e.g. on RewindTo(0)).
  void Reset() { valid_tokens_ = 0; }

 private:
  int64_t valid_tokens_{0};
};

}  // namespace Generators
