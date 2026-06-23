// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
  int64_t write_index;       // valid cache tokens before this step (scatter offset)
  int64_t nonpad_kv_seqlen;  // valid cache tokens after this step (Attention seqlens_k)
};

// Tracks the running static-scatter cache indices for a single (batch==1)
// generation stream. Kept as a standalone, dependency-free helper so the
// off-by-one / init behaviour can be unit-tested without standing up a Model.
//
// Sequencing contract (the crux mobius and genai must agree on):
//   * The very first step (prefill) writes at row 0 and reports nonpad equal to
//     the number of prefill tokens.
//   * Each subsequent step's write_index is the PREVIOUS step's nonpad_kv_seqlen,
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
  //
  // pad_token aliasing assumption: a decode step whose generated token equals
  // pad_token_id yields new_unpadded_tokens == 0 (the unpadded-length probe in
  // input_ids.cpp counts it as padding), so write_index / valid_tokens_ do NOT
  // advance and the NEXT real token would scatter onto the same cache slot.
  // The CONDITION that makes this safe: for the targeted models pad_token_id ==
  // eos_token_id, so the very step that produces a pad token also ENDS the
  // sequence -- generation terminates before any later step could read the
  // stalled slot. Given that, the aliasing is benign (and mirrors genai's
  // existing current_sequence_length logic). It would only break if a model made
  // pad_token_id a legal mid-stream generated token, which the targeted models
  // do not.
  StaticScatterIndices Advance(int64_t new_unpadded_tokens) {
    const int64_t write_index = valid_tokens_;
    valid_tokens_ += new_unpadded_tokens;
    return {write_index, valid_tokens_};
  }

  // Valid cached tokens before the next step. Zero before any Advance().
  int64_t valid_tokens() const { return valid_tokens_; }

 private:
  int64_t valid_tokens_{0};
};

// Discover which decoder layers expose a KV cache input, parsing the layer index
// out of each matching input name. `prefix` / `suffix` bracket the numeric index
// in the past-key name template (e.g. "past_key_values.%d.key" -> prefix
// "past_key_values.", suffix ".key").
//
// The parse is STRICT: the index segment must be a COMPLETE, non-negative
// integer. std::stoi would accept trailing junk (e.g. "past_key_values.0.bad.key"
// -> 0) and silently mis-map a layer, so std::from_chars must consume the whole
// segment with no leftover characters. Duplicate indices are rejected, because
// two inputs mapping to one layer would double-count layer_count_ and bind the
// same cache slot twice. Throws std::runtime_error on a malformed or duplicate
// index. Returned indices are sorted ascending.
//
// Kept here as a standalone, model-free helper (alongside StaticScatterIndex-
// Tracker) so the strict-parse / dedup behaviour can be unit-tested directly,
// without standing up a Model.
inline std::vector<int> DiscoverKvLayerIndices(const std::vector<std::string>& input_names,
                                               const std::string& prefix,
                                               const std::string& suffix) {
  std::vector<int> indices;
  for (const auto& name : input_names) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      const auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      int layer_idx = 0;
      const char* begin = idx_str.data();
      const char* end = begin + idx_str.size();
      auto [parse_end, ec] = std::from_chars(begin, end, layer_idx);
      if (ec != std::errc{} || parse_end != end || layer_idx < 0) {
        throw std::runtime_error(
            "StaticScatterKeyValueCache: input '" + name +
            "' has a malformed KV layer index '" + idx_str +
            "' (expected a non-negative integer).");
      }
      if (std::find(indices.begin(), indices.end(), layer_idx) != indices.end()) {
        throw std::runtime_error(
            "StaticScatterKeyValueCache: duplicate KV layer index '" +
            std::to_string(layer_idx) + "' (from input '" + name + "').");
      }
      indices.push_back(layer_idx);
    }
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

}  // namespace Generators
