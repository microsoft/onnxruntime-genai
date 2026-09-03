// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

// Documented bounds on a stop-string configuration. They are applied at construction, before any
// matcher state is allocated, so a caller cannot force unbounded per-request storage. Hosts may
// enforce smaller limits of their own.
inline constexpr size_t kMaxStopStringCount = 16;
inline constexpr size_t kMaxStopStringTotalBytes = 16 * 1024;

// Returns true when `bytes` is a well-formed UTF-8 sequence. Overlong encodings, surrogate code
// points, values above U+10FFFF, and truncated or stray continuation bytes are all rejected.
bool IsValidUtf8(std::string_view bytes);

// A completed stop-string match. `start_offset`/`end_offset` are absolute byte offsets into the
// stream of bytes handed to StopStringMatcher::Consume() since construction or the last Reset(),
// with `end_offset` exclusive. `index` is the caller's index into the stop-string vector the
// matcher was constructed with, which is preserved even when two entries have identical bytes.
struct StopStringMatch {
  size_t index{};
  uint64_t start_offset{};
  uint64_t end_offset{};
};

// Model-independent, incremental, exact UTF-8 byte matcher for stop strings.
//
// The matcher knows nothing about tokens or tokenizers: a caller decodes generated tokens into
// UTF-8 byte chunks and feeds those chunks to Consume() in order. Chunk boundaries are arbitrary
// and may split both a stop string and a single code point, so matching is done over the byte
// stream rather than per chunk. Matching is exact: no Unicode normalization, case folding, or
// whitespace trimming.
//
// Because a chunk may end with a byte sequence that is still a possible prefix of a stop string,
// output is split into two parts:
//   * safe output - bytes that provably cannot participate in a later match, retrieved with
//     TakeSafeOutput() and publishable immediately;
//   * pending output - the withheld suffix, visible through PendingOutput(), which becomes safe
//     only when later bytes rule out a match or the stream ends (Flush()).
// Pending storage is bounded by (longest stop string - 1) bytes between calls, and by that plus
// the current chunk during a call.
//
// When several stop strings could match, the matcher picks deterministically: earliest ending byte,
// then earliest starting byte (the longest match ending there), then the lowest caller index. With
// "ab" and "abc" the stream stops as soon as "ab" completes; with "abc" and "bc", which can end on
// the same byte, "abc" wins.
//
// A match is sticky. Once Consume() reports a match, the matcher is done: bytes trailing the match
// in that same chunk are dropped, later Consume() calls ignore their input and report no match, and
// the reported match cannot change until Reset().
//
// Matching runs one independent Knuth-Morris-Pratt automaton per configured stop string: each
// entry's KMP failure table is built once, at construction, in time linear in that entry's length
// (so construction is O(total configured bytes) overall), and Consume() advances every automaton by
// one byte at a time, so a call costs O(stop string count * bytes in the chunk) -- the per-automaton
// per-byte work is O(1) amortized, by the standard KMP argument, regardless of how long the current
// withheld suffix is. There is no per-byte rescan of the withheld suffix against every stop string
// (that would cost O(withheld length) per byte and made a shared-prefix, near-miss configuration
// quadratic in the number of decoded bytes).
class StopStringMatcher {
 public:
  // Copies and validates `stop_strings`. Throws std::runtime_error if there are more than
  // kMaxStopStringCount entries, if the entries total more than kMaxStopStringTotalBytes, or if any
  // entry is empty or is not well-formed UTF-8. Duplicate entries are allowed and keep their own
  // caller indices. An empty vector is valid and yields a matcher that never matches.
  explicit StopStringMatcher(std::vector<std::string> stop_strings);

  // Consumes the next chunk of decoded bytes. The chunk itself need not be valid UTF-8 on its own,
  // since a code point may straddle a chunk boundary. Returns the match that completes in this
  // chunk, if any; returns std::nullopt for every call after a match. Throws std::runtime_error if
  // called after Flush() -- Flush() means the stream has ended, so a later Consume() is a caller
  // bug, not something to silently absorb (see Flush()'s comment).
  std::optional<StopStringMatch> Consume(std::string_view bytes);

  // Bytes that cannot be part of a future match, removed from the matcher.
  std::string TakeSafeOutput();

  // The withheld suffix that is still a possible prefix of a stop string. Invalidated by Consume(),
  // Flush(), and Reset().
  std::string_view PendingOutput() const { return pending_; }

  // Ends the stream: no more bytes will be consumed, so the withheld suffix can no longer become
  // part of a match. Returns the safe output followed by that suffix and empties both. After a
  // match this returns only the bytes preceding the matched stop string. Marks the matcher flushed:
  // any later Consume() throws std::runtime_error rather than silently starting a new, unrelated
  // match window at stale offsets. Calling Flush() again before a Reset() is harmless and returns
  // an empty string, since there is nothing left to drain.
  std::string Flush();

  bool Matched() const { return match_.has_value(); }
  const std::optional<StopStringMatch>& Match() const { return match_; }

  // Total bytes handed to Consume(), including the chunk that completed a match and excluding
  // chunks ignored after it.
  uint64_t ConsumedBytes() const { return consumed_bytes_; }

  const std::vector<std::string>& StopStrings() const { return stop_strings_; }

  // Drops all stream state (offsets, buffered output, and the match) but keeps the configuration.
  // Also clears the flushed state from a prior Flush(), so the matcher accepts a fresh stream.
  // Used between generation turns.
  void Reset();

 private:
  std::vector<std::string> stop_strings_;
  // stop_string_failure_tables_[i] is the standard KMP failure (partial-match) table for
  // stop_strings_[i]: failure_table[k] is the length of the longest proper prefix of
  // stop_strings_[i][0..k] that is also a suffix of it. Built once in the constructor.
  std::vector<std::vector<size_t>> stop_string_failure_tables_;
  // current_prefix_lengths_[i] is stop_strings_[i]'s current KMP automaton state: the length of the
  // longest prefix of stop_strings_[i] that matches a suffix of every byte consumed so far this
  // stream. Reaching stop_strings_[i].size() means that entry just matched.
  std::vector<size_t> current_prefix_lengths_;
  size_t longest_stop_string_{};

  std::string safe_;                 // Bytes cleared for publication.
  std::string pending_;              // Withheld possible-prefix suffix.
  uint64_t pending_start_offset_{};  // Absolute offset of pending_[0].
  uint64_t consumed_bytes_{};
  std::optional<StopStringMatch> match_;
  // Set by Flush(), cleared by Reset(). Consume() throws if this is set: the stream has already
  // ended, so a later Consume() call is a caller lifecycle bug rather than legitimate new input.
  bool flushed_{false};
};

}  // namespace Generators
