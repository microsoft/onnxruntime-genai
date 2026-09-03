// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stop_string_matcher.h"

#include <algorithm>
#include <stdexcept>

namespace Generators {

namespace {

size_t Utf8SequenceLength(unsigned char lead) {
  if (lead < 0x80) return 1;
  if (lead >= 0xC2 && lead <= 0xDF) return 2;
  if (lead >= 0xE0 && lead <= 0xEF) return 3;
  if (lead >= 0xF0 && lead <= 0xF4) return 4;
  return 0;  // Continuation byte, overlong two-byte lead (0xC0/0xC1), or out of range (>= 0xF5).
}

}  // namespace

bool IsValidUtf8(std::string_view bytes) {
  size_t i = 0;
  while (i < bytes.size()) {
    const auto lead = static_cast<unsigned char>(bytes[i]);
    const size_t length = Utf8SequenceLength(lead);
    if (length == 0 || i + length > bytes.size())
      return false;

    for (size_t k = 1; k < length; ++k) {
      const auto continuation = static_cast<unsigned char>(bytes[i + k]);
      if (continuation < 0x80 || continuation > 0xBF)
        return false;
    }

    // Reject the encodings that are representable in this many bytes but are not the shortest form,
    // plus UTF-16 surrogates and code points above U+10FFFF.
    const auto second = length > 1 ? static_cast<unsigned char>(bytes[i + 1]) : static_cast<unsigned char>(0);
    if (length == 3 && lead == 0xE0 && second < 0xA0) return false;
    if (length == 3 && lead == 0xED && second > 0x9F) return false;
    if (length == 4 && lead == 0xF0 && second < 0x90) return false;
    if (length == 4 && lead == 0xF4 && second > 0x8F) return false;

    i += length;
  }
  return true;
}

StopStringMatcher::StopStringMatcher(std::vector<std::string> stop_strings)
    : stop_strings_{std::move(stop_strings)} {
  if (stop_strings_.size() > kMaxStopStringCount)
    throw std::runtime_error("Too many stop strings: " + std::to_string(stop_strings_.size()) + ", the maximum is " + std::to_string(kMaxStopStringCount) + ".");

  size_t total_bytes = 0;
  for (size_t i = 0; i < stop_strings_.size(); ++i) {
    const std::string& stop_string = stop_strings_[i];
    if (stop_string.empty())
      throw std::runtime_error("Stop string at index " + std::to_string(i) + " is empty.");
    if (!IsValidUtf8(stop_string))
      throw std::runtime_error("Stop string at index " + std::to_string(i) + " is not valid UTF-8.");

    total_bytes += stop_string.size();
    longest_stop_string_ = std::max(longest_stop_string_, stop_string.size());
  }

  if (total_bytes > kMaxStopStringTotalBytes)
    throw std::runtime_error("Stop strings total " + std::to_string(total_bytes) + " bytes, the maximum is " + std::to_string(kMaxStopStringTotalBytes) + ".");

  // Standard KMP failure (partial-match) table per stop string, each computed in time linear in
  // that entry's own length: failure_table[k] is the length of the longest proper prefix of
  // pattern[0..k] that is also a suffix of it.
  stop_string_failure_tables_.resize(stop_strings_.size());
  for (size_t i = 0; i < stop_strings_.size(); ++i) {
    const std::string& pattern = stop_strings_[i];
    std::vector<size_t>& failure_table = stop_string_failure_tables_[i];
    failure_table.assign(pattern.size(), 0);
    size_t matched = 0;
    for (size_t k = 1; k < pattern.size(); ++k) {
      while (matched > 0 && pattern[matched] != pattern[k])
        matched = failure_table[matched - 1];
      if (pattern[matched] == pattern[k])
        ++matched;
      failure_table[k] = matched;
    }
  }
  current_prefix_lengths_.assign(stop_strings_.size(), 0);
  pending_.reserve(longest_stop_string_);
}

std::optional<StopStringMatch> StopStringMatcher::Consume(std::string_view bytes) {
  if (flushed_)
    throw std::runtime_error("StopStringMatcher::Consume() called after Flush(); call Reset() first.");

  if (match_)
    return std::nullopt;  // Sticky: the stream ended at the match, so later bytes are dropped.

  consumed_bytes_ += bytes.size();
  if (stop_strings_.empty()) {
    safe_.append(bytes);
    return std::nullopt;
  }

  // Every live byte already in pending_ advanced every automaton in a previous call, so only the
  // newly appended bytes are fed through the automatons here. The live suffix is kept to serve
  // PendingOutput()/Flush() and to compute absolute offsets below.
  const size_t previously_pending = pending_.size() - pending_begin_;
  pending_.append(bytes);

  for (size_t k = 0; k < bytes.size(); ++k) {
    const auto byte = static_cast<unsigned char>(bytes[k]);
    size_t best_length = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < stop_strings_.size(); ++i) {
      const std::string& pattern = stop_strings_[i];
      const std::vector<size_t>& failure_table = stop_string_failure_tables_[i];
      size_t& prefix_length = current_prefix_lengths_[i];
      while (prefix_length > 0 && static_cast<unsigned char>(pattern[prefix_length]) != byte)
        prefix_length = failure_table[prefix_length - 1];
      if (static_cast<unsigned char>(pattern[prefix_length]) == byte)
        ++prefix_length;
      // Longest wins; the ascending index scan keeps the lowest index for equal lengths.
      if (prefix_length == pattern.size() && pattern.size() > best_length) {
        best_length = pattern.size();
        best_index = i;
      }
    }

    if (best_length != 0) {
      const uint64_t end = pending_start_offset_ + previously_pending + k + 1;
      match_ = StopStringMatch{best_index, end - best_length, end};
      safe_.append(pending_, pending_begin_, static_cast<size_t>(match_->start_offset - pending_start_offset_));
      pending_.clear();
      pending_begin_ = 0;
      pending_start_offset_ = match_->end_offset;
      return match_;
    }
  }

  // No match in this chunk. Only the longest currently active prefix can still take part in a
  // later match; the bytes before it are safe to publish.
  size_t longest_active_prefix = 0;
  for (size_t prefix_length : current_prefix_lengths_)
    longest_active_prefix = std::max(longest_active_prefix, prefix_length);
  const size_t release = pending_.size() - pending_begin_ - longest_active_prefix;
  safe_.append(pending_, pending_begin_, release);
  pending_begin_ += release;
  pending_start_offset_ += release;

  // Compact only after the discarded prefix is at least as large as the live suffix. This bounds
  // stale backing storage while amortizing the cost of moving a long near-match suffix.
  const size_t live_pending = pending_.size() - pending_begin_;
  if (pending_begin_ >= live_pending) {
    if (live_pending == 0) {
      pending_.clear();
    } else {
      pending_.erase(0, pending_begin_);
    }
    pending_begin_ = 0;
  }
  return std::nullopt;
}

std::string StopStringMatcher::TakeSafeOutput() {
  std::string output = std::move(safe_);
  safe_.clear();
  return output;
}

std::string StopStringMatcher::Flush() {
  std::string output = TakeSafeOutput();
  output.append(pending_, pending_begin_, pending_.size() - pending_begin_);
  pending_start_offset_ += pending_.size() - pending_begin_;
  pending_.clear();
  pending_begin_ = 0;
  flushed_ = true;
  return output;
}

void StopStringMatcher::Reset() {
  safe_.clear();
  pending_.clear();
  pending_begin_ = 0;
  pending_start_offset_ = 0;
  consumed_bytes_ = 0;
  match_.reset();
  flushed_ = false;
  std::fill(current_prefix_lengths_.begin(), current_prefix_lengths_.end(), 0);
}

}  // namespace Generators
