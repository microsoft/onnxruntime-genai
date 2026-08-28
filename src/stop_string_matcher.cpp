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
}

std::optional<StopStringMatch> StopStringMatcher::Consume(std::string_view bytes) {
  if (match_)
    return std::nullopt;  // Sticky: the stream ended at the match, so later bytes are dropped.

  consumed_bytes_ += bytes.size();
  if (stop_strings_.empty()) {
    safe_.append(bytes);
    return std::nullopt;
  }

  // Every match ending at or before the end of the previous chunk was already ruled out, so only
  // end positions past the retained suffix need to be examined.
  const size_t searched = pending_.size();
  pending_.append(bytes);

  if (const auto match = FindMatch(searched)) {
    safe_.append(pending_, 0, static_cast<size_t>(match->start_offset));
    match_ = StopStringMatch{match->index, pending_start_offset_ + match->start_offset, pending_start_offset_ + match->end_offset};
    pending_.clear();
    pending_start_offset_ = match_->end_offset;
    return match_;
  }

  // Only the longest suffix that is still a possible prefix can take part in a later match; the
  // bytes before it are safe to publish. Suffixes are nested, so no shorter candidate starts
  // earlier.
  const size_t release = pending_.size() - LongestPossiblePrefixSuffix();
  safe_.append(pending_, 0, release);
  pending_.erase(0, release);
  pending_start_offset_ += release;
  return std::nullopt;
}

std::string StopStringMatcher::TakeSafeOutput() {
  std::string output = std::move(safe_);
  safe_.clear();
  return output;
}

std::string StopStringMatcher::Flush() {
  std::string output = TakeSafeOutput();
  output.append(pending_);
  pending_start_offset_ += pending_.size();
  pending_.clear();
  return output;
}

void StopStringMatcher::Reset() {
  safe_.clear();
  pending_.clear();
  pending_start_offset_ = 0;
  consumed_bytes_ = 0;
  match_.reset();
}

std::optional<StopStringMatch> StopStringMatcher::FindMatch(size_t min_end) const {
  for (size_t end = min_end + 1; end <= pending_.size(); ++end) {
    size_t best_length = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < stop_strings_.size(); ++i) {
      const std::string& stop_string = stop_strings_[i];
      // Longest first, and the ascending index scan keeps the lowest index for equal lengths.
      if (stop_string.size() <= end && stop_string.size() > best_length &&
          pending_.compare(end - stop_string.size(), stop_string.size(), stop_string) == 0) {
        best_length = stop_string.size();
        best_index = i;
      }
    }
    if (best_length != 0)
      return StopStringMatch{best_index, end - best_length, end};
  }
  return std::nullopt;
}

size_t StopStringMatcher::LongestPossiblePrefixSuffix() const {
  if (stop_strings_.empty())
    return 0;

  // A complete match was already ruled out, so only proper prefixes are candidates.
  for (size_t length = std::min(pending_.size(), longest_stop_string_ - 1); length > 0; --length) {
    for (const std::string& stop_string : stop_strings_) {
      if (stop_string.size() > length && stop_string.compare(0, length, pending_, pending_.size() - length, length) == 0)
        return length;
    }
  }
  return 0;
}

}  // namespace Generators
