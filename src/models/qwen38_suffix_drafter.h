// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace Generators {

class Qwen38SuffixDrafter {
 public:
  Qwen38SuffixDrafter() { Reset(); }

  void Reset() noexcept {
    tokens_.clear();
    positions2_.fill(kMissing);
    positions3_.fill(kMissing);
  }

  void AppendToken(int32_t token) {
    if (token < 0) throw std::invalid_argument("qwen38_graph: suffix history requires nonnegative token ids");
    const size_t next = tokens_.size();
    tokens_.push_back(token);
    if (next >= 2) positions2_[Slot(tokens_[next - 2], tokens_[next - 1])] = next;
    if (next >= 3) positions3_[Slot(tokens_[next - 3], tokens_[next - 2], tokens_[next - 1])] = next;
  }

  void AppendSpan(std::span<const int32_t> tokens) {
    for (int32_t token : tokens)
      if (token < 0) throw std::invalid_argument("qwen38_graph: suffix history requires nonnegative token ids");
    tokens_.reserve(tokens_.size() + tokens.size());
    for (int32_t token : tokens) AppendToken(token);
  }

  size_t ProposeDrafts(int32_t pending, std::span<int32_t> drafts, int min_ngram = 3) const {
    if (pending < 0 || (min_ngram != 2 && min_ngram != 3))
      throw std::invalid_argument("qwen38_graph: suffix lookup requires a nonnegative token and a 2- or 3-gram");
    size_t position = kMissing;
    const size_t size = tokens_.size();
    if (size >= 2) {
      const int32_t a = tokens_[size - 2], b = tokens_[size - 1];
      const size_t candidate = positions3_[Slot(a, b, pending)];
      // Compare the actual tokens: a hash (including a truncated 21-bit reference key) is not an n-gram identity.
      if (candidate != kMissing && tokens_[candidate - 3] == a &&
          tokens_[candidate - 2] == b && tokens_[candidate - 1] == pending)
        position = candidate;
    }
    if (position == kMissing && min_ngram == 2 && size >= 1) {
      const int32_t a = tokens_.back();
      const size_t candidate = positions2_[Slot(a, pending)];
      if (candidate != kMissing && tokens_[candidate - 2] == a && tokens_[candidate - 1] == pending)
        position = candidate;
    }
    if (position == kMissing) return 0;
    const size_t count = std::min(drafts.size(), size - position);
    for (size_t i = 0; i < count; ++i) drafts[i] = tokens_[position + i];
    return count;
  }

 private:
  static constexpr size_t kTableSize = 8192;
  static constexpr size_t kMissing = static_cast<size_t>(-1);

  static uint64_t Hash(uint64_t value) noexcept {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    return value ^ (value >> 33);
  }

  static size_t Slot(int32_t a, int32_t b) noexcept {
    return Hash((static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b)) & (kTableSize - 1);
  }

  static size_t Slot(int32_t a, int32_t b, int32_t c) noexcept {
    return Hash(Hash((static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
                     static_cast<uint32_t>(b)) ^
                static_cast<uint32_t>(c)) &
           (kTableSize - 1);
  }

  std::vector<int32_t> tokens_;
  std::array<size_t, kTableSize> positions2_;
  std::array<size_t, kTableSize> positions3_;
};

}  // namespace Generators
