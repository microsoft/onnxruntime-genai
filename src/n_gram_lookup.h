// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace Generators {

class NGramLookup {
 public:
  explicit NGramLookup(int ngram_size)
      : ngram_size_{ngram_size}, key_length_{static_cast<size_t>(ngram_size - 1)} {
    if (ngram_size < 2)
      throw std::runtime_error("NGramLookup requires ngram_size >= 2.");
  }

  void Sync(std::span<const int32_t> committed) {
    const bool extends_history =
        committed.size() >= history_.size() &&
        std::equal(history_.begin(), history_.end(), committed.begin());
    if (!extends_history)
      Clear();

    for (size_t i = history_.size(); i < committed.size(); i++)
      Append(committed[i]);
  }

  std::vector<int32_t> Propose(size_t max_tokens) const {
    if (max_tokens == 0 || history_.size() < key_length_)
      return {};

    const auto key = MakeKey(history_.size() - key_length_);
    const auto occurrence = occurrences_.find(key);
    if (occurrence == occurrences_.end() || occurrence->second.empty())
      return {};

    const size_t continuation = occurrence->second.back() + key_length_;
    const size_t count = std::min(max_tokens, history_.size() - continuation);
    return {history_.begin() + static_cast<ptrdiff_t>(continuation),
            history_.begin() + static_cast<ptrdiff_t>(continuation + count)};
  }

  void Clear() {
    history_.clear();
    occurrences_.clear();
  }

  int NGramSize() const { return ngram_size_; }
  size_t HistorySize() const { return history_.size(); }

 private:
  struct KeyHash {
    size_t operator()(const std::vector<int32_t>& key) const noexcept {
      size_t hash = 1469598103934665603ull;
      for (int32_t token : key) {
        hash ^= static_cast<uint32_t>(token);
        hash *= 1099511628211ull;
      }
      return hash;
    }
  };

  void Append(int32_t token) {
    history_.push_back(token);
    if (history_.size() <= key_length_)
      return;

    const size_t newly_eligible_start = history_.size() - key_length_ - 1;
    occurrences_[MakeKey(newly_eligible_start)].push_back(newly_eligible_start);
  }

  std::vector<int32_t> MakeKey(size_t start) const {
    return {history_.begin() + static_cast<ptrdiff_t>(start),
            history_.begin() + static_cast<ptrdiff_t>(start + key_length_)};
  }

  int ngram_size_;
  size_t key_length_;
  std::vector<int32_t> history_;
  std::unordered_map<std::vector<int32_t>, std::vector<size_t>, KeyHash> occurrences_;
};

}  // namespace Generators
