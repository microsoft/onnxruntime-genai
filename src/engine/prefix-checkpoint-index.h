// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "../span.h"

namespace Generators {

// Entries provide Tokens() and IsResident(); payload ownership and safe capture boundaries remain backend-owned.
template <typename Entry>
class PrefixCheckpointIndex {
 public:
  // The result is borrowed until the next index mutation. Lookup never touches storage recency.
  const Entry* Find(std::span<const int32_t> tokens) const {
    const Entry* best = nullptr;
    for (const auto& entry : entries_) {
      const auto prefix = entry.Tokens();
      if (entry.IsResident() && !prefix.empty() && prefix.size() <= tokens.size() &&
          (!best || prefix.size() > best->Tokens().size()) &&
          std::equal(prefix.begin(), prefix.end(), tokens.begin()))
        best = &entry;
    }
    return best;
  }

  size_t Prune() {
    const size_t previous = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [](const Entry& entry) { return !entry.IsResident(); }),
                   entries_.end());
    return previous - entries_.size();
  }

  // Reserve before allocating snapshot storage so publication need not allocate after eviction.
  void ReserveForPublication() { entries_.reserve(entries_.size() + 1); }
  void Publish(Entry entry) { entries_.push_back(std::move(entry)); }

 private:
  std::vector<Entry> entries_;
};

}  // namespace Generators
