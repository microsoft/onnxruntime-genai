// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "../span.h"
#include "prefix_cache.h"

namespace Generators {

// Entries provide Tokens() and IsResident(); payloads and safe boundaries remain backend-owned.
template <typename Entry>
class PrefixCheckpointIndex {
 public:
  // Borrowed until the next index mutation. Lookup neither leases nor touches storage.
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

  void Publish(Entry entry) { entries_.push_back(std::move(entry)); }

  // Build identity before allocating snapshot storage. Capture returns an optional entry and
  // accounts for backend-specific refusals; final publication must not allocate payloads.
  template <typename Capture>
  void TryPublish(PrefixCacheMetrics& metrics, Capture&& capture) {
    static_assert(std::is_nothrow_move_constructible_v<Entry>, "Checkpoint publication must not allocate payloads.");
    try {
      if (entries_.size() == entries_.capacity()) {
        const auto capacity = entries_.capacity();
        const auto maximum = entries_.max_size();
        if (capacity == maximum) throw std::bad_alloc{};
        entries_.reserve(capacity == 0 ? 1 : capacity > maximum - capacity ? maximum
                                                                           : capacity * 2);
      }
      if (auto entry = capture()) Publish(std::move(*entry));
    } catch (const std::bad_alloc&) {
      ++metrics.publication_refusals;
    } catch (const std::system_error&) {
      ++metrics.publication_refusals;
    }
    metrics.evictions += Prune();
  }

 private:
  std::vector<Entry> entries_;
};

}  // namespace Generators
