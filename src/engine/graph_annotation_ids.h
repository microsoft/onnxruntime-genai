// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace Generators {

/**
 * @class GraphAnnotationIds
 * @brief Hands out one CUDA graph annotation id per distinct execution shape.
 *
 * A captured graph re-issues the launches recorded at capture time together with the device
 * pointers and launch dimensions it recorded, and never re-runs the operators' host-side code.
 * Anything that changes either must therefore land on its own id: tensor shapes, grid dimensions,
 * and the addresses of any buffer the step reads or writes. Callers build a key out of exactly
 * those quantities.
 *
 * ORT stores captured graphs on the session, and the public API lets two Engines share one Model
 * and therefore one session. Ids come from a process-wide counter and are never recycled, so
 * neither a concurrent allocator nor one created after this one is destroyed can name a graph it
 * does not own. Owners must still release their ids, so the session drops those graphs before the
 * buffers they point at are freed.
 *
 * The budget bounds how much capture memory a long-running session can accumulate. Shapes past it
 * run eagerly, which is slower but always correct.
 */
class GraphAnnotationIds {
 public:
  static constexpr size_t kMaxCapturedShapes = 64;
  static constexpr size_t kMaxKeyComponents = 8;
  // Unused trailing components are zero, so keys of different widths never collide.
  using Key = std::array<size_t, kMaxKeyComponents>;

  // Returns a positive annotation id, or -1 once the budget is exhausted. ORT reserves -1 for
  // "do not capture or replay".
  int Id(const Key& key) {
    const auto existing = ids_.find(key);
    if (existing != ids_.end()) {
      return existing->second;
    }
    if (ids_.size() >= kMaxCapturedShapes) {
      return -1;
    }
    const int id = NextProcessWideId();
    if (id < 0) {
      return -1;
    }
    return ids_.emplace(key, id).first->second;
  }

  size_t size() const { return ids_.size(); }

  // Every id handed out so far, so the owner can release the graphs the session still holds.
  std::vector<int> AssignedIds() const {
    std::vector<int> assigned;
    assigned.reserve(ids_.size());
    for (const auto& entry : ids_) {
      assigned.push_back(entry.second);
    }
    return assigned;
  }

 private:
  static int NextProcessWideId() {
    static std::atomic<int64_t> next{1};
    const int64_t id = next.fetch_add(1, std::memory_order_relaxed);
    return id <= std::numeric_limits<int>::max() ? static_cast<int>(id) : -1;
  }

  std::map<Key, int> ids_;
};

}  // namespace Generators
