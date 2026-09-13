// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <map>

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
    return ids_.emplace(key, static_cast<int>(ids_.size() + 1)).first->second;
  }

  size_t size() const { return ids_.size(); }

 private:
  std::map<Key, int> ids_;
};

}  // namespace Generators
