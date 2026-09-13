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
  // Capture costs a stalled step and permanent capture memory, and there is no eviction, so a shape
  // has to prove it recurs before it may claim one of the budgeted ids. Without this a workload that
  // walks through many one-off shapes spends the whole budget on graphs it never replays, and never
  // recovers.
  static constexpr size_t kSightingsBeforeCapture = 3;
  // Bounds the bookkeeping for shapes that never earn an id.
  static constexpr size_t kMaxTrackedShapes = 4 * kMaxCapturedShapes;
  // Unused trailing components are zero, so keys of different widths never collide.
  using Key = std::array<size_t, kMaxKeyComponents>;

  // Returns a positive annotation id, or -1 while the shape is still unproven or once the budget is
  // exhausted. ORT reserves -1 for "do not capture or replay".
  int Id(const Key& key) {
    const auto existing = shapes_.find(key);
    if (existing != shapes_.end()) {
      Shape& shape = existing->second;
      if (shape.id > 0) {
        return shape.id;
      }
      if (++shape.sightings < kSightingsBeforeCapture || assigned_ >= kMaxCapturedShapes) {
        return -1;
      }
      shape.id = static_cast<int>(++assigned_);
      return shape.id;
    }
    if (shapes_.size() >= kMaxTrackedShapes) {
      // Shapes that never proved they recur are dropped so a steady shape arriving later can still
      // be tracked. Captured shapes keep their ids.
      std::erase_if(shapes_, [](const auto& entry) { return entry.second.id == 0; });
    }
    shapes_.emplace(key, Shape{/*sightings=*/1, /*id=*/0});
    return -1;
  }

  size_t size() const { return assigned_; }

 private:
  struct Shape {
    size_t sightings{};
    int id{};
  };

  std::map<Key, Shape> shapes_;
  size_t assigned_{};
};

}  // namespace Generators
