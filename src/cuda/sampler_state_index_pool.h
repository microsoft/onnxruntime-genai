// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace Generators {

class SamplerStateIndexPool {
 public:
  template <typename Prepare>
  int Acquire(Prepare&& prepare) {
    const bool reusing = !free_indices_.empty();
    const int index = reusing ? free_indices_.back() : size_;
    const int required_size = reusing ? size_ : size_ + 1;

    // Release must remain noexcept, so reserve its future slot before any external preparation can
    // publish an acquired index. A throwing preparation leaves every pool counter unchanged.
    free_indices_.reserve(static_cast<size_t>(required_size));
    std::forward<Prepare>(prepare)(index, required_size);

    if (reusing) {
      free_indices_.pop_back();
    } else {
      size_ = required_size;
    }
    return index;
  }

  template <typename Prepare, typename Create>
  auto AcquireOwned(Prepare&& prepare, Create&& create) {
    const int index = Acquire(std::forward<Prepare>(prepare));
    try {
      return std::forward<Create>(create)(index);
    } catch (...) {
      Release(index);
      throw;
    }
  }

  void Release(int index) noexcept {
    assert(index >= 0 && index < size_);
    assert(std::find(free_indices_.begin(), free_indices_.end(), index) ==
           free_indices_.end());
    assert(free_indices_.size() < free_indices_.capacity());
    free_indices_.push_back(index);
  }

  int Size() const noexcept { return size_; }
  size_t FreeCount() const noexcept { return free_indices_.size(); }
  size_t ActiveCount() const noexcept {
    return static_cast<size_t>(size_) - free_indices_.size();
  }

 private:
  std::vector<int> free_indices_;
  int size_{};
};

}  // namespace Generators
