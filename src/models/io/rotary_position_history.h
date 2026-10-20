// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace Generators {

class RotaryPositionHistory {
 public:
  RotaryPositionHistory(size_t batch_size, size_t capacity,
                        int64_t max_position = std::numeric_limits<int64_t>::max())
      : batch_size_{batch_size}, capacity_{capacity}, max_position_{max_position} {
    if (batch_size == 0 || capacity == 0 ||
        batch_size > std::numeric_limits<size_t>::max() / 3 / capacity || max_position < 0) {
      throw std::runtime_error("Rotary position history requires a positive, representable batch and capacity");
    }
    values_.resize(3 * batch_size * capacity);
  }

  std::span<int64_t> Values() { return values_; }

  void Update(std::span<const int64_t> positions, size_t start, size_t length, size_t beams = 1) {
    if (beams == 0 || batch_size_ % beams != 0 || length == 0 ||
        start > capacity_ || length > capacity_ - start ||
        positions.size() != 3 * (batch_size_ / beams) * length) {
      throw std::runtime_error("Current T/H/W positions do not fit the rotary history");
    }
    const size_t source_batch = batch_size_ / beams;
    if (std::any_of(positions.begin(), positions.end(),
                    [&](int64_t value) { return value < 0 || value > max_position_; })) {
      throw std::runtime_error("Rotary coordinates exceed the available cache table");
    }
    for (size_t plane = 0; plane < 3; ++plane) {
      for (size_t row = 0; row < batch_size_; ++row) {
        auto source = positions.subspan((plane * source_batch + row / beams) * length, length);
        std::copy(source.begin(), source.end(), values_.begin() + (plane * batch_size_ + row) * capacity_ + start);
      }
    }
    Forecast(start + length);
  }

  void CopyFrom(std::span<const int64_t> source, size_t source_capacity) {
    if (source_capacity < capacity_ || source_capacity > std::numeric_limits<size_t>::max() / 3 / batch_size_ ||
        source.size() != 3 * batch_size_ * source_capacity) {
      throw std::runtime_error("Main and MTP rotary position histories have incompatible shapes");
    }
    for (size_t row = 0; row < 3 * batch_size_; ++row) {
      std::copy_n(source.begin() + row * source_capacity, capacity_, values_.begin() + row * capacity_);
    }
  }

  void Forecast(size_t length) {
    if (length > capacity_) {
      throw std::runtime_error("Rotary history rewind exceeds its capacity");
    }
    for (size_t row = 0; row < batch_size_; ++row) {
      int64_t next = 0;
      if (length != 0) {
        for (size_t plane = 0; plane < 3; ++plane) {
          auto begin = values_.begin() + (plane * batch_size_ + row) * capacity_;
          const int64_t maximum = *std::max_element(begin, begin + length);
          if (maximum == std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("Rotary coordinates exceed the supported integer range");
          }
          next = std::max(next, maximum + 1);
        }
      }
      next = std::min(next, max_position_);
      for (size_t plane = 0; plane < 3; ++plane) {
        auto begin = values_.begin() + (plane * batch_size_ + row) * capacity_;
        for (size_t token = length; token < capacity_; ++token) {
          const size_t offset = token - length;
          begin[token] = offset > static_cast<size_t>(max_position_ - next)
                             ? max_position_
                             : next + static_cast<int64_t>(offset);
        }
      }
    }
  }

 private:
  size_t batch_size_;
  size_t capacity_;
  int64_t max_position_;
  std::vector<int64_t> values_;
};

}  // namespace Generators
