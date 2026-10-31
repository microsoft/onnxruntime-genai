// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Generators {

// Fixed-capacity request-id index for transaction-owned state. Storage is allocated at
// construction; all ownership publication and removal operations are allocation-free.
class RequestIndex {
 public:
  explicit RequestIndex(size_t max_entries)
      : max_entries_{max_entries} {
    if (max_entries > (std::numeric_limits<size_t>::max() - 1) / 2) {
      throw std::overflow_error("Request index capacity overflow.");
    }
    size_t table_size = 1;
    const size_t required = max_entries * 2 + 1;
    while (table_size < required) {
      if (table_size > std::numeric_limits<size_t>::max() / 2) {
        throw std::overflow_error("Request index table size overflow.");
      }
      table_size *= 2;
    }
    entries_.resize(table_size);
  }

  RequestIndex(RequestIndex&& other) noexcept
      : entries_{std::move(other.entries_)},
        max_entries_{std::exchange(other.max_entries_, 0)},
        size_{std::exchange(other.size_, 0)} {}
  RequestIndex& operator=(RequestIndex&&) = delete;
  RequestIndex(const RequestIndex&) = delete;
  RequestIndex& operator=(const RequestIndex&) = delete;

  std::optional<size_t> Find(const void* request_id) const noexcept {
    if (!request_id) {
      return std::nullopt;
    }
    size_t index = Hash(request_id) & (entries_.size() - 1);
    for (size_t probe = 0; probe < entries_.size(); ++probe) {
      const auto& entry = entries_[index];
      if (entry.state == State::Empty) {
        return std::nullopt;
      }
      if (entry.state == State::Occupied &&
          entry.request_id == request_id) {
        return entry.value;
      }
      index = (index + 1) & (entries_.size() - 1);
    }
    return std::nullopt;
  }

  bool Insert(const void* request_id, size_t value) noexcept {
    if (!request_id || size_ >= max_entries_) {
      return false;
    }
    size_t index = Hash(request_id) & (entries_.size() - 1);
    for (size_t probe = 0; probe < entries_.size(); ++probe) {
      auto& entry = entries_[index];
      if (entry.state == State::Occupied &&
          entry.request_id == request_id) {
        return false;
      }
      if (entry.state == State::Empty) {
        entry = Entry{request_id, value, State::Occupied};
        ++size_;
        return true;
      }
      index = (index + 1) & (entries_.size() - 1);
    }
    return false;
  }

  bool Erase(const void* request_id) noexcept {
    if (!request_id) {
      return false;
    }
    size_t index = Hash(request_id) & (entries_.size() - 1);
    for (size_t probe = 0; probe < entries_.size(); ++probe) {
      auto& entry = entries_[index];
      if (entry.state == State::Empty) {
        return false;
      }
      if (entry.state == State::Occupied &&
          entry.request_id == request_id) {
        entry = {};
        --size_;

        // Closing the following probe cluster keeps misses bounded by the live load rather than
        // leaving tombstones that accumulate for the lifetime of a serving process.
        size_t displaced_index = (index + 1) & (entries_.size() - 1);
        while (entries_[displaced_index].state == State::Occupied) {
          const Entry displaced = entries_[displaced_index];
          entries_[displaced_index] = {};
          --size_;
          if (!Insert(displaced.request_id, displaced.value)) {
            std::terminate();
          }
          displaced_index =
              (displaced_index + 1) & (entries_.size() - 1);
        }
        return true;
      }
      index = (index + 1) & (entries_.size() - 1);
    }
    return false;
  }

  void Clear() noexcept {
    for (auto& entry : entries_) {
      entry = {};
    }
    size_ = 0;
  }

  size_t Size() const noexcept { return size_; }
  size_t Capacity() const noexcept { return max_entries_; }

 private:
  enum class State : uint8_t {
    Empty,
    Occupied,
  };

  struct Entry {
    const void* request_id{};
    size_t value{};
    State state{};
  };

  static size_t Hash(const void* request_id) noexcept {
    size_t value = reinterpret_cast<uintptr_t>(request_id);
    value ^= value >> (sizeof(size_t) * 4);
    value *= static_cast<size_t>(0xff51afd7ed558ccdULL);
    value ^= value >> (sizeof(size_t) * 4);
    return value;
  }

  std::vector<Entry> entries_;
  size_t max_entries_{};
  size_t size_{};
};

}  // namespace Generators
