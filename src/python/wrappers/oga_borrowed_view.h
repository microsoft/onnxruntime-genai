// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "oga_object.h"
#include <cstddef>

namespace OgaPy {

// Generic borrowed array view that keeps the parent object alive
// Used for int32_t arrays returned from C API that are owned by parent objects
template<typename Parent, typename T>
struct BorrowedArrayView : OgaObject {
  BorrowedArrayView(Parent* parent, const T* data, size_t size)
      : parent_(parent), data_(data), size_(size) {
    if (!parent) {
      throw std::invalid_argument("BorrowedArrayView: parent cannot be null");
    }
    if (!data && size > 0) {
      throw std::invalid_argument("BorrowedArrayView: data cannot be null when size > 0");
    }
    intrusive_inc_ref(parent_);
  }

  ~BorrowedArrayView() override {
    intrusive_dec_ref(parent_);
  }

  const T* data() const { return data_; }
  size_t size() const { return size_; }

  // Element access
  const T& operator[](size_t index) const {
    if (index >= size_) {
      throw std::out_of_range("BorrowedArrayView: index out of range");
    }
    return data_[index];
  }

  // Iterator support
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }

  // Prevent copying to avoid double-free issues
  BorrowedArrayView(const BorrowedArrayView&) = delete;
  BorrowedArrayView& operator=(const BorrowedArrayView&) = delete;

  // Allow move operations
  BorrowedArrayView(BorrowedArrayView&& other) noexcept
      : OgaObject(std::move(other)),
        parent_(other.parent_),
        data_(other.data_),
        size_(other.size_) {
    other.parent_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  BorrowedArrayView& operator=(BorrowedArrayView&& other) noexcept {
    if (this != &other) {
      intrusive_dec_ref(parent_);
      parent_ = other.parent_;
      data_ = other.data_;
      size_ = other.size_;
      other.parent_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

private:
  Parent* parent_;
  const T* data_;
  size_t size_;
};

// Forward declarations of wrapper types
struct OgaSequences;
struct OgaGenerator;
struct OgaTokenizer;

// Type aliases for specific borrowed array views  
using SequenceDataView = BorrowedArrayView<OgaSequences, int32_t>;
using GeneratorSequenceDataView = BorrowedArrayView<OgaGenerator, int32_t>;
using NextTokensView = BorrowedArrayView<OgaGenerator, int32_t>;
using EosTokenIdsView = BorrowedArrayView<OgaTokenizer, int32_t>;

} // namespace OgaPy
