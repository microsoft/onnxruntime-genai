// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifndef USE_CXX17
#pragma once
#include <span>
#else
#include <vector>
namespace std {

template <typename T>
struct span {
  span() = default;
  span(T* p, size_t length) : p_{p}, length_{length} {}

  span(const span<std::remove_const_t<T> >& s) : p_{const_cast<T*>(s.data())}, length_{s.size()} {}
  span(std::vector<std::remove_const_t<T> >& s) : p_{const_cast<T*>(s.data())}, length_{s.size()} {}
  template <auto N>
  span(std::array<std::remove_const_t<T>, N> s) : p_{const_cast<T*>(s.data())}, length_{s.size()} {}

  bool empty() const { return length_ == 0; }

  size_t size() const { return length_; }
  size_t size_bytes() const { return length_ * sizeof(T); }

  T* data() { return p_; }
  const T* data() const { return p_; }

  T& operator[](size_t index) const { return p_[index]; }

  T& back() const { return p_[length_ - 1]; }

  T* begin() const { return p_; }
  T* end() const { return p_ + length_; }

  span subspan(size_t index, size_t length) const { return span(p_ + index, length); }

 private:
  T* p_{};
  size_t length_{};
};
}  // namespace std
#endif
