// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#ifndef USE_CXX17
#include <span>
#else

#include <array>
#include <vector>
#include <initializer_list>

namespace std {

template <typename T>
struct span {
  span() = default;
  span(T* p, size_t length) noexcept : p_{p}, length_{length} {}

  span(const span<T>& s) noexcept : p_{s.p_}, length_{s.length_} {}
  span(std::vector<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  template <size_t N>
  span(std::array<T, N>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  span(std::initializer_list<T> list) noexcept : p_(&list.begin()), length_(list.size()) {}

  bool empty() const noexcept { return length_ == 0; }

  size_t size() const noexcept { return length_; }
  size_t size_bytes() const noexcept { return length_ * sizeof(T); }

  T* data() noexcept { return p_; }
  const T* data() const noexcept { return p_; }

  T& operator[](size_t index) const noexcept { return p_[index]; }

  T& back() const noexcept { return p_[length_ - 1]; }

  T* begin() const noexcept { return p_; }
  T* end() const noexcept { return p_ + length_; }

  span subspan(size_t index, size_t length) const noexcept { return span(p_ + index, length); }

 private:
  T* p_{};
  size_t length_{};
};

template <class T>
struct span<const T> {
  span() = default;
  span(const T* p, size_t length) noexcept : p_{p}, length_{length} {}

  span(const span<const T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  span(const span<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  span(const std::vector<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  template <size_t N>
  span(const std::array<T, N>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  span(std::initializer_list<const T> list) noexcept : p_(list.begin()), length_(list.size()) {}

  bool empty() const noexcept { return length_ == 0; }

  size_t size() const noexcept { return length_; }
  size_t size_bytes() const noexcept { return length_ * sizeof(T); }

  const T* data() const noexcept { return p_; }

  const T& operator[](size_t index) const noexcept { return p_[index]; }

  const T& back() const noexcept { return p_[length_ - 1]; }

  const T* begin() const noexcept { return p_; }
  const T* end() const noexcept { return p_ + length_; }

  span subspan(size_t index, size_t length) const noexcept { return span(p_ + index, length); }

 private:
  const T* p_{};
  size_t length_{};
};


}  // namespace std

#endif
