// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#ifndef USE_CXX17
#include <span>
#else

#include <array>
#include <stdexcept>
#include <vector>

namespace std {
namespace generators_span {
template <typename T>
struct span {
  span() = default;
  constexpr span(T* p, size_t length) noexcept : p_{p}, length_{length} {}
  constexpr span(const span<T>& s) noexcept : p_{s.p_}, length_{s.length_} {}

  span(std::vector<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  template <size_t N>
  constexpr span(std::array<T, N>& s) noexcept : p_{s.data()}, length_{s.size()} {}

  constexpr bool empty() const noexcept { return length_ == 0; }

  constexpr size_t size() const noexcept { return length_; }
  constexpr size_t size_bytes() const noexcept { return length_ * sizeof(T); }

  constexpr T* data() const noexcept { return p_; }

  constexpr T& operator[](size_t index) const noexcept { return p_[index]; }

  constexpr T& back() const noexcept { return p_[length_ - 1]; }

  constexpr T* begin() const noexcept { return p_; }
  constexpr T* end() const noexcept { return p_ + length_; }

  span subspan(size_t index, size_t length) const {
    if (index > length_ || length > length_ - index)
      throw std::out_of_range("Requested subspan is out of range");

    return span(p_ + index, length);
  }

 private:
  T* p_{};
  size_t length_{};
};

template <class T>
struct span<const T> {
  span() = default;
  constexpr span(const T* p, size_t length) noexcept : p_{p}, length_{length} {}

  constexpr span(const span<const T>& s) noexcept : p_{s.p_}, length_{s.length_} {}
  constexpr span(const span<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}

  span(const std::vector<T>& s) noexcept : p_{s.data()}, length_{s.size()} {}
  template <size_t N>
  constexpr span(const std::array<T, N>& s) noexcept : p_{s.data()}, length_{s.size()} {}

  constexpr bool empty() const noexcept { return length_ == 0; }

  constexpr size_t size() const noexcept { return length_; }
  constexpr size_t size_bytes() const noexcept { return length_ * sizeof(T); }

  constexpr const T* data() const noexcept { return p_; }

  constexpr const T& operator[](size_t index) const noexcept { return p_[index]; }

  constexpr const T& back() const noexcept { return p_[length_ - 1]; }

  constexpr const T* begin() const noexcept { return p_; }
  constexpr const T* end() const noexcept { return p_ + length_; }

  span subspan(size_t index, size_t length) const {
    if (index > length_ || length > length_ - index)
      throw std::out_of_range("Requested subspan is out of range");

    return span(p_ + index, length);
  }

 private:
  const T* p_{};
  size_t length_{};
};
}  // namespace generators_span

using generators_span::span;

}  // namespace std

#endif
