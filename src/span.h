// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifndef USE_CXX17
#include <span>
#else
namespace std {

template <typename T>
struct span {
  span() = default;
  span(T* p, size_t length) : p_{p}, length_{length} {}

  span(const span<std::remove_const_t<T>>& s) : p_{const_cast<T*>(s.data())}, length_{s.size()} {}
  span(std::vector<std::remove_const_t<T>>& s) : p_{const_cast<T*>(s.data())}, length_{s.size()} {}

  bool empty() const { return length_ == 0; }

  size_t size() const { return length_; }
  size_t size_bytes() const { return length_ * sizeof(T); }

  T* data() { return p_; }
  const T* data() const { return p_; }

  T& operator[](size_t index) { return p_[index]; }

  T& back() { return p_[length_ - 1]; }
  T back() const { return p_[length_ - 1]; }

  T* begin() { return p_; }
  T* end() { return p_ + length_; }

  span subspan(size_t index, size_t length) { return span(p_ + index, length); }
  span<const T> subspan(size_t index, size_t length) const { return span(p_ + index, length); }

 private:
  T* p_{};
  size_t length_{};
};
}  // namespace std
#endif
