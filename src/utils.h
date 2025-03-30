// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <assert.h>
#include <atomic>
#include <string>
#include <memory>

#include "span.h"

namespace Ort {
struct Allocator;
}

namespace Generators {
struct Search;
struct Sequences;
struct GeneratorParams;

template <typename T>
struct OrtxPtr {
  ~OrtxPtr() { OrtxDispose(&p_); }
  T** Address() {
    assert(!p_);
    return &p_;
  }
  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* p_{};
};

size_t SizeOf(ONNXTensorElementDataType type);

int64_t ElementCountFromShape(std::span<const int64_t> shape);

// NOTE: Until all compilers fully support C++23 and give us float16_t and bfloat16_t, we have to do it ourselves:

// Slower fp16 to fp32 conversion that handles NaN and Inf (useful for debugging vs runtime conversion)
float Float16ToFloat32(uint16_t v);
float BFloat16ToFloat32(uint16_t v);

inline float ToFloat32(Ort::Float16_t v) { return Float16ToFloat32(v); }
inline float ToFloat32(Ort::BFloat16_t v) { return BFloat16ToFloat32(v); }

// Fast fp16<->fp32 conversions that do not handle NaN and Inf but are fast (as these are not typical values)
float FastFloat16ToFloat32(const uint16_t x);
uint16_t FastFloat32ToFloat16(float v);

// A shared_ptr based type that we expose through our C API should inherit from this type.
// ExternalAddRef must be called when returning an object through the C API
// ExternalRelease must be called on the C API destroy method
template <typename T>
struct ExternalRefCounted {
  void ExternalAddRef() {
    if (++ref_count_ == 1)  // First reference?
      external_owner_ = static_cast<T*>(this)->shared_from_this();
  }
  void ExternalRelease() {
    if (--ref_count_ == 0)
      external_owner_ = nullptr;
  }

 private:
  std::shared_ptr<T> external_owner_;  // shared_ptr to ourselves to keep us alive
  std::atomic<int> ref_count_{};       // C API refcount (can't use only the shared_ptr)
};

namespace Location {
struct CPU {};
struct GPU {};
}  // namespace Location

template <typename T>
struct cpu_span : std::span<T> {
  using std::span<T>::span;
  explicit cpu_span(std::span<T> v) : std::span<T>(v) {}
};
template <typename T>
struct gpu_span : std::span<T> {
  using std::span<T>::span;
  explicit gpu_span(std::span<T> v) : std::span<T>(v) {}
};

template <typename T>
void copy(std::span<const T> source, std::span<T> dest) {
  assert(source.size() == dest.size());
  std::copy(source.begin(), source.end(), dest.begin());
}

template <typename T>
std::unique_ptr<T[]> AllocateArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p = new T[count];
  if (p_span)
    *p_span = std::span<T>(p, count);
  return std::unique_ptr<T[]>{p};
}

}  // namespace Generators
