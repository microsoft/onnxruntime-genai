// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "threadpool.h"
#include "span.h"

namespace Generators {
namespace detail {

inline std::ptrdiff_t CheckedParallelSize(size_t size) {
  constexpr size_t max_size =
      static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (size > max_size) {
    throw std::overflow_error("Parallel operation size exceeds ptrdiff_t range");
  }
  return static_cast<std::ptrdiff_t>(size);
}

}  // namespace detail

namespace ParallelCost {

// Convert byte/compute hints into the conservative units used by the shared threshold.
constexpr double kSchedulingCostScale = 1.0 / 32.0;

template <typename T>
constexpr double CopyPerElement() noexcept {
  return static_cast<double>(2 * sizeof(T)) * kSchedulingCostScale;
}

template <typename SrcT, typename DstT>
constexpr double ConvertPerElement(double compute_cost = 1.0) noexcept {
  return (static_cast<double>(sizeof(SrcT) + sizeof(DstT)) + compute_cost) *
         kSchedulingCostScale;
}

template <typename T>
constexpr double FillPerElement() noexcept {
  return static_cast<double>(sizeof(T)) * kSchedulingCostScale;
}

}  // namespace ParallelCost

// Source and destination must not overlap; this helper has no memmove semantics.
template <typename T>
void ParallelCopy(ThreadPool* thread_pool, std::span<const T> source,
                  std::span<T> destination) {
  static_assert(std::is_trivially_copyable_v<T>,
                "ParallelCopy requires a trivially copyable element type");
  if (source.size() != destination.size()) {
    throw std::invalid_argument("ParallelCopy source and destination sizes must match");
  }
  ThreadPool::TryParallelFor(
      thread_pool, detail::CheckedParallelSize(source.size()),
      ParallelCost::CopyPerElement<T>(),
      [source, destination](std::ptrdiff_t first, std::ptrdiff_t last) {
        const auto count = last - first;
        if (count == 0)
          return;
        std::memcpy(destination.data() + first, source.data() + first,
                    static_cast<size_t>(count) * sizeof(T));
      });
}

template <typename T>
void ParallelCopy(ThreadPool* thread_pool, const T* source, T* destination,
                  size_t count) {
  ParallelCopy(thread_pool, std::span<const T>{source, count},
               std::span<T>{destination, count});
}

// Transform must be safe for concurrent invocation. A local copy is used by each range.
template <typename SrcT, typename DstT, typename Transform>
void ParallelTransform(ThreadPool* thread_pool, std::span<const SrcT> source,
                       std::span<DstT> destination,
                       double compute_cost_per_element, Transform transform) {
  if (source.size() != destination.size()) {
    throw std::invalid_argument(
        "ParallelTransform source and destination sizes must match");
  }
  ThreadPool::TryParallelFor(
      thread_pool, detail::CheckedParallelSize(source.size()),
      ParallelCost::ConvertPerElement<SrcT, DstT>(compute_cost_per_element),
      [source, destination, transform](std::ptrdiff_t first, std::ptrdiff_t last) {
        auto local_transform = transform;
        std::transform(source.data() + first, source.data() + last,
                       destination.data() + first, local_transform);
      });
}

template <typename SrcT, typename DstT, typename Transform>
void ParallelTransform(ThreadPool* thread_pool, const SrcT* source,
                       DstT* destination, size_t count,
                       double compute_cost_per_element, Transform transform) {
  ParallelTransform(thread_pool, std::span<const SrcT>{source, count},
                    std::span<DstT>{destination, count},
                    compute_cost_per_element, std::move(transform));
}

// Transform must not read or write neighboring elements.
template <typename T, typename Transform>
void ParallelTransformInPlace(ThreadPool* thread_pool, std::span<T> values,
                              double compute_cost_per_element, Transform transform) {
  ThreadPool::TryParallelFor(
      thread_pool, detail::CheckedParallelSize(values.size()),
      ParallelCost::ConvertPerElement<T, T>(compute_cost_per_element),
      [values, transform](std::ptrdiff_t first, std::ptrdiff_t last) {
        auto local_transform = transform;
        std::transform(values.data() + first, values.data() + last,
                       values.data() + first, local_transform);
      });
}

template <typename T>
void ParallelFill(ThreadPool* thread_pool, std::span<T> destination,
                  const T& value) {
  ThreadPool::TryParallelFor(
      thread_pool, detail::CheckedParallelSize(destination.size()),
      ParallelCost::FillPerElement<T>(),
      [destination, value](std::ptrdiff_t first, std::ptrdiff_t last) {
        std::fill(destination.data() + first, destination.data() + last, value);
      });
}

}  // namespace Generators
