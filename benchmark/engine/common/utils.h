// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine_benchmark {

inline double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }

  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  return sum / static_cast<double>(values.size());
}

inline double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  const double t = rank - static_cast<double>(lo);
  return values[lo] + (values[hi] - values[lo]) * t;
}

inline uint64_t Max(const std::vector<uint64_t>& values) {
  if (values.empty()) {
    return 0;
  }
  return *std::max_element(values.begin(), values.end());
}

inline uint64_t SteadyStateAverageTail(const std::vector<uint64_t>& values) {
  if (values.empty()) {
    return 0;
  }

  const size_t tail = std::min<size_t>(10, values.size());
  const size_t start = values.size() - tail;
  uint64_t sum = 0;
  for (size_t i = start; i < values.size(); ++i) {
    sum += values[i];
  }
  return sum / static_cast<uint64_t>(tail);
}

}  // namespace engine_benchmark
