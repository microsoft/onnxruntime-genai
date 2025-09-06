// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

//================================================================================
// Welford's online algorithm is used here to compute mean and variance in one
// pass. This is more efficient and numerically stable than the standard
// two-pass approach (first pass for mean, second for sum of squared
// differences).
//================================================================================
struct SampleStats {
  size_t n = 0;
  double M1 = 0.0;  // Mean
  double M2 = 0.0;  // Sum of squares of differences from the current mean

  // Welford's online algorithm to update stats with a new value
  void update(double x) {
    n++;
    double delta = x - M1;
    M1 += delta / n;
    double delta2 = x - M1;
    M2 += delta * delta2;
  }

  // Static factory method to compute stats for a whole vector
  static SampleStats compute(const std::vector<double>& v) {
    SampleStats s;
    for (double x : v) {
      s.update(x);
    }
    return s;
  }

  double mean() const { return M1; }

  double variance() const {
    if (n < 2) return 0.0;
    return M2 / (n - 1);  // Sample variance
  }

  double stdev() const { return std::sqrt(variance()); }
};

// Mean
inline double mean(const std::vector<double>& v) {
  if (v.empty()) return NAN;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Standard deviation
inline double stdev(const std::vector<double>& v) {
  if (v.size() < 2)
    return 0.0;  // Stdev is 0 for a single point, undefined for empty.
  return SampleStats::compute(v).stdev();
}

// Percentile (0–100)
inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return NAN;
  if (p < 0.0 || p > 100.0) return NAN;

  std::sort(v.begin(), v.end());
  if (p == 100.0) {
    return v.back();
  }
  double idx = (p / 100.0) * (v.size() - 1);
  size_t i = static_cast<size_t>(idx);
  double frac = idx - i;
  if (i + 1 < v.size()) return v[i] * (1.0 - frac) + v[i + 1] * frac;
  return v[i];
}