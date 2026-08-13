// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace engine_benchmark {

struct ScenarioConfig {
  std::string scenario{"decode_baseline"};
  int concurrency{1};
  int prompt_length_k{4};
  bool synthetic{true};
  std::string model_path;
  std::string execution_provider{"cuda"};
  std::string execution_provider_library;
  int generation_tokens{128};
  int measured_runs{1};
};

struct BenchmarkContext {
  std::string ort_version{"unknown"};
  std::string genai_version{"unknown"};
};

struct RequestMetrics {
  int request_id{0};
  double ttft_ms{0.0};
  double inter_token_latency_ms{0.0};
};

struct ScenarioExecutionOutput {
  std::vector<RequestMetrics> requests;
  double ttft_p50_ms{0.0};
  double ttft_p95_ms{0.0};
  double inter_token_latency_p50_ms{0.0};
  double inter_token_latency_p95_ms{0.0};
  double peak_device_memory_mb{0.0};
  double steady_state_device_memory_mb{0.0};
  nlohmann::json scenario_metrics{nlohmann::json::object()};
};

inline double BytesToMb(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
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

}  // namespace engine_benchmark
