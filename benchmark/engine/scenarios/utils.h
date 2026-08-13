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
  uint64_t peak_device_memory_bytes{0};
  uint64_t steady_state_device_memory_bytes{0};
  nlohmann::json scenario_metrics{nlohmann::json::object()};
};

// Loads the model, issues `concurrency` identical synthetic-prompt requests per run, and records
// TTFT / inter-token latency / throughput. Shared by every scenario that only varies the prompt
// and generation sizes; `log_tag` prefixes the progress output.
ScenarioExecutionOutput RunEngineWorkload(const ScenarioConfig& config, const std::string& log_tag);

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
