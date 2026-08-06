// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

}  // namespace engine_benchmark
