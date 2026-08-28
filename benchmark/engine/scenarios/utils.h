// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ort_genai.h"

namespace engine_benchmark {

constexpr int kDefaultWarmupRuns = 5;
constexpr int kDefaultMeasuredRuns = 20;
constexpr int kDefaultGenerationTokens = 64;

struct ScenarioConfig {
  std::string scenario{"decode_baseline"};
  int concurrency{1};
  std::optional<int> prompt_length_k;
  std::string model_path;
  std::string execution_provider{"cuda"};
  std::string execution_provider_library;
  int generation_tokens{kDefaultGenerationTokens};
  int warmup_runs{kDefaultWarmupRuns};
  int measured_runs{kDefaultMeasuredRuns};
};

struct BenchmarkContext {
  std::string ort_version{"unknown"};
  std::string genai_version{"unknown"};
  std::string cuda_plugin_ep_version{"unknown"};
};

struct RequestMetrics {
  int request_id{0};
  double ttft_ms{0.0};
  double inter_token_latency_ms{0.0};
  // False when the request generated fewer tokens than config.generation_tokens.
  bool completed{true};
  std::string role{"request"};
};

struct ScenarioExecutionOutput {
  std::vector<RequestMetrics> requests;
  double ttft_p5_ms{0.0};
  double ttft_p50_ms{0.0};
  double ttft_p95_ms{0.0};
  double inter_token_latency_p50_ms{0.0};
  double inter_token_latency_p95_ms{0.0};
  double peak_device_memory_mb{0.0};
  double steady_state_device_memory_mb{0.0};
  nlohmann::json scenario_metrics{nlohmann::json::object()};
};

/// Polls device and host memory usage on a background thread.
///
/// Device usage is read from NVML, which is loaded lazily so the benchmark still runs on machines
/// without an NVIDIA driver; usage attributed to this process is preferred and the device-wide
/// delta from the pre-load baseline is used when the driver does not report per-process numbers.
/// All values are 0 when no source is available.
class MemorySampler {
 public:
  explicit MemorySampler(std::chrono::milliseconds interval = std::chrono::milliseconds(100));
  ~MemorySampler();

  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;

  /// Records the pre-load baseline and starts sampling. Call before the model is created.
  void Start();
  void Stop();

  uint64_t PeakDeviceBytes() const;
  /// Mean of the trailing samples, i.e. usage once allocations have settled.
  uint64_t SteadyStateDeviceBytes() const;
  uint64_t PeakHostBytes() const;

 private:
  void Loop();

  std::chrono::milliseconds interval_;
  std::vector<uint64_t> samples_;
  uint64_t baseline_device_bytes_{0};
  bool running_{false};
  mutable std::mutex mutex_;
  std::condition_variable stop_signal_;
  std::thread thread_;
};

struct EngineResources {
  explicit EngineResources(const ScenarioConfig& config);

  std::unique_ptr<OgaConfig> oga_config;
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
  std::unique_ptr<OgaEngine> engine;
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

inline bool RequireRequestEvent(
    const OgaEngineEvent& event, const std::string& scenario,
    size_t& consecutive_retries) {
  if (event.request) {
    consecutive_retries = 0;
    return true;
  }

  if ((event.flags & OgaEngineEventFlag_Failed) != 0) {
    throw std::runtime_error(
        scenario + ": Engine failed with error code " +
        std::to_string(static_cast<int>(event.error_code)));
  }
  if ((event.flags & (OgaEngineEventFlag_CapacityBlocked |
                      OgaEngineEventFlag_Retryable)) != 0) {
    constexpr size_t kMaxConsecutiveRetries = 100;
    if (++consecutive_retries > kMaxConsecutiveRetries) {
      throw std::runtime_error(
          scenario + ": Engine made no request progress after " +
          std::to_string(kMaxConsecutiveRetries) +
          " retryable events; last error code " +
          std::to_string(static_cast<int>(event.error_code)));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return false;
  }
  throw std::runtime_error(
      scenario + ": Engine returned an invalid request-less event with error code " +
      std::to_string(static_cast<int>(event.error_code)));
}

std::string ResolveModelPath(const std::string& model_path);
EngineResources CreateEngineResources(const ScenarioConfig& config);
std::unique_ptr<OgaSequences> BuildRulerPromptTokens(
    int prompt_length_k, const OgaTokenizer& tokenizer, std::mt19937& random);

}  // namespace engine_benchmark
