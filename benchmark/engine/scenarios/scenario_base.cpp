// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/scenario_base.h"

#include <exception>

namespace engine_benchmark {

std::map<std::string, ScenarioBase::Factory>& ScenarioBase::Factories() {
  static std::map<std::string, Factory> factories;
  return factories;
}

std::unique_ptr<ScenarioBase> ScenarioBase::Create(const std::string& name) {
  auto& factories = Factories();
  const auto it = factories.find(name);
  return it != factories.end() ? it->second() : nullptr;
}

void ScenarioBase::ValidateConfig(const ScenarioConfig& config) const {
  if (config.model_path.empty()) {
    throw std::invalid_argument("model_path must be provided.");
  }
  if (config.concurrency < 1) {
    throw std::invalid_argument("concurrency must be >= 1.");
  }
  if (config.prompt_length_k < 1) {
    throw std::invalid_argument("prompt_length_k must be >= 1.");
  }
  if (config.generation_tokens < 1) {
    throw std::invalid_argument("generation_tokens must be >= 1.");
  }
  if (config.measured_runs < 1) {
    throw std::invalid_argument("measured_runs must be >= 1.");
  }
}

nlohmann::json ScenarioBase::Run(const ScenarioConfig& config, const BenchmarkContext& context) const {
  nlohmann::json result;
  result["scenario"] = Name();
  result["config_metadata"] = {
      {"model_path", config.model_path},
      {"ort_version", context.ort_version},
      {"genai_version", context.genai_version},
      {"cuda_plugin_ep_version", context.cuda_plugin_ep_version},
      {"execution_provider", config.execution_provider},
      {"concurrency", config.concurrency},
      {"prompt_length_k", config.prompt_length_k},
      {"synthetic", config.synthetic},
      {"generation_tokens", config.generation_tokens},
      {"measured_runs", config.measured_runs}};

  try {
    ValidateConfig(config);
    const ScenarioExecutionOutput output = Execute(config, context);

    nlohmann::json raw_requests = nlohmann::json::array();
    int incomplete_count = 0;
    for (const auto& request : output.requests) {
      raw_requests.push_back({{"request_id", request.request_id},
                              {"ttft_ms", request.ttft_ms},
                              {"inter_token_latency_ms", request.inter_token_latency_ms},
                              {"completed", request.completed}});
      incomplete_count += request.completed ? 0 : 1;
    }

    // A request that returns fewer tokens than requested must not be reported as a success.
    if (incomplete_count > 0) {
      result["status"] = "failed";
      result["error"] = std::to_string(incomplete_count) + " of " + std::to_string(output.requests.size()) +
                         " request(s) completed with fewer tokens than requested";
    } else {
      result["status"] = "success";
      result["error"] = nullptr;
    }
    result["core_metrics"] = {
        {"summary",
         {{"ttft_ms", {{"p50", output.ttft_p50_ms}, {"p95", output.ttft_p95_ms}}},
          {"inter_token_latency_ms", {{"p50", output.inter_token_latency_p50_ms}, {"p95", output.inter_token_latency_p95_ms}}},
          {"peak_device_memory_mb", output.peak_device_memory_mb},
          {"steady_state_device_memory_mb", output.steady_state_device_memory_mb}}},
        {"raw_requests", std::move(raw_requests)}};
    result["scenario_metrics"] = output.scenario_metrics;
  } catch (const std::exception& ex) {
    result["status"] = "failed";
    result["error"] = ex.what();
    result["core_metrics"] = {{"summary", nlohmann::json::object()}, {"raw_requests", nlohmann::json::array()}};
    result["scenario_metrics"] = nlohmann::json::object();
  }

  return result;
}

}  // namespace engine_benchmark
