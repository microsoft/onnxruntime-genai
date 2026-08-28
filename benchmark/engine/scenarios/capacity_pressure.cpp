// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/capacity_pressure.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ort_genai.h"
#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

constexpr int kRandomSeed = 42;
constexpr std::array<int, 8> kPromptLengthsK = {4, 4, 32, 32, 48, 64, 96, 128};

}  // namespace

static const ScenarioBase::Registrar<CapacityPressureScenario> kRegistrar("capacity_pressure");

void CapacityPressureScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (config.prompt_length_k) {
    throw std::invalid_argument("capacity_pressure does not accept prompt_length_k; prompt lengths are fixed");
  }

  if (config.concurrency != static_cast<int>(kPromptLengthsK.size())) {
    throw std::invalid_argument("capacity_pressure requires concurrency=8");
  }

  if (config.generation_tokens != 1) {
    throw std::invalid_argument("capacity_pressure requires generation_tokens=1");
  }
}

ScenarioExecutionOutput CapacityPressureScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string tag = "[" + Name() + "] ";
  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', concurrency=" << config.concurrency
            << ", generation_tokens=" << config.generation_tokens << std::endl;

  // Use one shared model and engine for all pressure requests. The point of this
  // scenario is to ask the engine to admit several large requests at once.
  MemorySampler memory;
  memory.Start();
  auto engineResources = CreateEngineResources(config);

  // Build the fixed pressure set once: eight prompts ramp from 4K toward 128K.
  // This intentionally pushes the KV/cache budget instead of measuring normal throughput.
  std::mt19937 prompt_random(kRandomSeed);
  std::vector<std::unique_ptr<OgaSequences>> pressure_prompts;
  pressure_prompts.reserve(kPromptLengthsK.size());
  for (int prompt_length_k : kPromptLengthsK) {
    pressure_prompts.push_back(BuildRulerPromptTokens(prompt_length_k, *engineResources.tokenizer, prompt_random));
  }

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();
  nlohmann::json admitted_values = nlohmann::json::array();
  nlohmann::json rejected_values = nlohmann::json::array();
  const int total_runs = config.warmup_runs + config.measured_runs;
  int measured_run_index = 0;

  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = run < config.warmup_runs;
    std::vector<std::unique_ptr<OgaGeneratorParams>> params;
    std::vector<std::unique_ptr<OgaRequest>> requests;
    std::vector<std::vector<int32_t>> request_tokens(static_cast<size_t>(config.concurrency));
    std::vector<size_t> prompt_counts(static_cast<size_t>(config.concurrency));
    std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
    std::vector<bool> admitted(static_cast<size_t>(config.concurrency), false);
    std::vector<bool> rejected(static_cast<size_t>(config.concurrency), false);
    std::unordered_map<OgaRequest*, size_t> request_indices;
    params.reserve(static_cast<size_t>(config.concurrency));
    requests.reserve(static_cast<size_t>(config.concurrency));

    const auto run_start = std::chrono::steady_clock::now();
    int admitted_count = 0;
    int rejected_count = 0;
    size_t generated_tokens = 0;

    // Queue every pressure prompt independently. The turn API defers capacity decisions until
    // execution, so request-specific failure events are benchmark data rather than immediate
    // scenario failures. Preemption is deliberately out of scope for now.
    for (int i = 0; i < config.concurrency; ++i) {
      const size_t request_index = static_cast<size_t>(i);
      const auto& prompt = pressure_prompts[request_index];
      const size_t prompt_count = prompt->SequenceCount(0);
      prompt_counts[request_index] = prompt_count;
      params.emplace_back(OgaGeneratorParams::Create(*engineResources.model));
      params.back()->SetSearchOption("max_length", static_cast<double>(prompt_count + config.generation_tokens));
      params.back()->SetSearchOption("random_seed", kRandomSeed);
      request_tokens[request_index].assign(prompt->SequenceData(0), prompt->SequenceData(0) + prompt_count);

      try {
        auto request = engineResources.engine->CreateRequest(*params.back());
        request->BeginTurn(std::span<const int32_t>{
            prompt->SequenceData(0), prompt_count});
        request_indices.emplace(request.get(), request_index);
        requests.push_back(std::move(request));
      } catch (const std::exception& ex) {
        rejected[request_index] = true;
        ++rejected_count;
        std::cout << tag << "Turn rejected for request " << i
                  << " prompt_length_k=" << kPromptLengthsK[request_index]
                  << ": " << ex.what() << std::endl;
      }
    }

    // Execution phase: caller-buffered events reveal which queued requests were admitted to model
    // execution and which were permanently unserviceable. Completed requests are closed promptly
    // so deferred peers can reuse their cache capacity.
    std::vector<OgaEngineEvent> event_storage(
        static_cast<size_t>(config.concurrency));
    while (engineResources.engine->HasPendingRequests()) {
      const auto events = engineResources.engine->Run(event_storage);
      const auto now = std::chrono::steady_clock::now();
      for (const auto& event : events) {
        if (!event.request) {
          if ((event.flags & OgaEngineEventFlag_Failed) != 0) {
            throw std::runtime_error(
                Name() + ": Engine failed without a request-specific outcome");
          }
          continue;
        }

        const auto request_it = request_indices.find(event.request);
        if (request_it == request_indices.end()) {
          throw std::runtime_error(
              Name() + ": event request has no pressure benchmark state");
        }
        const size_t request_index = request_it->second;

        if ((event.flags & OgaEngineEventFlag_Failed) != 0) {
          if (!rejected[request_index]) {
            rejected[request_index] = true;
            ++rejected_count;
          }
        } else if (!admitted[request_index]) {
          admitted[request_index] = true;
          ++admitted_count;
        }

        if ((event.flags & OgaEngineEventFlag_Token) != 0) {
          // The first emitted token establishes TTFT for an admitted pressure request.
          request_tokens[request_index].push_back(event.token);
          ++generated_tokens;
          if (first_token_ms[request_index] < 0.0) {
            first_token_ms[request_index] =
                std::chrono::duration<double, std::milli>(
                    now - run_start)
                    .count();
          }
        }

        if ((event.flags & OgaEngineEventFlag_TurnFinished) != 0) {
          event.request->Close();
        }
      }
    }
    for (const auto& request : requests) {
      request->Close();
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(generated_tokens) / std::max(0.001, elapsed_ms / 1000.0);

    std::cout << tag << (is_warmup ? "Warmup run " : "Run ")
              << (is_warmup ? run + 1 : measured_run_index + 1)
              << " complete: admitted=" << admitted_count
              << ", rejected=" << rejected_count
              << ", elapsed_ms=" << elapsed_ms
              << ", tokens_per_s=" << tokens_per_s << std::endl;

    if (is_warmup) {
      continue;
    }

    e2e_ms_values.push_back(elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);
    admitted_values.push_back(admitted_count);
    rejected_values.push_back(rejected_count);

    // Request-level output includes only requests admitted to execution. Rejected requests are
    // reported separately so the benchmark can distinguish unserviceable work from truncation.
    for (int i = 0; i < config.concurrency; ++i) {
      const size_t request_index = static_cast<size_t>(i);
      if (!admitted[request_index]) {
        continue;
      }

      const size_t actual_generated = request_tokens[request_index].size() - prompt_counts[request_index];
      const bool completed = actual_generated == static_cast<size_t>(config.generation_tokens);
      const double ttft_ms = std::max(0.0, first_token_ms[request_index]);
      ttft_values.push_back(ttft_ms);
      output.requests.push_back({measured_run_index * config.concurrency + i, ttft_ms, 0.0, completed});
    }

    ++measured_run_index;
  }

  // preemption_enabled=false is intentional: this version measures service and rejection under
  // pressure without preempting active work.
  memory.Stop();
  output.ttft_p5_ms = Percentile(ttft_values, 5.0);
  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.peak_device_memory_mb = BytesToMb(memory.PeakDeviceBytes());
  output.steady_state_device_memory_mb = BytesToMb(memory.SteadyStateDeviceBytes());
  output.scenario_metrics = {
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
      {"admitted_requests", std::move(admitted_values)},
      {"rejected_admissions", std::move(rejected_values)},
      {"prompt_lengths_k", kPromptLengthsK},
      {"preemption_enabled", false},
      {"peak_host_memory_mb", BytesToMb(memory.PeakHostBytes())},
  };
  return output;
}

}  // namespace engine_benchmark
