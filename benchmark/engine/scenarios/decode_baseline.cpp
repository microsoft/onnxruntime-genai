// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/decode_baseline.h"

#include <algorithm>
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

struct RequestRunState {
  std::vector<int32_t> tokens;
  double first_token_ms{-1.0};
  std::chrono::steady_clock::time_point last_token_time;
  std::vector<double> inter_token_latency_ms;
};

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8;
}

}  // namespace

// Self-registers with ScenarioBase::Create so the dispatcher picks this up automatically.
static const ScenarioBase::Registrar<DecodeBaselineScenario> kRegistrar("decode_baseline");

void DecodeBaselineScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!config.prompt_length_k) {
    throw std::invalid_argument("decode_baseline requires prompt_length_k");
  }
  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("decode_baseline requires concurrency in [1,2,4,8]");
  }
}

ScenarioExecutionOutput DecodeBaselineScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string log_tag = Name();
  const std::string tag = "[" + log_tag + "] ";
  const int prompt_length_k = config.prompt_length_k.value();

  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', concurrency=" << config.concurrency
            << ", measured_runs=" << config.measured_runs
            << ", prompt_length_k=" << prompt_length_k
            << ", generation_tokens=" << config.generation_tokens
            << std::endl;

  MemorySampler memory;
  memory.Start();
  auto engineResources = CreateEngineResources(config);

  std::mt19937 prompt_random(kRandomSeed);
  auto prompt_tokens = BuildRulerPromptTokens(prompt_length_k, *engineResources.tokenizer, prompt_random);

  const size_t prompt_token_count = prompt_tokens->SequenceCount(0);
  std::cout << tag << "Prompt token count: " << prompt_token_count << std::endl;

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  // Inter-token gaps from measured runs only; warmup samples are never added here.
  std::vector<double> inter_token_latency_values;
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();

  const int total_runs = config.warmup_runs + config.measured_runs;
  int measured_run_index = 0;

  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = run < config.warmup_runs;

    std::vector<std::unique_ptr<OgaRequestOptions>> request_options;
    std::vector<std::unique_ptr<OgaTurnOptions>> turn_options;
    std::vector<RequestRunState> request_states(static_cast<size_t>(config.concurrency));
    std::vector<std::unique_ptr<OgaRequest>> requests;
    std::unordered_map<const OgaRequest*, RequestRunState*>
        request_states_by_request;

    request_options.reserve(static_cast<size_t>(config.concurrency));
    turn_options.reserve(static_cast<size_t>(config.concurrency));
    requests.reserve(static_cast<size_t>(config.concurrency));

    // Start timing before submission so TTFT captures arrival-to-first-token, including admission.
    const auto run_start = std::chrono::steady_clock::now();
    size_t generated_tokens = 0;

    for (int i = 0; i < config.concurrency; ++i) {
      request_options.emplace_back(OgaRequestOptions::Create());
      const size_t max_session_tokens =
          prompt_token_count + static_cast<size_t>(config.generation_tokens);
      request_options.back()->SetMaxSessionTokens(max_session_tokens);

      auto& request_state = request_states[static_cast<size_t>(i)];
      request_state.tokens.assign(
          prompt_tokens->SequenceData(0), prompt_tokens->SequenceData(0) + prompt_token_count);

      requests.emplace_back(
          engineResources.engine->CreateRequest(request_options.back().get()));
      request_states_by_request.emplace(
          requests.back().get(), &request_state);
      turn_options.push_back(requests.back()->CreateTurnOptions());
      if (engineResources.uses_dynamic_batching) {
        turn_options.back()->SetSeed(kRandomSeed);
      }
      requests.back()->BeginTurn(
          prompt_tokens->SequenceData(0), prompt_token_count,
          turn_options.back().get());
    }

    auto event_buffer = engineResources.engine->CreateEventBuffer(
        static_cast<size_t>(config.concurrency));
    size_t consecutive_retries = 0;
    while (engineResources.engine->HasPendingRequests()) {
      const size_t event_count = engineResources.engine->Run(
          *event_buffer);
      const auto now = std::chrono::steady_clock::now();
      for (size_t event_index = 0; event_index < event_count; ++event_index) {
        const auto& event = *event_buffer->Get(event_index);
        if (!RequireRequestEvent(
                event, Name(), consecutive_retries)) {
          continue;
        }
        const auto state_it =
            request_states_by_request.find(&event.Request()->get());
        if (state_it == request_states_by_request.end()) {
          throw std::runtime_error(
              log_tag + ": event request has no benchmark state");
        }
        auto* request_state = state_it->second;

        if ((event.Flags() & OgaEngineEventFlag_Token) != 0) {
          request_state->tokens.push_back(event.Token());
          ++generated_tokens;

          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(
                  now - run_start)
                  .count();
          if (request_state->first_token_ms < 0.0) {
            request_state->first_token_ms = elapsed_ms;
          } else {
            request_state->inter_token_latency_ms.push_back(
                std::chrono::duration<double, std::milli>(
                    now - request_state->last_token_time)
                    .count());
          }

          request_state->last_token_time = now;
        }
      }
    }
    for (const auto& request : requests) {
      request->Close();
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double run_elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(generated_tokens) / std::max(0.001, run_elapsed_ms / 1000.0);

    std::cout << tag << (is_warmup ? "Warmup run " : "Run ")
              << (is_warmup ? run + 1 : measured_run_index + 1) << "/"
              << (is_warmup ? config.warmup_runs : config.measured_runs)
              << " complete: generated_tokens=" << generated_tokens
              << ", elapsed_ms=" << run_elapsed_ms
              << ", tokens_per_s=" << tokens_per_s
              << std::endl;

    if (is_warmup) {
      continue;
    }

    e2e_ms_values.push_back(run_elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);

    for (int i = 0; i < config.concurrency; ++i) {
      auto& request_state = request_states[static_cast<size_t>(i)];
      const double ttft_ms = std::max(0.0, request_state.first_token_ms);
      ttft_values.push_back(ttft_ms);

      // Each request record reports its own median inter-token latency, not the global one.
      auto& request_samples = request_state.inter_token_latency_ms;
      const size_t actual_generated = request_state.tokens.size() - prompt_token_count;
      const bool completed = actual_generated == static_cast<size_t>(config.generation_tokens);
      output.requests.push_back(
          {measured_run_index * config.concurrency + i, ttft_ms, Percentile(request_samples, 50.0), completed});

      // Summary ITL percentiles are token-weighted: each inter-token gap contributes one sample.
      inter_token_latency_values.insert(
          inter_token_latency_values.end(), request_samples.begin(), request_samples.end());
    }

    ++measured_run_index;
  }

  memory.Stop();
  output.ttft_p5_ms = Percentile(ttft_values, 5.0);
  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);
  output.peak_device_memory_mb = BytesToMb(memory.PeakDeviceBytes());
  output.steady_state_device_memory_mb = BytesToMb(memory.SteadyStateDeviceBytes());

  output.scenario_metrics = {
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
      {"prompt_tokens", prompt_token_count},
      {"peak_host_memory_mb", BytesToMb(memory.PeakHostBytes())},
  };

  std::cout << tag << "Execute complete: ttft_p50_ms=" << output.ttft_p50_ms
            << ", ttft_p95_ms=" << output.ttft_p95_ms
            << ", inter_token_latency_p50_ms=" << output.inter_token_latency_p50_ms
            << ", inter_token_latency_p95_ms=" << output.inter_token_latency_p95_ms
            << std::endl;

  return output;
}

}  // namespace engine_benchmark
