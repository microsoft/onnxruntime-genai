// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/long_prefill.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "ort_genai.h"
#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

constexpr int kRandomSeed = 42;

bool IsLongPrefillLength(int prompt_length_k) {
  return prompt_length_k == 32 || prompt_length_k == 64 || prompt_length_k == 128;
}

}  // namespace

static const ScenarioBase::Registrar<LongPrefillScenario> kRegistrar("long_prefill");

void LongPrefillScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!config.prompt_length_k) {
    throw std::invalid_argument("long_prefill requires prompt_length_k");
  }
  if (config.concurrency != 1) {
    throw std::invalid_argument("long_prefill requires concurrency=1");
  }
  if (config.generation_tokens != 1) {
    throw std::invalid_argument("long_prefill requires generation_tokens=1");
  }
  if (!IsLongPrefillLength(*config.prompt_length_k)) {
    throw std::invalid_argument("long_prefill requires prompt_length_k in [32,64,128]");
  }
}

ScenarioExecutionOutput LongPrefillScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string tag = "[" + Name() + "] ";
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

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  nlohmann::json prefill_ms_values = nlohmann::json::array();
  nlohmann::json prompt_processing_tps_values = nlohmann::json::array();
  const int total_runs = config.warmup_runs + config.measured_runs;
  int measured_run_index = 0;

  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = run < config.warmup_runs;
    auto params = OgaGeneratorParams::Create(*engineResources.model);
    params->SetSearchOption("max_length", static_cast<double>(prompt_token_count + static_cast<size_t>(config.generation_tokens)));
    params->SetSearchOption("random_seed", kRandomSeed);

    auto request = engineResources.engine->CreateRequest(*params);
    request->BeginTurn(
        prompt_tokens->SequenceData(0), prompt_token_count);

    const auto start = std::chrono::steady_clock::now();
    auto event_buffer = engineResources.engine->CreateEventBuffer(1);
    size_t event_count = 0;
    size_t consecutive_retries = 0;
    bool has_request_event = false;
    while (!has_request_event &&
           engineResources.engine->HasPendingRequests()) {
      event_count = engineResources.engine->Run(
          *event_buffer);
      if (event_count != 0) {
        has_request_event = RequireRequestEvent(
            *event_buffer->Get(0), Name(), consecutive_retries);
      }
    }
    const auto first_token = std::chrono::steady_clock::now();
    if (!has_request_event) {
      throw std::runtime_error("long_prefill did not produce a first token");
    }
    if ((event_buffer->Get(0)->Flags() & OgaEngineEventFlag_Token) == 0) {
      throw std::runtime_error("long_prefill did not produce a first token");
    }
    request->Close();

    const double prefill_ms = std::chrono::duration<double, std::milli>(first_token - start).count();
    const double prompt_processing_tps = static_cast<double>(prompt_token_count) / std::max(0.001, prefill_ms / 1000.0);
    std::cout << tag << (is_warmup ? "Warmup run " : "Run ")
              << (is_warmup ? run + 1 : measured_run_index + 1) << "/"
              << (is_warmup ? config.warmup_runs : config.measured_runs)
              << " complete: prefill_ms=" << prefill_ms
              << ", prompt_processing_tps=" << prompt_processing_tps << std::endl;

    if (is_warmup) {
      continue;
    }

    ttft_values.push_back(prefill_ms);
    prefill_ms_values.push_back(prefill_ms);
    prompt_processing_tps_values.push_back(prompt_processing_tps);
    output.requests.push_back({measured_run_index, prefill_ms, 0.0});
    ++measured_run_index;
  }

  memory.Stop();
  output.ttft_p5_ms = Percentile(ttft_values, 5.0);
  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.peak_device_memory_mb = BytesToMb(memory.PeakDeviceBytes());
  output.steady_state_device_memory_mb = BytesToMb(memory.SteadyStateDeviceBytes());
  output.scenario_metrics = {
      {"prefill_ms", std::move(prefill_ms_values)},
      {"prompt_processing_tps", std::move(prompt_processing_tps_values)},
      {"prompt_tokens", prompt_token_count},
      {"peak_host_memory_mb", BytesToMb(memory.PeakHostBytes())},
  };

  return output;
}

}  // namespace engine_benchmark