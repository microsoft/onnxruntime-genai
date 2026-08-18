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
#include <utility>
#include <vector>

#include "ort_genai.h"
#include "scenarios/utils.h"

namespace engine_benchmark {
namespace {

constexpr int kRandomSeed = 42;

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8;
}

}  // namespace

// Self-registers with ScenarioBase::Create so the dispatcher picks this up automatically.
static const ScenarioBase::Registrar<DecodeBaselineScenario> kRegistrar("decode_baseline");

void DecodeBaselineScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("decode_baseline requires concurrency in [1,2,4,8]");
  }
}

ScenarioExecutionOutput DecodeBaselineScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string log_tag = Name();
  const std::string tag = "[" + log_tag + "] ";

  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', concurrency=" << config.concurrency
            << ", measured_runs=" << config.measured_runs
            << ", prompt_length_k=" << config.prompt_length_k
            << ", generation_tokens=" << config.generation_tokens
            << std::endl;

  const std::string resolved_model_path = ResolveModelPath(config.model_path);

  MemorySampler memory;
  memory.Start();

  auto oga_config = OgaConfig::Create(resolved_model_path.c_str());
  oga_config->ClearProviders();
  oga_config->AppendProvider(config.execution_provider.c_str());
  auto model = OgaModel::Create(*oga_config);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto engine = OgaEngine::Create(*model);

  std::mt19937 prompt_random(kRandomSeed);
  auto prompt_tokens = BuildRulerPromptTokens(config.prompt_length_k, *tokenizer, prompt_random);

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

    std::vector<std::unique_ptr<OgaGeneratorParams>> params;
    std::vector<std::unique_ptr<OgaRequest>> requests;
    std::vector<std::vector<int32_t>> request_tokens(static_cast<size_t>(config.concurrency));
    std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
    std::vector<std::chrono::steady_clock::time_point> last_token_time(static_cast<size_t>(config.concurrency));
    std::vector<std::vector<double>> request_itl_ms(static_cast<size_t>(config.concurrency));

    params.reserve(static_cast<size_t>(config.concurrency));
    requests.reserve(static_cast<size_t>(config.concurrency));

    // Start timing before submission so TTFT captures arrival-to-first-token, including admission.
    const auto run_start = std::chrono::steady_clock::now();
    size_t generated_tokens = 0;

    for (int i = 0; i < config.concurrency; ++i) {
      params.emplace_back(OgaGeneratorParams::Create(*model));
      const size_t max_length = prompt_token_count + static_cast<size_t>(config.generation_tokens);
      params.back()->SetSearchOption(
          "max_length", static_cast<double>(max_length));
      params.back()->SetSearchOption("random_seed", kRandomSeed);

      request_tokens[static_cast<size_t>(i)].assign(
          prompt_tokens->SequenceData(0), prompt_tokens->SequenceData(0) + prompt_token_count);

      requests.emplace_back(OgaRequest::Create(*params.back()));
      requests.back()->AddTokens(*prompt_tokens);
      requests.back()->SetOpaqueData(&request_tokens[static_cast<size_t>(i)]);
      engine->Add(*requests.back());
    }

    while (auto ready_request = engine->Step()) {
      const auto now = std::chrono::steady_clock::now();
      auto* tokens = reinterpret_cast<std::vector<int32_t>*>(ready_request->GetOpaqueData());
      if (tokens == nullptr) {
        throw std::runtime_error(log_tag + ": null opaque data from request");
      }

      const auto base_addr = reinterpret_cast<std::uintptr_t>(request_tokens.data());
      const auto ptr_addr = reinterpret_cast<std::uintptr_t>(tokens);
      const auto end_addr =
          reinterpret_cast<std::uintptr_t>(request_tokens.data() + request_tokens.size());

      if (ptr_addr < base_addr || ptr_addr >= end_addr) {
        throw std::runtime_error(log_tag + ": opaque data pointer not in request_tokens");
      }

      const auto request_index =
          static_cast<size_t>((ptr_addr - base_addr) / sizeof(std::vector<int32_t>));

      while (ready_request->HasUnseenTokens()) {
        tokens->push_back(ready_request->GetUnseenToken());
        ++generated_tokens;

        const double elapsed_ms = std::chrono::duration<double, std::milli>(now - run_start).count();
        if (first_token_ms[request_index] < 0.0) {
          first_token_ms[request_index] = elapsed_ms;
        } else {
          request_itl_ms[request_index].push_back(
              std::chrono::duration<double, std::milli>(now - last_token_time[request_index]).count());
        }

        last_token_time[request_index] = now;
      }
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
      const double ttft_ms = std::max(0.0, first_token_ms[static_cast<size_t>(i)]);
      ttft_values.push_back(ttft_ms);

      // Each request record reports its own median inter-token latency, not the global one.
      auto& request_samples = request_itl_ms[static_cast<size_t>(i)];
      output.requests.push_back(
          {measured_run_index * config.concurrency + i, ttft_ms, Percentile(request_samples, 50.0)});

      // Summary ITL percentiles are token-weighted: each inter-token gap contributes one sample.
      inter_token_latency_values.insert(
          inter_token_latency_values.end(), request_samples.begin(), request_samples.end());
    }

    ++measured_run_index;
  }

  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);

  memory.Stop();
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
