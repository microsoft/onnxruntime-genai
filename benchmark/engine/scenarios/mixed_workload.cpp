// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/mixed_workload.h"

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
constexpr int kPrefillPromptLengthK = 128;
constexpr int kPrefillGenerationTokens = 1;

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 4 || concurrency == 8;
}

}  // namespace

static const ScenarioBase::Registrar<MixedWorkloadScenario> kRegistrar("mixed_workload");

void MixedWorkloadScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!config.prompt_length_k) {
    throw std::invalid_argument("mixed_workload requires prompt_length_k");
  }
  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("mixed_workload requires concurrency in [4,8]");
  }
}

ScenarioExecutionOutput MixedWorkloadScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string tag = "[" + Name() + "] ";
  const int prompt_length_k = config.prompt_length_k.value();
  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', concurrency=" << config.concurrency
            << ", prompt_length_k=" << prompt_length_k
            << ", prefill_prompt_length_k=" << kPrefillPromptLengthK
            << ", prefill_generation_tokens=" << kPrefillGenerationTokens
            << ", generation_tokens=" << config.generation_tokens << std::endl;

  MemorySampler memory;
  memory.Start();
  auto engineResources = CreateEngineResources(config);

  // Use one long prompt to exercise prefill while shorter prompts represent active decodes.
  std::mt19937 prompt_random(kRandomSeed);
  auto decode_prompt = BuildRulerPromptTokens(prompt_length_k, *engineResources.tokenizer, prompt_random);
  auto prefill_prompt = BuildRulerPromptTokens(kPrefillPromptLengthK, *engineResources.tokenizer, prompt_random);
  const size_t decode_prompt_count = decode_prompt->SequenceCount(0);
  const size_t prefill_prompt_count = prefill_prompt->SequenceCount(0);

  ScenarioExecutionOutput output;
  std::vector<double> prefill_ttft_ms_values;
  std::vector<double> decode_ttft_values;
  std::vector<double> inter_token_latency_values;
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();
  const int total_runs = config.warmup_runs + config.measured_runs;
  int measured_run_index = 0;

  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = run < config.warmup_runs;

    // Co-schedule one 128K prefill with active decode requests in the same engine workload.
    std::vector<std::unique_ptr<OgaGeneratorParams>> params;
    std::vector<std::unique_ptr<OgaRequest>> requests;
    std::vector<std::vector<int32_t>> request_tokens(static_cast<size_t>(config.concurrency));
    std::vector<size_t> prompt_counts(static_cast<size_t>(config.concurrency), decode_prompt_count);
    std::vector<int> target_generation_tokens(static_cast<size_t>(config.concurrency), config.generation_tokens);
    std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
    std::vector<std::chrono::steady_clock::time_point> last_token_time(static_cast<size_t>(config.concurrency));
    std::vector<std::vector<double>> request_itl_ms(static_cast<size_t>(config.concurrency));
    params.reserve(static_cast<size_t>(config.concurrency));
    requests.reserve(static_cast<size_t>(config.concurrency));

    const auto run_start = std::chrono::steady_clock::now();
    size_t generated_tokens = 0;
    for (int i = 0; i < config.concurrency; ++i) {
      const auto& prompt = i == 0 ? prefill_prompt : decode_prompt;
      const size_t prompt_count = i == 0 ? prefill_prompt_count : decode_prompt_count;
      // Keep the long-prefill request bounded to one generated token so mixed runs do not
      // exceed model context/KV limits and intermittently fault CUDA under high pressure.
      const int generation_tokens = i == 0 ? kPrefillGenerationTokens : config.generation_tokens;
      prompt_counts[static_cast<size_t>(i)] = prompt_count;
      target_generation_tokens[static_cast<size_t>(i)] = generation_tokens;
      params.emplace_back(OgaGeneratorParams::Create(*engineResources.model));
      params.back()->SetSearchOption("max_length", static_cast<double>(prompt_count + generation_tokens));
      params.back()->SetSearchOption("random_seed", kRandomSeed);
      request_tokens[static_cast<size_t>(i)].assign(prompt->SequenceData(0), prompt->SequenceData(0) + prompt_count);
      requests.emplace_back(OgaRequest::Create(*params.back()));
      requests.back()->AddTokens(*prompt);
      requests.back()->SetOpaqueData(&request_tokens[static_cast<size_t>(i)]);
      engineResources.engine->Add(*requests.back());
    }

    // Measure whether the long prefill delays decode first-token and inter-token latency.
    while (auto ready_request = engineResources.engine->Step()) {
      const auto now = std::chrono::steady_clock::now();
      auto* tokens = reinterpret_cast<std::vector<int32_t>*>(ready_request->GetOpaqueData());
      if (tokens == nullptr) {
        throw std::runtime_error(Name() + ": null opaque data from request");
      }
      const auto base_addr = reinterpret_cast<std::uintptr_t>(request_tokens.data());
      const auto ptr_addr = reinterpret_cast<std::uintptr_t>(tokens);
      const auto end_addr = reinterpret_cast<std::uintptr_t>(request_tokens.data() + request_tokens.size());
      if (ptr_addr < base_addr || ptr_addr >= end_addr) {
        throw std::runtime_error(Name() + ": opaque data pointer not in request_tokens");
      }
      const size_t request_index = (ptr_addr - base_addr) / sizeof(std::vector<int32_t>);
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
    const double elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(generated_tokens) / std::max(0.001, elapsed_ms / 1000.0);
    std::cout << tag << (is_warmup ? "Warmup run " : "Run ")
              << (is_warmup ? run + 1 : measured_run_index + 1) << " complete: elapsed_ms=" << elapsed_ms
              << ", tokens_per_s=" << tokens_per_s << std::endl;
    if (is_warmup) {
      continue;
    }

    // Aggregate the mixed workload's throughput and per-request prefill/decode timing.
    e2e_ms_values.push_back(elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);
    for (int i = 0; i < config.concurrency; ++i) {
      const size_t request_index = static_cast<size_t>(i);
      const double ttft_ms = std::max(0.0, first_token_ms[request_index]);
      const bool is_prefill = i == 0;
      if (is_prefill) {
        prefill_ttft_ms_values.push_back(ttft_ms);
      } else {
        decode_ttft_values.push_back(ttft_ms);
      }
      auto& samples = request_itl_ms[request_index];
      const bool completed = request_tokens[request_index].size() - prompt_counts[request_index] ==
                             static_cast<size_t>(target_generation_tokens[request_index]);
      output.requests.push_back({measured_run_index * config.concurrency + i,
                                 ttft_ms,
                                 Percentile(samples, 50.0),
                                 completed,
                                 is_prefill ? "prefill" : "decode"});
      inter_token_latency_values.insert(inter_token_latency_values.end(), samples.begin(), samples.end());
    }
    ++measured_run_index;
  }

  memory.Stop();
  output.ttft_p5_ms = Percentile(decode_ttft_values, 5.0);
  output.ttft_p50_ms = Percentile(decode_ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(decode_ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);
  output.peak_device_memory_mb = BytesToMb(memory.PeakDeviceBytes());
  output.steady_state_device_memory_mb = BytesToMb(memory.SteadyStateDeviceBytes());
  output.scenario_metrics = {
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
      {"prefill_ttft_ms", std::move(prefill_ttft_ms_values)},
      {"decode_prompt_tokens", decode_prompt_count},
      {"prefill_prompt_tokens", prefill_prompt_count},
      {"prefill_generation_tokens", kPrefillGenerationTokens},
      {"peak_host_memory_mb", BytesToMb(memory.PeakHostBytes())},
  };
  return output;
}

}  // namespace engine_benchmark
