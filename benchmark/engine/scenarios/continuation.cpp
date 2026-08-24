// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/continuation.h"

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
constexpr int kContinuationTurns = 3;

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 4 || concurrency == 8;
}

std::unique_ptr<OgaSequences> MakeSequences(const std::vector<int32_t>& tokens) {
  auto sequences = OgaSequences::Create();
  sequences->Append(tokens.data(), tokens.size());
  return sequences;
}

}  // namespace

static const ScenarioBase::Registrar<ContinuationScenario> kRegistrar("continuation");

void ContinuationScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("continuation requires concurrency in [4,8]");
  }
}

ScenarioExecutionOutput ContinuationScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string tag = "[" + Name() + "] ";
  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', concurrency=" << config.concurrency
            << ", continuation_turns=" << kContinuationTurns
            << ", prompt_length_k=" << config.prompt_length_k
            << ", generation_tokens=" << config.generation_tokens << std::endl;

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
  auto base_prompt = BuildRulerPromptTokens(config.prompt_length_k, *tokenizer, prompt_random);
  const size_t base_prompt_count = base_prompt->SequenceCount(0);

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  std::vector<double> inter_token_latency_values;
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();
  nlohmann::json final_context_tokens_values = nlohmann::json::array();
  const int total_runs = config.warmup_runs + config.measured_runs;
  int measured_run_index = 0;

  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = run < config.warmup_runs;

    // session_tokens is the evolving context for each concurrent logical session.
    // It begins as the base prompt and grows after each generated continuation turn.
    std::vector<std::vector<int32_t>> session_tokens(static_cast<size_t>(config.concurrency));
    for (auto& tokens : session_tokens) {
      tokens.assign(base_prompt->SequenceData(0), base_prompt->SequenceData(0) + base_prompt_count);
    }

    const auto run_start = std::chrono::steady_clock::now();
    size_t generated_tokens = 0;

    for (int turn = 0; turn < kContinuationTurns; ++turn) {
      // Each turn resubmits the previous turn's generated tokens as appended context,
      // exercising continuation/session-cache reuse under concurrent request pressure.
      std::vector<std::unique_ptr<OgaGeneratorParams>> params;
      std::vector<std::unique_ptr<OgaRequest>> requests;
      std::vector<std::unique_ptr<OgaSequences>> prompts;
      std::vector<std::vector<int32_t>> request_tokens(static_cast<size_t>(config.concurrency));
      std::vector<size_t> prompt_counts(static_cast<size_t>(config.concurrency));
      std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
      std::vector<std::chrono::steady_clock::time_point> last_token_time(static_cast<size_t>(config.concurrency));
      std::vector<std::vector<double>> request_itl_ms(static_cast<size_t>(config.concurrency));
      params.reserve(static_cast<size_t>(config.concurrency));
      requests.reserve(static_cast<size_t>(config.concurrency));
      prompts.reserve(static_cast<size_t>(config.concurrency));

      // Submit one request per active session for this turn. Each request receives the full
      // context accumulated so far, and max_length permits exactly one more generated segment.
      const auto turn_start = std::chrono::steady_clock::now();
      for (int i = 0; i < config.concurrency; ++i) {
        const size_t request_index = static_cast<size_t>(i);
        prompt_counts[request_index] = session_tokens[request_index].size();
        params.emplace_back(OgaGeneratorParams::Create(*model));
        params.back()->SetSearchOption(
            "max_length", static_cast<double>(prompt_counts[request_index] + config.generation_tokens));
        params.back()->SetSearchOption("random_seed", kRandomSeed + turn);
        prompts.push_back(MakeSequences(session_tokens[request_index]));
        request_tokens[request_index] = session_tokens[request_index];
        requests.emplace_back(OgaRequest::Create(*params.back()));
        requests.back()->AddTokens(*prompts.back());
        requests.back()->SetOpaqueData(&request_tokens[request_index]);
        engine->Add(*requests.back());
      }

      // Drain the engine for this turn. Opaque data maps each ready request back to its
      // per-session token vector so unseen tokens can be appended as they arrive.
      while (auto ready_request = engine->Step()) {
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
          // First unseen token establishes TTFT for this turn; later unseen tokens contribute
          // inter-token latency samples for the same logical session.
          tokens->push_back(ready_request->GetUnseenToken());
          ++generated_tokens;
          const double elapsed_ms = std::chrono::duration<double, std::milli>(now - turn_start).count();
          if (first_token_ms[request_index] < 0.0) {
            first_token_ms[request_index] = elapsed_ms;
          } else {
            request_itl_ms[request_index].push_back(
                std::chrono::duration<double, std::milli>(now - last_token_time[request_index]).count());
          }
          last_token_time[request_index] = now;
        }
      }

      for (int i = 0; i < config.concurrency; ++i) {
        const size_t request_index = static_cast<size_t>(i);
        const size_t actual_generated = request_tokens[request_index].size() - prompt_counts[request_index];
        const bool completed = actual_generated == static_cast<size_t>(config.generation_tokens);

        // Carry the appended tokens into the next turn, which is the core continuation behavior.
        session_tokens[request_index] = std::move(request_tokens[request_index]);

        if (is_warmup) {
          continue;
        }

        const double ttft_ms = std::max(0.0, first_token_ms[request_index]);
        ttft_values.push_back(ttft_ms);
        auto& samples = request_itl_ms[request_index];
        const int request_id =
            (measured_run_index * kContinuationTurns + turn) * config.concurrency + static_cast<int>(request_index);
        output.requests.push_back({request_id, ttft_ms, Percentile(samples, 50.0), completed});
        inter_token_latency_values.insert(inter_token_latency_values.end(), samples.begin(), samples.end());
      }
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double run_elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(generated_tokens) / std::max(0.001, run_elapsed_ms / 1000.0);

    std::cout << tag << (is_warmup ? "Warmup run " : "Run ")
              << (is_warmup ? run + 1 : measured_run_index + 1)
              << " complete: generated_tokens=" << generated_tokens
              << ", elapsed_ms=" << run_elapsed_ms
              << ", tokens_per_s=" << tokens_per_s << std::endl;

    if (is_warmup) {
      continue;
    }

    // Run-level metrics describe the complete multi-turn continuation exchange.
    e2e_ms_values.push_back(run_elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);
    final_context_tokens_values.push_back(session_tokens.empty() ? 0 : session_tokens.front().size());
    ++measured_run_index;
  }

  output.ttft_p5_ms = Percentile(ttft_values, 5.0);
  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);
  memory.Stop();

  // Summaries aggregate all measured turns; scenario_metrics keeps continuation-specific context.
  output.peak_device_memory_mb = BytesToMb(memory.PeakDeviceBytes());
  output.steady_state_device_memory_mb = BytesToMb(memory.SteadyStateDeviceBytes());
  output.scenario_metrics = {
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
      {"base_prompt_tokens", base_prompt_count},
      {"continuation_turns", kContinuationTurns},
      {"final_context_tokens", std::move(final_context_tokens_values)},
      {"peak_host_memory_mb", BytesToMb(memory.PeakHostBytes())},
  };
  return output;
}

}  // namespace engine_benchmark
