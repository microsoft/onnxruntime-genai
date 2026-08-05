// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/decode_baseline.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/utils.h"
#include "ort_genai.h"

namespace fs = std::filesystem;

namespace engine_benchmark {
namespace {

bool IsAllowedConcurrency(int concurrency) {
  return concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8;
}

std::string ResolveModelPath(const std::string& model_path) {
  fs::path path(model_path);
  if (!fs::exists(path)) {
    throw std::invalid_argument("model_path does not exist: " + model_path);
  }

  return fs::absolute(path).string();
}

std::string BuildPromptText(int prompt_length_k) {
  std::string prompt = "Summarize the following benchmark context:\n";
  const int target_words = prompt_length_k * 750;
  prompt.reserve(static_cast<size_t>(target_words) * 8);

  for (int i = 0; i < target_words; ++i) {
    prompt += "token";
    prompt += std::to_string(i % 997);
    prompt += ' ';
  }

  prompt += "\nAnswer concisely.";
  return prompt;
}

}  // namespace

void DecodeBaselineScenario::ValidateConfig(const ScenarioConfig& config) const {
  ScenarioBase::ValidateConfig(config);

  if (!IsAllowedConcurrency(config.concurrency)) {
    throw std::invalid_argument("decode_baseline requires concurrency in [1,2,4,8]");
  }
  if (!config.synthetic) {
    throw std::invalid_argument("decode_baseline requires synthetic=true");
  }
}

ScenarioExecutionOutput DecodeBaselineScenario::Execute(const ScenarioConfig& config, const BenchmarkContext&) const {
  const std::string resolved_model_path = ResolveModelPath(config.model_path);
  auto oga_config = OgaConfig::Create(resolved_model_path.c_str());
  oga_config->ClearProviders();
  oga_config->AppendProvider(config.execution_provider.c_str());

  auto model = OgaModel::Create(*oga_config);
  auto tokenizer = OgaTokenizer::Create(*model);
  const std::string prompt = BuildPromptText(config.prompt_length_k);

  auto prompt_sequences = OgaSequences::Create();
  for (int i = 0; i < config.concurrency; ++i) {
    tokenizer->Encode(prompt.c_str(), *prompt_sequences);
  }

  const size_t prompt_token_count = prompt_sequences->SequenceCount(0);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", static_cast<double>(config.concurrency));
  params->SetSearchOption(
      "max_length", static_cast<double>(prompt_token_count + static_cast<size_t>(config.generation_tokens)));

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  std::vector<double> inter_token_latency_values;
  nlohmann::json outputs = nlohmann::json::array();
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();

  for (int run = 0; run < config.measured_runs; ++run) {
    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*prompt_sequences);

    std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
    const auto run_start = std::chrono::steady_clock::now();
    int emitted_steps = 0;

    while (!generator->IsDone() && emitted_steps < config.generation_tokens) {
      const auto step_start = std::chrono::steady_clock::now();
      generator->GenerateNextToken();
      const auto step_end = std::chrono::steady_clock::now();

      const double elapsed_ms = std::chrono::duration<double, std::milli>(step_end - run_start).count();
      const double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      if (emitted_steps > 0) {
        inter_token_latency_values.push_back(step_ms);
      }

      for (int i = 0; i < config.concurrency; ++i) {
        if (first_token_ms[static_cast<size_t>(i)] < 0.0) {
          first_token_ms[static_cast<size_t>(i)] = elapsed_ms;
        }
      }

      ++emitted_steps;
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double run_elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(config.generation_tokens) / std::max(0.001, run_elapsed_ms / 1000.0);

    e2e_ms_values.push_back(run_elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);

    for (int i = 0; i < config.concurrency; ++i) {
      const double ttft_ms = std::max(0.0, first_token_ms[static_cast<size_t>(i)]);
      ttft_values.push_back(ttft_ms);
      output.requests.push_back({run * config.concurrency + i, ttft_ms, Percentile(inter_token_latency_values, 50.0)});

      const size_t total_tokens = generator->GetSequenceCount(static_cast<size_t>(i));
      const int32_t* seq_data = generator->GetSequenceData(static_cast<size_t>(i));
      const size_t output_tokens = total_tokens > prompt_token_count ? total_tokens - prompt_token_count : 0;
      std::string output_text;
      if (output_tokens > 0) {
        const auto decoded = tokenizer->Decode(seq_data + prompt_token_count, output_tokens);
        output_text = static_cast<const char*>(decoded);
      }

      outputs.push_back({
          {"run_index", run},
          {"request_id", run * config.concurrency + i},
          {"text", output_text},
      });
    }
  }

  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);
  output.scenario_metrics = {
      {"outputs", std::move(outputs)},
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
  };

  return output;
}

}  // namespace engine_benchmark
