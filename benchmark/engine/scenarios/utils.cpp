// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/utils.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ort_genai.h"

namespace fs = std::filesystem;

namespace engine_benchmark {
namespace {

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

ScenarioExecutionOutput RunEngineWorkload(const ScenarioConfig& config, const std::string& log_tag) {
  const std::string tag = "[" + log_tag + "] ";

  std::cout << tag << "Execute start: model_path='" << config.model_path
            << "', provider='" << config.execution_provider
            << "', provider_library='" << config.execution_provider_library
            << "', concurrency=" << config.concurrency
            << ", measured_runs=" << config.measured_runs
            << ", prompt_length_k=" << config.prompt_length_k
            << ", generation_tokens=" << config.generation_tokens
            << std::endl;
  std::cout << tag << "Current working directory: " << fs::current_path().string() << std::endl;

  std::cout << tag << "Resolving model path..." << std::endl;
  const std::string resolved_model_path = ResolveModelPath(config.model_path);
  std::cout << tag << "Resolved model path: " << resolved_model_path << std::endl;

  std::cout << tag << "Creating OGA config..." << std::endl;
  auto oga_config = OgaConfig::Create(resolved_model_path.c_str());
  std::cout << tag << "OGA config created. Clearing providers..." << std::endl;
  oga_config->ClearProviders();
  std::cout << tag << "Providers cleared. Appending provider '"
            << config.execution_provider << "'..." << std::endl;
  oga_config->AppendProvider(config.execution_provider.c_str());
  std::cout << tag << "Provider appended." << std::endl;

  std::cout << tag << "About to create OgaModel..." << std::endl;
  auto model = OgaModel::Create(*oga_config);
  std::cout << tag << "OgaModel created." << std::endl;

  std::cout << tag << "About to create OgaTokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  std::cout << tag << "OgaTokenizer created." << std::endl;

  std::cout << tag << "About to create OgaEngine..." << std::endl;
  auto engine = OgaEngine::Create(*model);
  std::cout << tag << "OgaEngine created." << std::endl;

  std::cout << tag << "Building synthetic prompt..." << std::endl;
  const std::string prompt = BuildPromptText(config.prompt_length_k);
  std::cout << tag << "Prompt size (chars): " << prompt.size() << std::endl;

  std::cout << tag << "Encoding prompt..." << std::endl;
  auto prompt_sequences = OgaSequences::Create();
  tokenizer->Encode(prompt.c_str(), *prompt_sequences);

  const size_t prompt_token_count = prompt_sequences->SequenceCount(0);
  std::cout << tag << "Prompt token count: " << prompt_token_count << std::endl;

  ScenarioExecutionOutput output;
  std::vector<double> ttft_values;
  std::vector<double> inter_token_latency_values;
  nlohmann::json outputs = nlohmann::json::array();
  nlohmann::json e2e_ms_values = nlohmann::json::array();
  nlohmann::json tokens_per_s_values = nlohmann::json::array();

  for (int run = 0; run < config.measured_runs; ++run) {
    std::cout << tag << "Run " << run + 1 << "/" << config.measured_runs
              << ": creating requests..." << std::endl;

    std::vector<std::unique_ptr<OgaGeneratorParams>> params;
    std::vector<std::unique_ptr<OgaRequest>> requests;
    std::vector<std::vector<int32_t>> request_tokens(static_cast<size_t>(config.concurrency));
    std::vector<double> first_token_ms(static_cast<size_t>(config.concurrency), -1.0);
    std::vector<std::chrono::steady_clock::time_point> last_token_time(static_cast<size_t>(config.concurrency));

    params.reserve(static_cast<size_t>(config.concurrency));
    requests.reserve(static_cast<size_t>(config.concurrency));

    for (int i = 0; i < config.concurrency; ++i) {
      params.emplace_back(OgaGeneratorParams::Create(*model));
      const size_t max_length = prompt_token_count + static_cast<size_t>(config.generation_tokens);
      params.back()->SetSearchOption(
          "max_length", static_cast<double>(max_length));

      request_tokens[static_cast<size_t>(i)].assign(
          prompt_sequences->SequenceData(0), prompt_sequences->SequenceData(0) + prompt_token_count);

      requests.emplace_back(OgaRequest::Create(*params.back()));
      requests.back()->AddTokens(*prompt_sequences);
      requests.back()->SetOpaqueData(&request_tokens[static_cast<size_t>(i)]);
      engine->Add(*requests.back());

      std::cout << tag << "Run " << run + 1
                << ": added request " << i
                << " (max_length=" << max_length
                << ", prompt_tokens=" << prompt_token_count << ")"
                << std::endl;
    }

    std::cout << tag << "Run " << run + 1 << ": entering step loop..." << std::endl;
    const auto run_start = std::chrono::steady_clock::now();
    size_t generated_tokens = 0;
    size_t step_events = 0;

    while (auto ready_request = engine->Step()) {
      ++step_events;
      const auto now = std::chrono::steady_clock::now();
      auto* tokens = reinterpret_cast<std::vector<int32_t>*>(ready_request->GetOpaqueData());
      if (tokens == nullptr) {
        std::cout << tag << "ERROR: ready_request has null opaque data at run " << run + 1
                  << ", step_event=" << step_events << std::endl;
        throw std::runtime_error(log_tag + ": null opaque data from request");
      }

      const auto base_addr = reinterpret_cast<std::uintptr_t>(request_tokens.data());
      const auto ptr_addr = reinterpret_cast<std::uintptr_t>(tokens);
      const auto end_addr =
          reinterpret_cast<std::uintptr_t>(request_tokens.data() + request_tokens.size());

      if (ptr_addr < base_addr || ptr_addr >= end_addr) {
        std::cout << tag << "ERROR: opaque token pointer out of range at run " << run + 1
                  << ", step_event=" << step_events
                  << ", ptr=" << ptr_addr
                  << ", base=" << base_addr
                  << ", end=" << end_addr
                  << std::endl;
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
          inter_token_latency_values.push_back(
              std::chrono::duration<double, std::milli>(now - last_token_time[request_index]).count());
        }

        last_token_time[request_index] = now;
      }

      if (step_events <= 8 || step_events % 50 == 0) {
        std::cout << tag << "Run " << run + 1
                  << ": processed step_event=" << step_events
                  << ", request_index=" << request_index
                  << ", generated_tokens_so_far=" << generated_tokens
                  << std::endl;
      }
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double run_elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
    const double tokens_per_s = static_cast<double>(generated_tokens) / std::max(0.001, run_elapsed_ms / 1000.0);

    std::cout << tag << "Run " << run + 1
              << " complete: step_events=" << step_events
              << ", generated_tokens=" << generated_tokens
              << ", elapsed_ms=" << run_elapsed_ms
              << ", tokens_per_s=" << tokens_per_s
              << std::endl;

    e2e_ms_values.push_back(run_elapsed_ms);
    tokens_per_s_values.push_back(tokens_per_s);

    for (int i = 0; i < config.concurrency; ++i) {
      const double ttft_ms = std::max(0.0, first_token_ms[static_cast<size_t>(i)]);
      ttft_values.push_back(ttft_ms);
      output.requests.push_back({run * config.concurrency + i, ttft_ms, Percentile(inter_token_latency_values, 50.0)});

      const auto& tokens = request_tokens[static_cast<size_t>(i)];
      const size_t output_tokens = tokens.size() > prompt_token_count ? tokens.size() - prompt_token_count : 0;
      std::string output_text;
      if (output_tokens > 0) {
        std::cout << tag << "Run " << run + 1
                  << ": decoding output for request " << i
                  << " with output_tokens=" << output_tokens
                  << std::endl;
        const auto decoded = tokenizer->Decode(tokens.data() + prompt_token_count, output_tokens);
        output_text = static_cast<const char*>(decoded);
      }

      outputs.push_back({
          {"run_index", run},
          {"request_id", run * config.concurrency + i},
          {"text", output_text},
      });
    }

    std::cout << tag << "Run " << run + 1 << ": request post-processing complete." << std::endl;
  }

  output.ttft_p50_ms = Percentile(ttft_values, 50.0);
  output.ttft_p95_ms = Percentile(ttft_values, 95.0);
  output.inter_token_latency_p50_ms = Percentile(inter_token_latency_values, 50.0);
  output.inter_token_latency_p95_ms = Percentile(inter_token_latency_values, 95.0);
  output.scenario_metrics = {
      {"outputs", std::move(outputs)},
      {"e2e_ms", std::move(e2e_ms_values)},
      {"tokens_per_s", std::move(tokens_per_s_values)},
      {"prompt_tokens", prompt_token_count},
  };

  std::cout << tag << "Execute complete: ttft_p50_ms=" << output.ttft_p50_ms
            << ", ttft_p95_ms=" << output.ttft_p95_ms
            << ", inter_token_latency_p50_ms=" << output.inter_token_latency_p50_ms
            << ", inter_token_latency_p95_ms=" << output.inter_token_latency_p95_ms
            << std::endl;

  return output;
}

}  // namespace engine_benchmark
