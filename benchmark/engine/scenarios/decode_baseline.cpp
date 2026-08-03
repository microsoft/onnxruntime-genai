// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/decode_baseline.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "common/utils.h"
#include "ort_genai.h"

#ifdef _WIN32
#include <Windows.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#endif
  if (!IsAllowedConcurrency(concurrency)) {
    *error_message = "decode_baseline requires concurrency in [1,2,4,8]";
    return false;
  }
  if (prompt_k <= 0) {
    *error_message = "decode_baseline requires prompt length (1000s) > 0";
    return false;
  }
  if (model_path.empty()) {
    *error_message = "decode_baseline requires model_path in config";
    return false;
  }
  if (generation_tokens <= 0) {
    *error_message = "decode_baseline requires generation_tokens > 0";
    return false;
  }
  if (measured_runs <= 0) {
    *error_message = "decode_baseline requires measured_runs > 0";
    return false;
  }

  try {
    const std::string resolved_model_path = ResolveModelPath(model_path);
    auto config = OgaConfig::Create(resolved_model_path.c_str());
    config->ClearProviders();
    config->AppendProvider(execution_provider.c_str());

    auto model = OgaModel::Create(*config);
    auto tokenizer = OgaTokenizer::Create(*model);
    const std::string prompt = BuildPromptText(prompt_k, synthetic);

    auto prompt_sequences = OgaSequences::Create();
    for (int i = 0; i < concurrency; ++i) {
      tokenizer->Encode(prompt.c_str(), *prompt_sequences);
    }

    const size_t prompt_token_count = prompt_sequences->SequenceCount(0);
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("batch_size", static_cast<double>(concurrency));
    params->SetSearchOption("max_length", static_cast<double>(prompt_token_count + static_cast<size_t>(generation_tokens)));

    std::vector<double> ttft_values;
    std::vector<double> e2e_values;
    std::vector<double> tps_values;
    std::ostringstream raw;
    raw << "[";

    for (int run = 0; run < measured_runs; ++run) {
      auto generator = OgaGenerator::Create(*model, *params);
      generator->AppendTokenSequences(*prompt_sequences);

      std::vector<double> first_token_ms(static_cast<size_t>(concurrency), -1.0);
      std::vector<double> e2e_ms(static_cast<size_t>(concurrency), 0.0);

      const auto run_start = std::chrono::steady_clock::now();
      int emitted_steps = 0;
      while (!generator->IsDone() && emitted_steps < generation_tokens) {
        const auto step_end_before = std::chrono::steady_clock::now();
        generator->GenerateNextToken();
        const auto step_end = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(step_end - run_start).count();
        for (int i = 0; i < concurrency; ++i) {
          if (first_token_ms[static_cast<size_t>(i)] < 0.0) {
            first_token_ms[static_cast<size_t>(i)] = elapsed_ms;
          }
          (void)step_end_before;
        }

        ++emitted_steps;
      }

      const auto run_end = std::chrono::steady_clock::now();
      const double run_elapsed_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();

      for (int i = 0; i < concurrency; ++i) {
        e2e_ms[static_cast<size_t>(i)] = run_elapsed_ms;
        const double ttft_ms = std::max(0.0, first_token_ms[static_cast<size_t>(i)]);
        const double tps = static_cast<double>(generation_tokens) / std::max(0.001, run_elapsed_ms / 1000.0);

        ttft_values.push_back(ttft_ms);
        e2e_values.push_back(run_elapsed_ms);
        tps_values.push_back(tps);

        const size_t total_tokens = generator->GetSequenceCount(static_cast<size_t>(i));
        const int32_t* seq_data = generator->GetSequenceData(static_cast<size_t>(i));
        const size_t output_tokens = total_tokens > prompt_token_count ? (total_tokens - prompt_token_count) : 0;
        std::string output_text;
        if (output_tokens > 0) {
          const auto decoded = tokenizer->Decode(seq_data + prompt_token_count, output_tokens);
          output_text = static_cast<const char*>(decoded);
        }

        raw << "{"
            << "\"run_index\":" << run << ","
            << "\"request_id\":\"r" << run << "_" << i << "\","
            << "\"ttft_ms\":" << ttft_ms << ","
            << "\"e2e_ms\":" << run_elapsed_ms << ","
            << "\"generated_tokens_per_s\":" << tps << ","
            << "\"output\":\"" << EscapeJson(output_text) << "\""
            << "}";

        const bool last = (run == measured_runs - 1) && (i == concurrency - 1);
        if (!last) {
          raw << ",";
        }
      }
    }

    raw << "]";

    std::ostringstream json;
    json << "{";
    json << "\"scenario\":\"decode_baseline\",";
    json << "\"concurrency\":" << concurrency << ",";
    json << "\"prompt_length_k\":" << prompt_k << ",";
    json << "\"synthetic\":" << (synthetic ? "true" : "false") << ",";
    json << "\"model_path\":\"" << EscapeJson(model_path) << "\",";
    json << "\"execution_provider\":\"" << EscapeJson(execution_provider) << "\",";
    json << "\"generation_tokens\":" << generation_tokens << ",";
    json << "\"measured_runs\":" << measured_runs << ",";
    json << "\"status\":\"ok\",";
    json << "\"summary\":{";
    json << "\"ttft_p50_ms\":" << Percentile(ttft_values, 50.0) << ",";
    json << "\"ttft_p95_ms\":" << Percentile(ttft_values, 95.0) << ",";
    json << "\"e2e_p50_ms\":" << Percentile(e2e_values, 50.0) << ",";
    json << "\"e2e_p95_ms\":" << Percentile(e2e_values, 95.0) << ",";
    json << "\"tokens_per_s_p50\":" << Percentile(tps_values, 50.0);
    json << "},";
    json << "\"raw_requests\":" << raw.str();
    json << "}";

    *run_json = json.str();
    return true;
  } catch (const std::exception& e) {
    *error_message = e.what();
    return false;
  }
}
