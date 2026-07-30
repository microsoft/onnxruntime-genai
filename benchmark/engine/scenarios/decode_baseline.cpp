#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "ort_genai.h"

namespace {

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
  const size_t low = static_cast<size_t>(std::floor(rank));
  const size_t high = static_cast<size_t>(std::ceil(rank));
  if (low == high) {
    return values[low];
  }
  const double weight = rank - static_cast<double>(low);
  return values[low] * (1.0 - weight) + values[high] * weight;
}

bool IsAllowedConcurrency(int c) {
  return c == 1 || c == 2 || c == 4 || c == 8;
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string StripQueryAndTrailingSlash(std::string value) {
  const size_t query_pos = value.find('?');
  if (query_pos != std::string::npos) {
    value = value.substr(0, query_pos);
  }
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

std::string LastSegment(const std::string& uri_path) {
  const size_t slash = uri_path.find_last_of('/');
  if (slash == std::string::npos || slash + 1 >= uri_path.size()) {
    throw std::runtime_error("Invalid model path: " + uri_path);
  }
  return uri_path.substr(slash + 1);
}

bool DirectoryMissingOrEmpty(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return true;
  }
  return std::filesystem::directory_iterator(dir) == std::filesystem::directory_iterator();
}

std::string ResolveModelPath(const std::string& configured_model_path) {
  if (!(StartsWith(configured_model_path, "https://") || StartsWith(configured_model_path, "http://"))) {
    return configured_model_path;
  }

  if (configured_model_path.find(".blob.core.windows.net/") == std::string::npos) {
    throw std::runtime_error("Only Azure Blob URLs are supported for remote model pull");
  }

  std::string source_with_auth = configured_model_path;
  if (source_with_auth.find('?') == std::string::npos) {
    if (const char* sas = std::getenv("ENGINE_BENCHMARK_MODEL_SAS"); sas && *sas != '\0') {
      source_with_auth += "?";
      source_with_auth += sas;
    }
  }

  const std::string uri_no_query = StripQueryAndTrailingSlash(source_with_auth);
  const std::string model_folder = LastSegment(uri_no_query);
  const std::filesystem::path cache_root = std::filesystem::temp_directory_path() / "engine-benchmark-prototype-models";
  const std::filesystem::path local_model_dir = cache_root / model_folder;

  if (DirectoryMissingOrEmpty(local_model_dir)) {
    std::filesystem::create_directories(local_model_dir);
    std::string source_url = source_with_auth;
    if (!source_url.empty() && source_url.back() != '/') {
      source_url.push_back('/');
    }

    const std::string destination_dir = local_model_dir.string();
    const std::string command = "azcopy sync \"" + source_url + "\" \"" + destination_dir + "\" --recursive=true";
    const int exit_code = std::system(command.c_str());
    if (exit_code != 0) {
      throw std::runtime_error(
          "azcopy sync failed with exit code " + std::to_string(exit_code) +
          ". Ensure azcopy is installed, then authenticate with `azcopy login` or append a SAS token to model_path.");
    }
  }

  return local_model_dir.string();
}

std::string BuildPromptText(int prompt_k, bool synthetic) {
  const int approx_tokens = prompt_k * 1000;
  if (synthetic) {
    // Keep token-like words simple and deterministic.
    std::string out;
    out.reserve(static_cast<size_t>(approx_tokens) * 4);
    for (int i = 0; i < approx_tokens; ++i) {
      out += "tok ";
    }
    return out;
  }

  static const char* kCodingSeed =
      "You are a helpful coding assistant. Explain this C++ function and propose one optimization.\n"
      "int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2);}\n";

  std::string out;
  out.reserve(static_cast<size_t>(approx_tokens) * 6);
  while (static_cast<int>(out.size()) < approx_tokens * 4) {
    out += kCodingSeed;
  }
  return out;
}

std::string EscapeJson(std::string text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

}  // namespace

bool RunDecodeBaseline(int concurrency,
                       int prompt_k,
                       bool synthetic,
                       const std::string& model_path,
                       const std::string& execution_provider,
                       int generation_tokens,
                       int measured_runs,
                       std::string* run_json,
                       std::string* error_message) {
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
