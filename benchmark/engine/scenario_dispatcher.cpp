// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ort_genai.h"

#include "scenarios/utils.h"
#include "scenarios/decode_baseline.h"

namespace fs = std::filesystem;

namespace engine_benchmark {
namespace {

std::string GetGenAIVersion() {
#ifdef ORT_GENAI_BENCH_GENAI_VERSION
  return ORT_GENAI_BENCH_GENAI_VERSION;
#else
  return "unknown";
#endif
}

std::vector<ScenarioConfig> ParseScenarioConfigs(const nlohmann::json& root) {
  std::vector<nlohmann::json> entries;
  if (root.is_array()) {
    for (const auto& e : root) {
      entries.push_back(e);
    }
  } else if (root.contains("entries") && root["entries"].is_array()) {
    for (const auto& e : root["entries"]) {
      entries.push_back(e);
    }
  } else if (root.contains("scenarios") && root["scenarios"].is_array()) {
    for (const auto& e : root["scenarios"]) {
      entries.push_back(e);
    }
  } else if (root.is_object()) {
    entries.push_back(root);
  } else {
    throw std::runtime_error("Unsupported config.json shape.");
  }

  std::vector<ScenarioConfig> configs;
  configs.reserve(entries.size());
  for (const auto& e : entries) {
    ScenarioConfig config;
    config.scenario = e.value("scenario", config.scenario);
    config.concurrency = e.value("concurrency", config.concurrency);
    config.prompt_length_k = e.value("prompt_length_k", config.prompt_length_k);
    config.synthetic = e.value("synthetic", config.synthetic);
    config.model_path = e.value("model_path", std::string{});
    config.execution_provider = e.value("execution_provider", config.execution_provider);
    config.execution_provider_library = e.value("execution_provider_library", std::string{});
    config.generation_tokens = e.value("generation_tokens", config.generation_tokens);
    config.measured_runs = e.value("measured_runs", config.measured_runs);
    configs.push_back(std::move(config));
  }

  return configs;
}

// Provider libraries can only be registered once per process, so this runs before any scenario.
void RegisterExecutionProviderLibraries(const std::vector<ScenarioConfig>& configs) {
  std::set<std::string> registered;

  for (const auto& config : configs) {
    if (config.execution_provider != "cuda") {
      continue;
    }

    if (config.execution_provider_library.empty()) {
      throw std::invalid_argument(
          "execution_provider_library is required when execution_provider is 'cuda'");
    }

    const fs::path provider_library = fs::absolute(config.execution_provider_library);
    if (!fs::exists(provider_library)) {
      throw std::invalid_argument("execution provider library does not exist: " + provider_library.string());
    }

    if (!registered.insert("CUDAExecutionProvider").second) {
      continue;
    }

    std::cout << "[dispatcher] Registering CUDA execution provider library: "
              << provider_library.string() << std::endl;
    OgaRegisterExecutionProviderLibrary("CUDAExecutionProvider", provider_library.c_str());
  }
}

std::string MakeResultFilename(const std::string& scenario_name, size_t id) {
  std::ostringstream oss;
  oss << scenario_name << "_results_" << std::setw(3) << std::setfill('0') << id << ".json";
  return oss.str();
}

void WriteJsonFile(const fs::path& path, const nlohmann::json& json) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  out << std::setw(2) << json << "\n";
}

}  // namespace

int DispatchScenarios(const fs::path& config_path, const fs::path& out_dir) {
  OgaHandle handle;

  std::ifstream config_file(config_path, std::ios::binary);
  if (!config_file) {
    throw std::runtime_error("Failed to open config file: " + config_path.string());
  }

  nlohmann::json root;
  config_file >> root;

  const auto configs = ParseScenarioConfigs(root);
  if (configs.empty()) {
    throw std::runtime_error("No scenarios found in config.");
  }

  fs::create_directories(out_dir);

  RegisterExecutionProviderLibraries(configs);

  BenchmarkContext context;
  context.genai_version = GetGenAIVersion();

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& cfg = configs[i];

    if (cfg.scenario != "decode_baseline") {
      throw std::runtime_error("Only decode_baseline is implemented in the MVP. Found: " + cfg.scenario);
    }

    DecodeBaselineScenario scenario;
    const nlohmann::json result = scenario.Run(cfg, context);

    WriteJsonFile(out_dir / MakeResultFilename(cfg.scenario, i + 1), result);
  }

  return 0;
}

}  // namespace engine_benchmark

int main(int argc, char** argv) {
  try {
    fs::path config_path = "config.json";
    fs::path out_dir = "out";

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (arg == "--out" && i + 1 < argc) {
        out_dir = argv[++i];
      } else {
        throw std::runtime_error("Unknown argument: " + arg + ". Expected --config <path> and --out <dir>.");
      }
    }

    return engine_benchmark::DispatchScenarios(config_path, out_dir);
  } catch (const std::exception& ex) {
    std::cerr << "engine_benchmark failed: " << ex.what() << std::endl;
    return 1;
  }
}
