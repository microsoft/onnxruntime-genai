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
#include "scenarios/scenario_base.h"
#include "scenarios/decode_baseline.h"  // Included so its self-registration runs; not referenced directly.

namespace fs = std::filesystem;

namespace engine_benchmark {
namespace {

// versions.json is written next to the executable when the build stages the benchmark dependencies.
nlohmann::json ReadStagedVersions() {
  std::ifstream file("build/Linux/RelWithDebInfo/benchmark/engine/versions.json", std::ios::binary);
  if (!file) {
    throw std::runtime_error("versions.json not found; expected the build to stage it under benchmark/engine.");
  }

  nlohmann::json versions;
  file >> versions;
  return versions;
}

std::vector<ScenarioConfig> ParseScenarioConfigs(const nlohmann::json& root) {
  std::vector<ScenarioConfig> configs;
  try {
    configs.reserve(root.size());
    for (const auto& e : root) {
      ScenarioConfig config;
      config.scenario = e.value("scenario", config.scenario);
      config.concurrency = e.value("concurrency", config.concurrency);
      config.prompt_length_k = e.value("prompt_length_k", config.prompt_length_k);
      config.model_path = e.value("model_path", std::string{});
      config.execution_provider = e.value("execution_provider", config.execution_provider);
      config.execution_provider_library = e.value("execution_provider_library", std::string{});
      config.generation_tokens = e.value("generation_tokens", config.generation_tokens);
      config.warmup_runs = e.value("warmup_runs", config.warmup_runs);
      config.measured_runs = e.value("measured_runs", config.measured_runs);
      configs.push_back(std::move(config));
    }
  } catch (const std::exception&) {
    throw std::runtime_error("Error parsing config.");
  }

  return configs;
}

// Provider libraries can only be registered once per process, so this runs before any scenario.
void RegisterExecutionProviderLibraries(const std::vector<ScenarioConfig>& configs) {
  std::set<std::string> registered;

  for (const auto& config : configs) {
    if (config.execution_provider_library.empty()) {
      throw std::invalid_argument(
          "execution_provider_library is required when execution_provider is 'cuda' or 'webgpu'");
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

  const nlohmann::json staged_versions = ReadStagedVersions();

  BenchmarkContext context;
  context.ort_version = staged_versions.value("ort_version", std::string{"unknown"});
  context.cuda_plugin_ep_version = staged_versions.value("cuda_plugin_ep_version", std::string{"unknown"});
  context.genai_version = ORT_GENAI_BENCH_GENAI_VERSION;  // ORT_GENAI_BENCH_GENAI_VERSION is a compiler -D macro,
                                                          // set in CMakeLists.txt

  bool any_scenario_failed = false;
  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& cfg = configs[i];

    const auto scenario = ScenarioBase::Create(cfg.scenario);
    if (!scenario) {
      throw std::runtime_error("Unknown scenario: " + cfg.scenario);
    }

    const nlohmann::json result = scenario->Run(cfg, context);
    any_scenario_failed |= result.value("status", "failed") != "success";

    std::ostringstream results_file_name;
    results_file_name << cfg.scenario << "_results_" << std::setw(3) << std::setfill('0') << i + 1 << ".json";
    WriteJsonFile(out_dir / results_file_name.str(), result);
  }

  return any_scenario_failed ? 1 : 0;
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
