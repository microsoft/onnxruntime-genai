// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "onnxruntime_c_api.h"
#include "ort_genai.h"

#include "common/benchmark_types.h"
#include "scenarios/decode_baseline.h"

namespace fs = std::filesystem;

namespace engine_benchmark {
namespace {

std::string GetOrtVersion() {
  const OrtApiBase* api = OrtGetApiBase();
  if (api == nullptr || api->GetVersionString == nullptr) {
    return "unknown";
  }
  return api->GetVersionString();
}

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
    config.generation_tokens = e.value("generation_tokens", config.generation_tokens);
    config.measured_runs = e.value("measured_runs", config.measured_runs);
    configs.push_back(std::move(config));
  }

  return configs;
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

void WriteVisualizer(const fs::path& out_dir, const std::vector<nlohmann::json>& results) {
  nlohmann::json embedded = nlohmann::json::array();
  for (const auto& result : results) {
    embedded.push_back(result);
  }

  const std::string html =
      "<!doctype html><html><head><meta charset='utf-8'><title>Benchmark Results</title>"
      "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:16px}"
      ".tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}"
      "button{padding:6px 10px;border:1px solid #999;background:#f4f4f4;cursor:pointer}"
      "button.active{background:#e1f0ff;border-color:#5da9ff}"
      "pre{background:#111;color:#ddd;padding:12px;border-radius:6px;overflow:auto}</style>"
      "</head><body><h2>Scenario Results</h2><div id='tabs' class='tabs'></div><pre id='content'></pre><script>"
      "const results = " + embedded.dump() + ";"
      "const tabs = document.getElementById('tabs'); const content = document.getElementById('content');"
      "function show(i){document.querySelectorAll('button').forEach((b,idx)=>b.classList.toggle('active', idx===i));"
      "content.textContent = JSON.stringify(results[i], null, 2);}"
      "results.forEach((r,i)=>{const b=document.createElement('button'); b.textContent=`${r.scenario} (#${String(i+1).padStart(3,'0')})`;"
      "b.onclick=()=>show(i); tabs.appendChild(b);});"
      "if(results.length>0){show(0);}"
      "</script></body></html>";

  std::ofstream out(out_dir / "visualize.html", std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open visualize.html output file.");
  }
  out << html;
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

  BenchmarkContext context;
  context.ort_version = GetOrtVersion();
  context.genai_version = GetGenAIVersion();

  std::vector<nlohmann::json> all_results;
  all_results.reserve(configs.size());

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& cfg = configs[i];

    if (cfg.scenario != "decode_baseline") {
      throw std::runtime_error("Only decode_baseline is implemented in the MVP. Found: " + cfg.scenario);
    }

    DecodeBaselineScenario scenario;
    nlohmann::json result = scenario.Run(cfg, context);

    const std::string file_name = MakeResultFilename(cfg.scenario, i + 1);
    WriteJsonFile(out_dir / file_name, result);
    all_results.push_back(std::move(result));
  }

  WriteVisualizer(out_dir, all_results);
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
