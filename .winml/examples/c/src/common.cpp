// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include <cassert>

void Timing::RecordStartTimestamp() {
  assert(start_timestamp_.time_since_epoch().count() == 0);
  start_timestamp_ = Clock::now();
}

void Timing::RecordFirstTokenTimestamp() {
  assert(first_token_timestamp_.time_since_epoch().count() == 0);
  first_token_timestamp_ = Clock::now();
}

void Timing::RecordEndTimestamp() {
  assert(end_timestamp_.time_since_epoch().count() == 0);
  end_timestamp_ = Clock::now();
}

void Timing::Log(const int prompt_tokens_length, const int new_tokens_length) {
  assert(start_timestamp_.time_since_epoch().count() != 0);
  assert(first_token_timestamp_.time_since_epoch().count() != 0);
  assert(end_timestamp_.time_since_epoch().count() != 0);

  Duration prompt_time = (first_token_timestamp_ - start_timestamp_);
  Duration run_time = (end_timestamp_ - first_token_timestamp_);

  const auto default_precision{std::cout.precision()};
  std::cout << std::endl;
  std::cout << "-------------" << std::endl;
  std::cout << std::fixed << std::showpoint << std::setprecision(2)
            << "Prompt length: " << prompt_tokens_length << ", New tokens: " << new_tokens_length
            << ", Time to first: " << prompt_time.count() << "s"
            << ", Prompt tokens per second: " << prompt_tokens_length / prompt_time.count() << " tps"
            << ", New tokens per second: " << new_tokens_length / run_time.count() << " tps"
            << std::setprecision(default_precision) << std::endl;
  std::cout << "-------------" << std::endl;
}

bool FileExists(const char* path) {
  return static_cast<bool>(std::ifstream(path));
}

std::string Trim(const std::string& str) {
  const size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  const size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " <model_path> <execution_provider>" << std::endl;
  std::cerr << "  model_path: [required] Path to the folder containing onnx models, genai_config.json, etc." << std::endl;
  std::cerr << "  execution_provider: [optional] Force use of a particular execution provider (e.g. \"cpu\")" << std::endl;
  std::cerr << "                      If not specified, EP / provider options specified in genai_config.json will be used." << std::endl;
}

bool parse_args(int argc, char** argv, std::string& model_path, std::string& ep) {
  if (argc < 2) {
    print_usage(argc, argv);
    return false;
  }
  model_path = argv[1];
  if (argc > 2) {
    ep = argv[2];
  } else {
    ep = "follow_config";
  }
  return true;
}

void append_provider(OgaConfig& config, const std::string& provider) {
  if (provider.compare("follow_config") != 0) {
    config.ClearProviders();
    if (provider.compare("cpu") != 0) {
      config.AppendProvider(provider.c_str());
      if (provider.compare("cuda") == 0) {
        config.SetProviderOption(provider.c_str(), "enable_cuda_graph", "0");
      }
    }
  }
}