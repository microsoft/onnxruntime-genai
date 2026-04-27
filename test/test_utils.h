// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

namespace test_utils {

// Helper function to get the appropriate model path based on available models.
// Caches results per model_type so different models resolve independently.
inline const std::string& GetModelPath(const std::string& model_type) {
  static std::unordered_map<std::string, std::string> model_paths;
  auto it = model_paths.find(model_type);
  if (it != model_paths.end()) {
    return it->second;
  }

  std::vector<std::string> candidate_paths = {
      std::string(MODEL_PATH) + model_type + "/int4/cuda",
      std::string(MODEL_PATH) + model_type + "/int4/dml",
      std::string(MODEL_PATH) + model_type + "/int4/webgpu",
      std::string(MODEL_PATH) + model_type + "/int4/cpu"};

  for (const auto& path : candidate_paths) {
    std::filesystem::path model_path_fs(path);
    if (std::filesystem::exists(model_path_fs / "genai_config.json")) {
      return model_paths.emplace(model_type, path).first->second;
    }
  }

  // Fallback to CPU path
  return model_paths.emplace(model_type, std::string(MODEL_PATH) + model_type + "/int4/cpu").first->second;
}

// Helper to detect if we're using WebGPU or DML EP based on the model path
inline bool IsEngineTestsEnabled() {
#if TEST_PHI2
  std::string path = GetModelPath("phi-2");
  // Skip engine tests for DML and WebGPU (batching not fully tested)
  return path.find("/dml") == std::string::npos &&
         path.find("/webgpu") == std::string::npos;
#else
  return false;
#endif
}

}  // namespace test_utils

#ifndef PHI2_PATH
#define PHI2_PATH test_utils::GetModelPath("phi-2").c_str()
#endif

#ifndef QWEN_2_5_PATH
#define QWEN_2_5_PATH test_utils::GetModelPath("qwen-2.5-0.5b").c_str()
#endif
