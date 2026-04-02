// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

namespace test_utils {

// Helper function to get the appropriate model path based on available models
inline const std::string& GetModelPath(const std::string& model_type) {
  static std::string model_path;
  if (!model_path.empty()) {
    return model_path;
  }

  std::vector<std::string> candidate_paths = {
      std::string(MODEL_PATH) + model_type + "/int4/cuda",
      std::string(MODEL_PATH) + model_type + "/int4/dml",
      std::string(MODEL_PATH) + model_type + "/int4/webgpu",
      std::string(MODEL_PATH) + model_type + "/int4/cpu"};

  for (const auto& path : candidate_paths) {
    std::filesystem::path model_path_fs(path);
    if (std::filesystem::exists(model_path_fs / "genai_config.json")) {
      model_path = path;
      return model_path;
    }
  }

  // Fallback to CPU path
  model_path = std::string(MODEL_PATH) + model_type + "/int4/cpu";
  return model_path;
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
