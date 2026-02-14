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

// Helper function to get the appropriate PHI2 model path based on available models
inline const std::string& GetPhi2Path() {
  static std::string phi2_path;
  if (!phi2_path.empty()) {
    return phi2_path;
  }

  std::vector<std::string> candidate_paths = {
      MODEL_PATH "phi-2/int4/cuda",
      MODEL_PATH "phi-2/int4/dml",
      MODEL_PATH "phi-2/int4/webgpu",
      MODEL_PATH "phi-2/int4/cpu"};

  for (const auto& path : candidate_paths) {
    std::filesystem::path model_path(path);
    if (std::filesystem::exists(model_path / "genai_config.json")) {
      phi2_path = path;
      return phi2_path;
    }
  }

  // Fallback to CPU path
  phi2_path = MODEL_PATH "phi-2/int4/cpu";
  return phi2_path;
}

// Helper to detect if we're using WebGPU or DML EP based on the model path
inline bool IsEngineTestsEnabled() {
#if TEST_PHI2
  std::string path = GetPhi2Path();
  // Skip engine tests for DML and WebGPU (batching not fully tested)
  return path.find("/dml") == std::string::npos &&
         path.find("/webgpu") == std::string::npos;
#else
  return false;
#endif
}

}  // namespace test_utils
